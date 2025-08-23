"""
ModelBuilder — 根據 config 超參數建模，支援 Optuna 超參數搜尋
相容介面：
  - build_models(X, y, task="binary", params=None)
  - build_models(best_params)   # 先 run_optuna 再建模的舊用法
  - run_optuna(X, y, task)
"""

from __future__ import annotations
from typing import Dict, Optional

import warnings
import numpy as np
import optuna
from optuna.pruners import MedianPruner

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


class ModelBuilder:
    def __init__(
        self,
        config: dict,
        use_optuna: bool = False,
        max_trials_xgb: int = 15,
        max_trials_others: int = 10,
        enable_pruner: bool = True,
    ) -> None:
        self.config = config
        self.use_optuna = use_optuna
        self.max_trials_xgb = max_trials_xgb
        self.max_trials_others = max_trials_others
        self.pruner = MedianPruner() if enable_pruner else None

        # -------- 降噪（不影響錯誤拋出）---------
        warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
        warnings.filterwarnings("ignore", message=".*gpu_hist.*deprecated.*")
        warnings.filterwarnings("ignore", message='.*Parameters: { "predictor" } are not used.*')
        warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")
        warnings.filterwarnings("ignore", message=".*No further splits with positive gain.*")

    # ============================================================
    # Public APIs
    # ============================================================

    def build_models(
        self,
        X=None,
        y=None,
        task: str = "binary",
        params: Optional[Dict[str, dict]] = None,
    ) -> Dict[str, object]:
        """
        1) build_models(best_params)  # 若第一參數就是 dict 則視為 tuned params
        2) build_models(X, y, task="binary", params=None)  # 依 config/optuna 建模
        """
        tuned_params: Optional[Dict[str, dict]] = None

        # 舊用法：直接丟 best_params
        if isinstance(X, dict) and y is None:
            tuned_params = X  # type: ignore[assignment]
            X, y = None, None
        else:
            # 新用法：內部決定是否要 run_optuna
            if self.use_optuna and X is not None and y is not None:
                tuned_params = self.run_optuna(X, y, task)
            else:
                tuned_params = params

        base_params = self.config.get("MODEL_PARAMS", {})
        task_args = self._task_specific_args(y, task)

        models = {
            "XGB": XGBClassifier(**self._merge_xgb_params(base_params.get("XGB", {}), tuned_params, task_args)),
            "LGB": LGBMClassifier(**self._merge_lgb_params(base_params.get("LGB", {}), tuned_params, task_args, y, task)),
            "CAT": CatBoostClassifier(**self._merge_cat_params(base_params.get("CAT", {}), tuned_params, task_args)),
            "RF": RandomForestClassifier(**self._merge_tree_params("RF", base_params, tuned_params)),
            "ET": ExtraTreesClassifier(**self._merge_tree_params("ET", base_params, tuned_params)),
        }

        for name, est in models.items():
            if not hasattr(est, "fit"):
                raise TypeError(f"❌ Model {name} 建立失敗：不是 estimator（取得 {type(est)}）")

        return models

    def run_optuna(self, X, y, task_type: str = "binary") -> Dict[str, dict]:
        """
        對所有模型進行尋參。任何單折失敗不會中止整個 study（以 NaN 計分→當作低分）。
        """
        best_params: Dict[str, dict] = {}
        rng = self.config.get("RANDOM_STATE", 42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng)
        pruner = self.pruner

        # 為避免多進程把裝置警告刷爆，CV 階段固定 n_jobs=1（模型內仍可 n_jobs=-1）
        cv_n_jobs = 1

        def _safe_cv_score(model) -> float:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 修正點：error_score 必須是 float（或 'raise'），給 np.nan
                scores = cross_val_score(
                    model,
                    X,
                    y,
                    cv=cv,
                    scoring="accuracy",
                    n_jobs=cv_n_jobs,
                    error_score=np.nan,
                )
            m = np.nanmean(scores)
            return 0.0 if np.isnan(m) else float(m)

        # ============== XGBoost =================
        def xgb_objective(trial):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 80, 220),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "tree_method": "hist",
                    "device": "cuda",
                    "predictor": "gpu_predictor",
                    "random_state": rng,
                    "n_jobs": -1,
                }
                if task_type == "binary":
                    params.update({"objective": "binary:logistic", "eval_metric": "logloss"})
                else:
                    params.update(
                        {
                            "objective": "multi:softprob",
                            "eval_metric": "mlogloss",
                            "num_class": int(np.unique(y).shape[0]),
                        }
                    )
                return _safe_cv_score(XGBClassifier(**params))

        if self.use_optuna:
            print("🔍 Optuna 搜尋 XGBoost ...")
            study_xgb = optuna.create_study(direction="maximize", pruner=pruner)
            study_xgb.optimize(xgb_objective, n_trials=self.max_trials_xgb, show_progress_bar=True)
            best_params["XGB"] = {
                **study_xgb.best_params,
                "tree_method": "hist",
                "device": "cuda",
                "predictor": "gpu_predictor",
                "n_jobs": -1,
                "random_state": rng,
                **(
                    {"objective": "binary:logistic", "eval_metric": "logloss"}
                    if task_type == "binary"
                    else {
                        "objective": "multi:softprob",
                        "eval_metric": "mlogloss",
                        "num_class": int(np.unique(y).shape[0]),
                    }
                ),
            }
        else:
            best_params["XGB"] = self.config.get("MODEL_PARAMS", {}).get("XGB", {})

        # ============== LightGBM =================
        def lgb_objective(trial):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                max_depth = trial.suggest_int("max_depth", 4, 12)
                max_leaves = (1 << max_depth) - 1
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 150, 500),
                    "max_depth": max_depth,
                    "num_leaves": min(trial.suggest_int("num_leaves", 31, 255), max_leaves),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 500),
                    "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-2, 1.0, log=True),
                    "min_gain_to_split": trial.suggest_float("min_gain_to_split", 1e-3, 1.0, log=True),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
                    "bagging_freq": trial.suggest_int("bagging_freq", 0, 5),
                    "max_bin": trial.suggest_int("max_bin", 127, 255),
                    "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
                    "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
                    "device_type": "gpu",  # 新版參數名
                    "verbosity": -1,
                    "n_jobs": -1,
                }
                if task_type == "binary":
                    params.update({"objective": "binary", "metric": "binary_logloss"})
                else:
                    params.update(
                        {
                            "objective": "multiclass",
                            "metric": "multi_logloss",
                            "num_class": int(np.unique(y).shape[0]),
                        }
                    )
                return _safe_cv_score(LGBMClassifier(**params))

        if self.use_optuna:
            print("🔍 Optuna 搜尋 LightGBM ...")
            try:
                study_lgb = optuna.create_study(direction="maximize", pruner=pruner)
                study_lgb.optimize(lgb_objective, n_trials=self.max_trials_others, show_progress_bar=True)
                best_params["LGB"] = {
                    **study_lgb.best_params,
                    "device_type": "gpu",
                    "verbosity": -1,
                    "n_jobs": -1,
                    **(
                        {"objective": "binary", "metric": "binary_logloss"}
                        if task_type == "binary"
                        else {
                            "objective": "multiclass",
                            "metric": "multi_logloss",
                            "num_class": int(np.unique(y).shape[0]),
                        }
                    ),
                }
            except Exception as e:
                print(f"[LightGBM] Optuna 失敗，改用安全預設。原因：{e}")
                best_params["LGB"] = self._safe_lgb_defaults(y, task_type)
        else:
            best_params["LGB"] = self._fix_lgb_device(self.config.get("MODEL_PARAMS", {}).get("LGB", {}), y, task_type)

        # ============== CatBoost =================
        def cat_objective(trial):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = {
                    "iterations": trial.suggest_int("iterations", 200, 500),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                    "task_type": "GPU",
                    "devices": "0",
                    "verbose": 0,
                }
                if task_type != "binary":
                    params["loss_function"] = "MultiClass"
                return _safe_cv_score(CatBoostClassifier(**params))

        if self.use_optuna:
            print("🔍 Optuna 搜尋 CatBoost ...")
            study_cat = optuna.create_study(direction="maximize", pruner=pruner)
            study_cat.optimize(cat_objective, n_trials=self.max_trials_others, show_progress_bar=True)
            best_params["CAT"] = {
                **study_cat.best_params,
                "task_type": "GPU",
                "devices": "0",
                "verbose": 0,
                **({} if task_type == "binary" else {"loss_function": "MultiClass"}),
            }
        else:
            best_params["CAT"] = self.config.get("MODEL_PARAMS", {}).get("CAT", {})

        # ============== RF =================
        def rf_objective(trial):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 120, 300),
                    "max_depth": trial.suggest_int("max_depth", 4, 12),
                    "n_jobs": -1,
                }
                return _safe_cv_score(RandomForestClassifier(**params))

        if self.use_optuna:
            print("🔍 Optuna 搜尋 RandomForest ...")
            study_rf = optuna.create_study(direction="maximize", pruner=pruner)
            study_rf.optimize(rf_objective, n_trials=self.max_trials_others, show_progress_bar=True)
            best_params["RF"] = {**study_rf.best_params, "n_jobs": -1}
        else:
            best_params["RF"] = self.config.get("MODEL_PARAMS", {}).get("RF", {})

        # ============== ET =================
        def et_objective(trial):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 120, 300),
                    "max_depth": trial.suggest_int("max_depth", 4, 12),
                    "n_jobs": -1,
                }
                return _safe_cv_score(ExtraTreesClassifier(**params))

        if self.use_optuna:
            print("🔍 Optuna 搜尋 ExtraTrees ...")
            study_et = optuna.create_study(direction="maximize", pruner=pruner)
            study_et.optimize(et_objective, n_trials=self.max_trials_others, show_progress_bar=True)
            best_params["ET"] = {**study_et.best_params, "n_jobs": -1}
        else:
            best_params["ET"] = self.config.get("MODEL_PARAMS", {}).get("ET", {})

        return best_params

    # ============================================================
    # Internal helpers
    # ============================================================

    def _task_specific_args(self, y, task: str) -> Dict[str, dict]:
        args = {"XGB": {}, "LGB": {}, "CAT": {}}
        if task == "binary":
            args["XGB"] = {"objective": "binary:logistic", "eval_metric": "logloss"}
            args["LGB"] = {"objective": "binary", "metric": "binary_logloss"}
            args["CAT"] = {}
        else:
            ncls = int(np.unique(y).shape[0]) if y is not None else None
            if ncls is None or ncls <= 1:
                raise ValueError("多元分類需要 y 提供至少兩個以上類別。")
            args["XGB"] = {"objective": "multi:softprob", "eval_metric": "mlogloss", "num_class": ncls}
            args["LGB"] = {"objective": "multiclass", "metric": "multi_logloss", "num_class": ncls}
            args["CAT"] = {"loss_function": "MultiClass"}
        return args

    def _safe_lgb_defaults(self, y, task_type: str) -> dict:
        """LGB 的安全預設（在 Optuna 崩潰或未啟用時使用）"""
        p = {
            "max_depth": 8,
            "num_leaves": 127,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "min_data_in_leaf": 50,
            "min_sum_hessian_in_leaf": 1e-2,
            "min_gain_to_split": 1e-3,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "max_bin": 255,
            "lambda_l1": 0.0,
            "lambda_l2": 0.0,
            "device_type": "gpu",
            "verbosity": -1,
            "n_jobs": -1,
        }
        if task_type == "binary":
            p.update({"objective": "binary", "metric": "binary_logloss"})
        else:
            p.update({"objective": "multiclass", "metric": "multi_logloss", "num_class": int(np.unique(y).shape[0])})
        return p

    def _fix_lgb_device(self, params: dict, y, task_type: str) -> dict:
        """把舊 config 的 device='gpu' 改為 device_type='gpu'，並補上安全預設。"""
        p = dict(params)
        if p.get("device", None) == "gpu":
            p["device_type"] = "gpu"
            p.pop("device", None)

        p.setdefault("verbosity", -1)
        p.setdefault("n_jobs", -1)
        p.setdefault("max_bin", 255)
        p.setdefault("min_data_in_leaf", 50)
        p.setdefault("min_sum_hessian_in_leaf", 1e-2)
        p.setdefault("min_gain_to_split", 1e-3)
        p.setdefault("feature_fraction", 0.9)
        p.setdefault("bagging_fraction", 0.9)
        p.setdefault("bagging_freq", 1)

        if p.get("max_depth", -1) == -1:
            p["max_depth"] = 8
        md = p.get("max_depth", 8)
        if md and md > 0:
            p["num_leaves"] = min(p.get("num_leaves", 127), (1 << md) - 1)

        if task_type == "binary":
            p.setdefault("objective", "binary")
            p.setdefault("metric", "binary_logloss")
        else:
            p.setdefault("objective", "multiclass")
            p.setdefault("metric", "multi_logloss")
            ncls = int(np.unique(y).shape[0]) if y is not None else None
            if ncls:
                p.setdefault("num_class", ncls)

        return p

    def _merge_xgb_params(self, base: dict, tuned: Optional[Dict[str, dict]], task_args: Dict[str, dict]) -> dict:
        p = dict(base)
        if tuned and "XGB" in tuned:
            p.update(tuned["XGB"])
        p.setdefault("tree_method", "hist")
        p.setdefault("device", "cuda")
        p.setdefault("predictor", "gpu_predictor")
        p.setdefault("n_jobs", -1)
        p.setdefault("random_state", self.config.get("RANDOM_STATE", 42))
        p.update(task_args["XGB"])
        return p

    def _merge_lgb_params(
        self,
        base: dict,
        tuned: Optional[Dict[str, dict]],
        task_args: Dict[str, dict],
        y,
        task: str,
    ) -> dict:
        p = self._fix_lgb_device(base, y, task)
        if tuned and "LGB" in tuned:
            p.update(tuned["LGB"])
        p.update(task_args["LGB"])
        return p

    def _merge_cat_params(self, base: dict, tuned: Optional[Dict[str, dict]], task_args: Dict[str, dict]) -> dict:
        p = dict(base)
        if tuned and "CAT" in tuned:
            p.update(tuned["CAT"])
        p.setdefault("task_type", "GPU")
        p.setdefault("devices", "0")
        p.setdefault("verbose", 0)
        p.update(task_args["CAT"])
        return p

    def _merge_tree_params(self, key: str, base_params: Dict[str, dict], tuned: Optional[Dict[str, dict]]) -> dict:
        p = dict(base_params.get(key, {}))
        if tuned and key in tuned:
            p.update(tuned[key])
        p.setdefault("n_jobs", -1)
        return p
