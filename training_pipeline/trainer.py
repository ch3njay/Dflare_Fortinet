# training_pipeline/trainer.py
from __future__ import annotations

import numpy as np
import warnings
from typing import Dict, Any, Tuple

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

# 靜音部分無關緊要的警告，保留你原有輸出風格
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class Trainer:
    """
    訓練器：逐一訓練 models dict 內的模型。
    - 保留原有印出格式與語氣。
    - 對 LightGBM 加入多層級「安全參數」與資料清理，避免 left_count==0 的 Fatal。
    - 全程不改你的互動方式與 CLI。
    """

    def __init__(self) -> None:
        pass

    # ===================== Public API =====================
    def train(self, models: Dict[str, Any], X, y) -> Dict[str, Any]:
        fitted = {}
        for name, est in models.items():
            print(f"🏋️  訓練模型：{name}")
            fitted[name] = self._fit_one(name, est, X, y)
        return fitted

    # ===================== Internal Helpers =====================
    def _fit_one(self, name: str, est, X, y):
        """
        - 統一先做資料健檢與清理（對所有模型一致）。
        - 對 LGBM 額外做多層級 fallback。
        """
        X_pre, y_clean = self._sanitize_xy(X, y)

        # LightGBM：套用安全訓練流程
        if self._is_lgbm(est):
            # level 1
            safe_est = self._make_lgb_safe(est, level=1)
            print("🛡️  [LGB] 使用 LightGBM 安全參數訓練（避免空子節點 Fatal）")
            try:
                return self._fit_silent(safe_est, X_pre, y_clean)
            except Exception as e1:
                print(f"\n⚠️  [LGB] 安全參數仍失敗，升級保守度後再試。錯誤：{e1}\n")

            # level 2
            safer_est = self._make_lgb_safe(est, level=2)
            try:
                return self._fit_silent(safer_est, X_pre, y_clean)
            except Exception as e2:
                print(f"\n⚠️  [LGB] 第二層保守參數仍失敗，嘗試最嚴格模式（row-wise + 更小 max_bin）。錯誤：{e2}\n")

            # level 3（最嚴格）
            safest_est = self._make_lgb_safe(est, level=3)
            try:
                return self._fit_silent(safest_est, X_pre, y_clean)
            except Exception as e3:
                # 到這層仍失敗，回拋例外，讓上層顯示一致的「發生未預期錯誤」
                raise e3

        # 其他模型：原樣訓練
        return self._fit_silent(est, X_pre, y_clean)

    def _fit_silent(self, est, X, y):
        # 維持 sklearn 風格，避免雜訊輸出
        est.fit(X, y)
        return est

    # ===================== LightGBM Utilities =====================
    def _is_lgbm(self, est) -> bool:
        cls = est.__class__.__name__.lower()
        return ("lgbm" in cls) or ("lightgbm" in cls)

    def _make_lgb_safe(self, est, level: int = 1):
        """
        多層級防呆設定，僅在參數存在時才設定，不會破壞你原本傳入參數。
        目標：避免「best_split_info.left_count == 0」的 Fatal。
        """
        m = clone(est)
        params = m.get_params()

        def set_if_has(**pairs):
            exist = {k: v for k, v in pairs.items() if k in params}
            if exist:
                m.set_params(**exist)

        # 通用安全項（各層共享）
        # - CPU + 決定式 + 行為收斂
        set_if_has(n_jobs=1)
        if "device_type" in params:
            set_if_has(device_type="cpu")
        elif "device" in params:
            set_if_has(device="cpu")

        # LightGBM sklearn API 的常見鍵
        for k in ("verbosity", "force_col_wise", "deterministic", "zero_as_missing",
                  "feature_pre_filter", "enable_bundle", "max_bin", "min_data_in_bin",
                  "min_data_in_leaf", "min_sum_hessian_in_leaf", "min_gain_to_split",
                  "num_leaves", "bagging_freq", "bagging_fraction",
                  "feature_fraction", "lambda_l1", "lambda_l2", "max_depth"):
            if k not in params:
                params[k] = None  # 讓 set_if_has 可以辨識

        # 針對類別不平衡（binary 才生效；multiclass 忽略）
        if "is_unbalance" in params:
            set_if_has(is_unbalance=True)

        # Level 1：溫和
        if level == 1:
            set_if_has(
                verbosity=-1,
                force_col_wise=True,            # column-wise 通常較穩
                deterministic=True,
                zero_as_missing=True,
                feature_pre_filter=False,       # 關閉預過濾，避免早期無效分裂
                enable_bundle=False,            # 關閉特徵打包
                max_bin=127,                    # 預設 255，降低一半
                min_data_in_bin=5,              # 每個 bin 至少樣本數
                min_data_in_leaf=50,            # 葉節點最少樣本數
                min_sum_hessian_in_leaf=1e-3,   # 避免 hessian 太小
                min_gain_to_split=0.0,
                num_leaves=31,
                bagging_freq=0,
                bagging_fraction=1.0,
                feature_fraction=1.0,
                lambda_l1=0.0,
                lambda_l2=0.0,
                max_depth=-1,
            )
            return m

        # Level 2：保守
        if level == 2:
            set_if_has(
                verbosity=-1,
                force_col_wise=True,
                deterministic=True,
                zero_as_missing=True,
                feature_pre_filter=False,
                enable_bundle=False,
                max_bin=63,                     # 再縮小分箱
                min_data_in_bin=15,
                min_data_in_leaf=200,
                min_sum_hessian_in_leaf=1e-2,   # 加強 hessian 下限
                min_gain_to_split=1e-8,
                num_leaves=15,
                bagging_freq=0,
                bagging_fraction=1.0,
                feature_fraction=0.8,           # 略降特徵比例
                lambda_l1=1e-3,
                lambda_l2=1e-2,
                max_depth=8,
            )
            return m

        # Level 3：最嚴格（且切換 row-wise 算法）
        if level == 3:
            # 注意：有些版本 key 為 force_row_wise，有些沒有；set_if_has 會自動忽略不存在的鍵。
            if "force_row_wise" in params:
                m.set_params(force_row_wise=True)

            set_if_has(
                verbosity=-1,
                deterministic=True,
                zero_as_missing=True,
                feature_pre_filter=False,
                enable_bundle=False,
                max_bin=31,                     # 最小化分箱數，降低「空子節點」機率
                min_data_in_bin=30,
                min_data_in_leaf=1000,          # 葉節點更大，保證雙側有人
                min_sum_hessian_in_leaf=5e-2,   # 明顯提高 hessian 下限
                min_gain_to_split=0.0,
                num_leaves=7,
                bagging_freq=0,
                bagging_fraction=1.0,
                feature_fraction=0.6,
                lambda_l1=1e-2,
                lambda_l2=1e-1,
                max_depth=6,
            )
            return m

        return m

    # ===================== Data Sanitation =====================
    def _sanitize_xy(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        """
        - 轉為 float32 / int64（視 y 而定）
        - 將 inf / -inf 轉為 nan，再以 0.0 填補
        - 刪除全常數欄（對 LGB/XGB 等是安全的；不影響你下游互動）
        - 保留輸出與互動風格，不額外打印
        """
        # 轉為 ndarray
        Xv = X.values if hasattr(X, "values") else X
        yv = y.values if hasattr(y, "values") else y

        Xv = np.asarray(Xv)
        yv = np.asarray(yv)

        # dtype 處理
        if Xv.dtype != np.float32:
            Xv = Xv.astype(np.float32, copy=False)

        # 無窮值 -> NaN -> 0
        Xv[~np.isfinite(Xv)] = np.nan
        # 若整列皆 NaN，LightGBM 會自行忽略；這裡統一填 0，避免分箱全空
        nan_mask = np.isnan(Xv)
        if nan_mask.any():
            Xv[nan_mask] = 0.0

        # 刪除全常數欄（防資訊洩漏；同時減少分裂異常）
        if Xv.ndim == 2:
            col_max = np.nanmax(Xv, axis=0)
            col_min = np.nanmin(Xv, axis=0)
            const_cols = (col_max == col_min)
            if const_cols.any():
                Xv = Xv[:, ~const_cols]

        # y 型別
        if not np.issubdtype(yv.dtype, np.integer):
            # 多數分類標籤可安全轉 int64
            try:
                yv = yv.astype(np.int64, copy=False)
            except Exception:
                # 若不可轉，保留原 dtype
                pass

        return Xv, yv
 