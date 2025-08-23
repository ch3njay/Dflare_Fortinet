# training_pipeliner.py
# ===== 訓練 Pipeline CLI =====
import os
import sys
import traceback

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

from training_pipeline.pipeline_main import TrainingPipeline

# --- Tkinter 檔案選擇（保留原設計與提示） ---
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    _TK_OK = True
except Exception:
    _TK_OK = False
    tk = None
    filedialog = None
    messagebox = None


def choose_file(prompt="請選擇資料檔案"):
    if not _TK_OK:
        print(f"\n[提示] {prompt}（目前環境無法使用圖形介面，請於此輸入路徑）")
        file_path = input("請輸入 CSV 檔案路徑：").strip()
        if not file_path:
            raise Exception("未選擇資料檔案！")
        return file_path

    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    try:
        messagebox.showinfo("選擇檔案", prompt, parent=root)
    except Exception:
        pass
    file_path = filedialog.askopenfilename(
        title=prompt,
        filetypes=[("CSV Files", "*.csv")],
        parent=root
    )
    try:
        root.destroy()
    except Exception:
        pass

    if not file_path:
        raise Exception("未選擇資料檔案！")
    return file_path


def ask_task_type() -> str:
    print("===== 訓練 Pipeline CLI =====")
    print("請選擇任務型態：")
    print("1. 二元分類（binary, is_attack）")
    print("2. 多元分類（multiclass, crlevel）")
    task_raw = input("請輸入 1 或 2：").strip()
    if task_raw not in {"1", "2"}:
        print("輸入錯誤，預設為 1（二元分類）")
        task_raw = "1"
    return "binary" if task_raw == "1" else "multiclass"


def ask_optuna_enabled() -> bool:
    print("\n是否要啟用超參數優化 (Optuna)？")
    print("1. 是")
    print("2. 否")
    opt_raw = input("請輸入 1 或 2：").strip()
    if opt_raw not in {"1", "2"}:
        print("輸入錯誤，預設為 2（否）")
        opt_raw = "2"
    return opt_raw == "1"


def ask_opt_scope() -> tuple[bool, bool]:
    print("\n選擇要優化的範圍：")
    print("1. 僅基模型優化（單一模型：XGB/LGBM/Cat/RF）")
    print("2. 僅集成優化（Stacking/Voting/門檻等）")
    print("3. 兩者皆是")
    print("4. 都不要（跳過優化，快速除錯）")
    sel = input("請輸入 1 / 2 / 3 / 4：").strip()
    if sel not in {"1", "2", "3", "4"}:
        print("輸入錯誤，預設為 4（都不要）")
        sel = "4"
    if sel == "1":
        return True, False
    if sel == "2":
        return False, True
    if sel == "3":
        return True, True
    return False, False


def ask_use_tuned_for_training() -> bool:
    """
    僅在 optuna_enabled=True 且 (optimize_base or optimize_ensemble)=True 時呼叫。
    詢問是否將「Optuna 最佳結果」套用到後續訓練與集成。
    """
    print("\n是否要『套用』Optuna 產出的最佳超參數到後續訓練與集成？")
    print("1. 是（用最佳參數訓練與集成）")
    print("2. 否（僅執行搜尋並輸出覆寫片段，但訓練用 config 原設定）")
    sel = input("請輸入 1 或 2：").strip()
    if sel not in {"1", "2"}:
        print("輸入錯誤，預設為 1（是）")
        sel = "1"
    return sel == "1"


def ask_ensemble_mode() -> str:
    """詢問 Optuna 集成搜尋模式。"""
    print("\n選擇 Optuna 集成搜尋模式：")
    print("1. 自由選擇子模型（free）")
    print("2. 固定候選子集（fixed）")
    sel = input("請輸入 1 或 2：").strip()
    if sel not in {"1", "2"}:
        print("輸入錯誤，預設為 1（free）")
        sel = "1"
    return "free" if sel == "1" else "fixed"


def main():
    try:
        task_type = ask_task_type()
        optuna_enabled = ask_optuna_enabled()

        if optuna_enabled:
            optimize_base, optimize_ensemble = ask_opt_scope()

            # ★ 修正點：若選了「4. 都不要」，視同完全不啟用 Optuna，且不再詢問是否套用
            if not (optimize_base or optimize_ensemble):
                optuna_enabled = False
                use_tuned_for_training = False
            else:
                use_tuned_for_training = ask_use_tuned_for_training()
                if optimize_ensemble and use_tuned_for_training:
                    ensemble_mode = ask_ensemble_mode()
                else:
                    ensemble_mode = "free"
        else:
            optimize_base, optimize_ensemble = False, False
            use_tuned_for_training = False
            ensemble_mode = "free"

        file_path = choose_file(prompt="請選擇資料檔案")

        pipeline = TrainingPipeline(
            task_type=task_type,
            optuna_enabled=optuna_enabled,
            optimize_base=optimize_base,
            optimize_ensemble=optimize_ensemble,
            use_tuned_for_training=use_tuned_for_training
        )
        pipeline.config.setdefault("ENSEMBLE_SETTINGS", {})["MODE"] = ensemble_mode
        _ = pipeline.run(file_path)

    except KeyboardInterrupt:
        print("\n使用者中止。")
    except Exception:
        print("發生未預期錯誤：")
        traceback.print_exc()


if __name__ == "__main__":
    main()
