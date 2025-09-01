import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog


def load_log_file(path: str) -> pd.DataFrame:
    """Load a log file that may be csv/txt/gz/rar."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".rar":
        try:
            import rarfile  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Reading .rar files requires the 'rarfile' package."
            ) from exc
        with rarfile.RarFile(path) as rf:
            inner = rf.namelist()[0]
            with rf.open(inner) as f:
                return pd.read_csv(f, sep=None, engine="python")
    else:
        # pandas can infer compression for .gz
        return pd.read_csv(path, sep=None, engine="python")


def main() -> None:
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select log file",
        filetypes=[
            ("Log files", "*.csv *.txt *.gz *.rar"),
            ("All files", "*.*"),
        ],
    )
    if not file_path:
        print("No file selected. Exiting.")
        return

    try:
        df = load_log_file(file_path)
    except Exception as exc:
        messagebox.showerror("Error", f"Failed to load file: {exc}")
        return

    if "crlevel" not in df.columns or "crscore" not in df.columns:
        messagebox.showerror(
            "Missing columns",
            "File must contain 'crlevel' and 'crscore' columns.",
        )
        return

    df["is_attack"] = (df["crscore"] > 0).astype(int)

    crlevel_counts = df["crlevel"].value_counts()
    is_attack_counts = df["is_attack"].value_counts()

    print("crlevel distribution:")
    print(crlevel_counts)
    print("\nis_attack distribution:")
    print(is_attack_counts)

    if not messagebox.askyesno("Sampling", "Do you want to sample the data?"):
        return

    crlevel_value = simpledialog.askstring(
        "crlevel", "Enter crlevel value to filter (blank for all):"
    )
    is_attack_value = simpledialog.askstring(
        "is_attack", "Enter is_attack value (0 or 1, blank for all):"
    )
    sample_size = simpledialog.askinteger(
        "Sample size", "Enter number of rows to sample:", minvalue=1
    )

    filtered = df
    if crlevel_value:
        filtered = filtered[filtered["crlevel"].astype(str) == str(crlevel_value)]
    if is_attack_value in {"0", "1"}:
        filtered = filtered[filtered["is_attack"] == int(is_attack_value)]

    if sample_size is None or sample_size > len(filtered):
        messagebox.showerror(
            "Invalid sample size",
            "Sample size exceeds available rows or was cancelled.",
        )
        return

    sample_df = filtered.sample(n=sample_size)

    save_path = filedialog.asksaveasfilename(
        title="Save sampled data",
        defaultextension=".csv",
        filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
    )
    if not save_path:
        messagebox.showinfo("Cancelled", "No save path selected.")
        return

    sample_df.to_csv(save_path, index=False)
    messagebox.showinfo("Success", f"Sample saved to {save_path}")


if __name__ == "__main__":
    main()
