import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog


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
        print(f"Failed to load file: {exc}")
        return

    if "crlevel" not in df.columns or "crscore" not in df.columns:
        print("File must contain 'crlevel' and 'crscore' columns.")
        return

    df["is_attack"] = (df["crscore"] > 0).astype(int)

    crlevel_counts = df["crlevel"].value_counts()
    is_attack_counts = df["is_attack"].value_counts()

    crlevel_options = list(crlevel_counts.items())
    is_attack_options = list(is_attack_counts.items())

    print("crlevel distribution:")
    for idx, (value, count) in enumerate(crlevel_options, start=1):
        print(f"{idx}. {value}: {count}")

    print("\nis_attack distribution:")
    for idx, (value, count) in enumerate(is_attack_options, start=1):
        print(f"{idx}. {value}: {count}")

    print("\nSelect filter option:")
    print("1. Filter by crlevel")
    print("2. Filter by is_attack")
    print("3. Filter by both")
    print("4. No filtering")
    choice = input("Enter choice number: ").strip()

    filtered = df
    if choice == "1":
        try:
            sel = int(input("Enter number for crlevel value: "))
            crlevel_value = crlevel_options[sel - 1][0]
        except (ValueError, IndexError):
            print("Invalid selection.")
            return
        filtered = filtered[filtered["crlevel"] == crlevel_value]
    elif choice == "2":
        try:
            sel = int(input("Enter number for is_attack value: "))
            is_attack_value = is_attack_options[sel - 1][0]
        except (ValueError, IndexError):
            print("Invalid selection.")
            return
        filtered = filtered[filtered["is_attack"] == is_attack_value]
    elif choice == "3":
        try:
            sel1 = int(input("Enter number for crlevel value: "))
            crlevel_value = crlevel_options[sel1 - 1][0]
            sel2 = int(input("Enter number for is_attack value: "))
            is_attack_value = is_attack_options[sel2 - 1][0]
        except (ValueError, IndexError):
            print("Invalid selection.")
            return
        filtered = filtered[
            (filtered["crlevel"] == crlevel_value)
            & (filtered["is_attack"] == is_attack_value)
        ]
    elif choice == "4":
        pass
    else:
        print("Invalid choice.")
        return

    try:
        sample_size = int(input("Enter number of rows to sample: "))
    except ValueError:
        print("Invalid sample size.")
        return
    if sample_size <= 0 or sample_size > len(filtered):
        print("Sample size must be between 1 and number of selected rows.")
        return

    sample_df = filtered.sample(n=sample_size)

    save_path = filedialog.asksaveasfilename(
        title="Save sampled data",
        defaultextension=".csv",
        filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
    )
    if not save_path:
        print("No save path selected. Exiting.")
        return

    sample_df.to_csv(save_path, index=False)
    print(f"Sample saved to {save_path}")


if __name__ == "__main__":
    main()

