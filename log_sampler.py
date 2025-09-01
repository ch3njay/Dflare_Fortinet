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

        except ImportError as exc:  # pragma: no cover - rarfile is optional

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



def parse_selection(options: list[tuple[str, int]], label: str) -> list[str]:
    """Prompt user to select multiple options by index."""
    sel = input(f"Enter numbers for {label} (comma-separated or blank for all): ").strip()
    if not sel:
        return [opt[0] for opt in options]
    try:
        idxs = [int(s.strip()) for s in sel.split(",") if s.strip()]
        return [options[i - 1][0] for i in idxs]
    except (ValueError, IndexError):
        raise ValueError("Invalid selection")


def balanced_sample(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Under-sample to balance the dataset by the given column."""
    counts = df[col].value_counts()
    if len(counts) < 2:
        raise ValueError(f"Balanced sampling requires at least two {col} values")
    min_count = counts.min()
    n_input = input(
        f"Enter number per {col} (max {min_count}, default {min_count}): "
    ).strip()
    n = min_count if not n_input else int(n_input)
    if n <= 0 or n > min_count:
        raise ValueError("Number per group out of range")
    return (
        df.groupby(col).apply(lambda x: x.sample(n=n)).reset_index(drop=True)
    )



def main() -> None:
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select log file",

        filetypes=[("Log files", "*.csv *.txt *.gz *.rar"), ("All files", "*.*")],

    )
    if not file_path:
        print("No file selected. Exiting.")
        return

    try:
        df = load_log_file(file_path)

    except Exception as exc:  # pragma: no cover - GUI feedback only

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

    samples: list[pd.DataFrame] = []

    while True:
        print("crlevel distribution:")
        for idx, (value, count) in enumerate(crlevel_options, start=1):
            print(f"{idx}. {value}: {count}")
        print("\nis_attack distribution:")
        for idx, (value, count) in enumerate(is_attack_options, start=1):
            print(f"{idx}. {value}: {count}")

        print("\nSelect filter option (enter to finish):")
        print("1. Filter by crlevel")
        print("2. Filter by is_attack")
        print("3. Filter by both")
        print("4. No filtering")
        choice = input("Enter choice number: ").strip()
        if not choice:
            break

        filtered = df
        try:
            if choice == "1":
                cr_vals = parse_selection(crlevel_options, "crlevel")
                filtered = filtered[filtered["crlevel"].isin(cr_vals)]
            elif choice == "2":
                atk_vals = parse_selection(is_attack_options, "is_attack")
                filtered = filtered[filtered["is_attack"].isin(atk_vals)]
            elif choice == "3":
                cr_vals = parse_selection(crlevel_options, "crlevel")
                atk_vals = parse_selection(is_attack_options, "is_attack")
                filtered = filtered[
                    filtered["crlevel"].isin(cr_vals)
                    & filtered["is_attack"].isin(atk_vals)
                ]
            elif choice == "4":
                pass
            else:
                print("Invalid choice.")
                continue
        except ValueError:
            print("Invalid selection.")
            continue

        if filtered.empty:
            print("No data matches selection.")
            continue

        print("Sampling methods:")
        print("1. Random sample")
        print("2. Balanced under-sampling by crlevel")
        print("3. Balanced under-sampling by is_attack")
        method = input("Enter method number: ").strip()

        try:
            if method == "1":
                n = int(input("Enter number of rows to sample: "))
                if n <= 0 or n > len(filtered):
                    raise ValueError
                sample_df = filtered.sample(n=n)
            elif method == "2":
                sample_df = balanced_sample(filtered, "crlevel")
            elif method == "3":
                sample_df = balanced_sample(filtered, "is_attack")
            else:
                print("Invalid method.")
                continue
        except ValueError:
            print("Invalid sample request.")
            continue

        samples.append(sample_df)
        cont = input("Add another sample? (y/n): ").strip().lower()
        if cont != "y":
            break

    if not samples:
        print("No samples collected. Exiting.")
        return

    combined = pd.concat(samples, ignore_index=True)

    save_path = filedialog.asksaveasfilename(
        title="Save sampled data",
        defaultextension=".csv",
        filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
    )
    if not save_path:

        print("No save path selected. Exiting.")
        return

    combined.to_csv(save_path, index=False)
    print(f"Sample saved to {save_path}")



if __name__ == "__main__":
    main()

