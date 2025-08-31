import os
import time
import io
import re
import contextlib
from pathlib import Path

import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

try:
    from streamlit_autorefresh import st_autorefresh
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    st_autorefresh = None

from etl_pipeliner import run_pipeline
from etl_pipeline import log_cleaning as LC
from notifier import notify_from_csv


def _rerun() -> None:
    """Trigger a Streamlit rerun across versions."""
    rerun = getattr(st, "rerun", getattr(st, "experimental_rerun", None))
    if rerun is not None:  # pragma: no branch - either rerun or experimental_rerun
        rerun()

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except Exception:  # pragma: no cover - watchdog may not be installed
    Observer = None
    FileSystemEventHandler = object

try:  # pragma: no cover - tkinter may not be available
    import tkinter as tk
    from tkinter import filedialog
except Exception:  # pragma: no cover - running without GUI support
    tk = None
    filedialog = None

class _FileMonitorHandler(FileSystemEventHandler):
    """Watchdog handler that records supported file events."""

    SUPPORTED_EXTS = (
        ".csv",
        ".txt",
        ".log",
        ".csv.gz",
        ".txt.gz",
        ".log.gz",
        ".zip",
    )

    def __init__(self):
        self.events = []

    def _track(self, event_type: str, path: str) -> None:
        """Record events for supported files regardless of case."""
        if path.lower().endswith(self.SUPPORTED_EXTS):
            self.events.append((event_type, path))

    def on_created(self, event):  # pragma: no cover - requires filesystem events
        if not event.is_directory:
            self._track("created", event.src_path)

    def on_modified(self, event):  # pragma: no cover - requires filesystem events
        if not event.is_directory:
            self._track("modified", event.src_path)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _log_toast(msg: str) -> None:
    """Append *msg* to log and show a toast if supported."""
    st.session_state.log_lines.append(msg)
    if hasattr(st, "toast"):
        st.toast(msg)
    else:  # pragma: no cover - toast not available
        st.write(msg)


def _run_etl_and_infer(path: str, progress_bar) -> None:
    """Run ETL pipeline and model inference on *path*."""
    bin_model = st.session_state.get("binary_model")
    mul_model = st.session_state.get("multi_model")
    if not (bin_model and mul_model):
        st.session_state.log_lines.append("Models not uploaded; skipping")
        return

    p = Path(path)
    while p.suffix in {".gz", ".zip"}:
        p = p.with_suffix("")

    ext = p.suffix.lower()
    stem = p.stem.lower()

    clean_csv = path
    do_map = True
    do_fe = True

    if ext in {".txt", ".log"}:
        clean_csv = str(p.with_name(p.stem + "_clean.csv"))

        _log_toast("Running cleaning for raw log")
        LC.clean_logs(quiet=True, paths=[path], clean_csv=clean_csv)
    else:
        clean_csv = path
        if stem.endswith("_engineered"):
            do_map = False
            do_fe = False
        elif stem.endswith("_preprocessed"):
            do_map = False
            do_fe = True

    base = p.with_suffix("")
    pre_csv = clean_csv if not do_map else f"{base}_preprocessed.csv"
    fe_csv = pre_csv if not do_fe else f"{base}_engineered.csv"

    try:
        _log_toast(f"Detected new file: {path}")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_pipeline(
                do_clean=False,
                do_map=do_map,
                do_fe=do_fe,
                clean_out=clean_csv,
                preproc_out=pre_csv,
                fe_out=fe_csv,
            )
        for line in ANSI_RE.sub("", buf.getvalue()).splitlines():
            if line.strip():
                st.session_state.log_lines.append(line.strip())



        # feature engineered data for model inference

        df = pd.read_csv(fe_csv)
        if df.isna().any().any():
            _log_toast("Detected NaNs; filling with 0")
            df.fillna(0, inplace=True)


        # original data retained for notification context
        raw_df = pd.read_csv(clean_csv)
        if raw_df.isna().any().any():
            fill_values = {
                col: 0 if pd.api.types.is_numeric_dtype(raw_df[col]) else ""
                for col in raw_df.columns
            }
            raw_df.fillna(value=fill_values, inplace=True)


        features = [c for c in df.columns if c not in {"is_attack", "crlevel"}]

        bin_clf = bin_model
        bin_features = list(getattr(bin_clf, "feature_names_in_", features))
        missing = [f for f in bin_features if f not in df.columns]
        if missing:
            _log_toast(f"Missing features for binary model: {missing}; filling with 0")
        df_bin = df.reindex(columns=bin_features, fill_value=0)

        _log_toast("Running binary classification")
        bin_pred = bin_clf.predict(df_bin)
        result = raw_df.copy()

        result["is_attack"] = bin_pred
        result["crlevel"] = 0
        mask = result["is_attack"] == 1
        _log_toast(f"Binary classification found {mask.sum()} attack rows")
        if mask.any():
            mul_clf = mul_model
            mul_features = list(getattr(mul_clf, "feature_names_in_", features))
            missing_mul = [f for f in mul_features if f not in df.columns]
            if missing_mul:
                _log_toast(
                    f"Missing features for multiclass model: {missing_mul}; filling with 0"
                )
            df_mul = df.loc[mask].reindex(columns=mul_features, fill_value=0)

            _log_toast("Running multiclass classification for attack rows")
            result.loc[mask, "crlevel"] = mul_clf.predict(df_mul)
        else:
            _log_toast("No attacks detected; skipping multiclass classification")


        report_path = f"{base}_report.csv"
        result.to_csv(report_path, index=False)
        gen_files = {report_path}
        for f in (clean_csv, pre_csv, fe_csv):
            if f != path:
                gen_files.add(f)
        st.session_state.generated_files.update(gen_files)
        webhook = st.session_state.get("discord_webhook", "")
        gemini_key = st.session_state.get("gemini_key", "")
        line_token = st.session_state.get("line_token", "")


        def _log(msg: str) -> None:
            st.session_state.log_lines.append(msg)
            st.write(msg)

        notify_from_csv(
            report_path,
            webhook,
            gemini_key,
            risk_levels={"3", "4"},
            ui_log=_log,
            line_token=line_token,
        )

        # store counts for visualization
        st.session_state.last_counts = {
            "is_attack": result["is_attack"].value_counts().reindex([0, 1], fill_value=0),
            "crlevel": result["crlevel"].value_counts().reindex([0, 1, 2, 3, 4], fill_value=0),
        }
        st.session_state.last_critical = result[result["crlevel"] >= 4]

        _log_toast(f"Processed {path} -> {report_path}")
        for pct in range(0, 101, 20):
            progress_bar.progress(pct)
            time.sleep(0.05)
    except Exception as exc:  # pragma: no cover - processing errors
        _log_toast(f"Processing failed {path}: {exc}")


def _cleanup_generated(hours: int, *, force: bool = False) -> None:
    """Remove generated files older than *hours* or all if *force* is True."""
    now = time.time()
    to_remove = []
    for f in list(st.session_state.get("generated_files", set())):
        try:
            if force or now - os.path.getmtime(f) > hours * 3600:
                os.remove(f)
                to_remove.append(f)
        except OSError:
            to_remove.append(f)
    for f in to_remove:
        st.session_state.generated_files.discard(f)
        st.session_state.log_lines.append(f"Removed {f}")


def _process_events(handler: _FileMonitorHandler, progress_bar) -> None:
    """Process newly detected files."""
    new_events = handler.events[len(st.session_state.get("processed_events", [])) :]
    for _, path in new_events:
        if path in st.session_state.get("generated_files", set()):
            continue
        try:
            if time.time() - os.path.getmtime(path) < 5:
                continue
        except OSError:
            continue
        if path in st.session_state.get("processed_files", set()):
            continue
        _run_etl_and_infer(path, progress_bar)
        st.session_state.setdefault("processed_files", set()).add(path)
    st.session_state.processed_events = handler.events[:]

def app() -> None:
    st.title("Folder Monitor")
    st.info(
        "Select a folder to monitor for CSV/TXT/log files including compressed "
        "formats (.gz, .zip). Files are processed after 5 seconds of inactivity, "
        "and only new data is read to conserve memory."
    )

    if "folder" not in st.session_state:
        st.session_state.folder = os.getcwd()

    # separate widget value to allow programmatic updates without Streamlit errors
    if "folder_input" not in st.session_state:
        st.session_state.folder_input = st.session_state.folder

    if "observer" not in st.session_state:
        st.session_state.observer = None
        st.session_state.handler = None
    st.session_state.setdefault("log_lines", [])
    st.session_state.setdefault("processed_events", [])
    st.session_state.setdefault("generated_files", set())

    col1, col2 = st.columns([3, 1])
    with col1:
        st.text_input("Folder to monitor", key="folder_input")

    def _browse_folder() -> None:
        if tk is None or filedialog is None:
            return
        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askdirectory()
        if selected:
            st.session_state.folder_input = selected
            st.session_state.folder = selected

            _rerun()


    with col2:
        st.button(
            "Browse",
            disabled=tk is None or filedialog is None,
            on_click=_browse_folder,
        )

    folder = st.session_state.folder_input
    st.session_state.folder = folder



    bin_upload = st.file_uploader(
        "Upload binary model",
        type=["pkl", "joblib"],
        help="Max file size: 2GB",
        key="binary_model_upload",
    )
    if bin_upload is not None:
        try:
            st.session_state.binary_model = joblib.load(bin_upload)
        except Exception:  # pragma: no cover - invalid model file
            st.session_state.log_lines.append("Failed to load binary model")

    mul_upload = st.file_uploader(
        "Upload multiclass model",
        type=["pkl", "joblib"],
        help="Max file size: 2GB",
        key="multi_model_upload",
    )
    if mul_upload is not None:
        try:
            st.session_state.multi_model = joblib.load(mul_upload)
        except Exception:  # pragma: no cover - invalid model file
            st.session_state.log_lines.append("Failed to load multiclass model")

    retention = st.number_input(
        "Auto clear files older than (hours, 0=off)",

        min_value=0,
        value=0,
        step=1,
        key="cleanup_hours",
    )
    if st.button("Clear data now"):

        _cleanup_generated(0, force=True)

    if Observer is None:
        st.error("watchdog is not installed")
        return

    start_disabled = st.session_state.observer is not None
    stop_disabled = st.session_state.observer is None

    if st.button("Start monitoring", disabled=start_disabled):
        handler = _FileMonitorHandler()
        observer = Observer()
        observer.schedule(handler, folder, recursive=False)
        observer.start()
        st.session_state.observer = observer
        st.session_state.handler = handler
        _log_toast(f"Monitoring started on {folder}")

    if st.button("Stop monitoring", disabled=stop_disabled):
        observer = st.session_state.observer
        if observer is not None:
            observer.stop()
            observer.join()
            st.session_state.observer = None
            st.session_state.handler = None
            _log_toast("Monitoring stopped")

    log_placeholder = st.empty()
    progress_bar = st.progress(0)

    if st.session_state.observer is not None:
        _process_events(st.session_state.handler, progress_bar)
        _cleanup_generated(retention)

    counts = st.session_state.get("last_counts")
    if counts:
        st.markdown(
            """
            <style>
            .enlarge-chart img {transition: transform 0.2s;}
            .enlarge-chart img:hover {transform: scale(1.5);}
            </style>
            """,
            unsafe_allow_html=True,
        )

        def _show(fig):
            st.markdown('<div class="enlarge-chart">', unsafe_allow_html=True)
            st.pyplot(fig, use_container_width=False)
            st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("is_attack distribution")
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(counts["is_attack"].index.astype(str), counts["is_attack"].values, color=["green", "red"])
        _show(fig)

        st.subheader("is_attack distribution (pie)")
        fig_pie, ax_pie = plt.subplots(figsize=(4, 3))
        ax_pie.pie(
            counts["is_attack"].values,
            labels=counts["is_attack"].index.astype(str),
            colors=["green", "red"],
            autopct="%1.1f%%",
        )
        ax_pie.axis("equal")
        _show(fig_pie)

        st.subheader("crlevel distribution")
        cr_colors = ["green", "yellowgreen", "gold", "orange", "red"]
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.bar(counts["crlevel"].index.astype(str), counts["crlevel"].values, color=cr_colors)
        _show(fig2)

        st.subheader("crlevel distribution (pie)")
        fig2_pie, ax2_pie = plt.subplots(figsize=(4, 3))
        ax2_pie.pie(
            counts["crlevel"].values,
            labels=counts["crlevel"].index.astype(str),
            colors=cr_colors,
            autopct="%1.1f%%",
        )
        ax2_pie.axis("equal")
        _show(fig2_pie)

        critical = st.session_state.get("last_critical")
        if critical is not None and not critical.empty:
            st.subheader("Critical traffic (crlevel ≥ 4)")
            st.dataframe(critical)
        else:
            st.info("No critical traffic predicted in the latest batch.")

    log_placeholder.text("\n".join(st.session_state.log_lines))
    if st.session_state.observer is not None:
        if st_autorefresh is not None:
            st_autorefresh(interval=1000, key="monitor_refresh")
        else:  # pragma: no cover - fallback when autorefresh missing
            time.sleep(1)
            _rerun()
