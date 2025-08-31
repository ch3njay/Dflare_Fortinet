import os
import time
import pandas as pd
import joblib
import streamlit as st

from etl_pipeliner import run_pipeline
from notifier import notify_from_csv

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
        if path.endswith(self.SUPPORTED_EXTS):
            self.events.append((event_type, path))

    def on_created(self, event):  # pragma: no cover - requires filesystem events
        if not event.is_directory:
            self._track("created", event.src_path)

    def on_modified(self, event):  # pragma: no cover - requires filesystem events
        if not event.is_directory:
            self._track("modified", event.src_path)

def _run_etl_and_infer(path: str, progress_bar) -> None:
    """Run ETL pipeline and model inference on *path*."""
    bin_model = st.session_state.get("binary_model")
    mul_model = st.session_state.get("multi_model")
    if not (bin_model and mul_model):
        st.session_state.log_lines.append("Model paths not set; skipping")
        return

    base = os.path.splitext(path)[0]
    pre_csv = base + "_preprocessed.csv"
    fe_csv = base + "_engineered.csv"
    try:
        run_pipeline(
            do_clean=False,
            do_map=True,
            do_fe=True,
            clean_out=path,
            preproc_out=pre_csv,
            fe_out=fe_csv,
        )
        df = pd.read_csv(fe_csv)
        features = [c for c in df.columns if c not in {"is_attack", "crlevel"}]
        bin_clf = joblib.load(bin_model)
        bin_pred = bin_clf.predict(df[features])
        result = df.copy()
        result["is_attack"] = bin_pred
        mask = result["is_attack"] == 1
        if mask.any():
            mul_clf = joblib.load(mul_model)
            result.loc[mask, "crlevel"] = mul_clf.predict(df.loc[mask, features])
        report_path = base + "_report.csv"
        result.to_csv(report_path, index=False)
        st.session_state.generated_files.update({pre_csv, fe_csv, report_path})
        webhook = st.session_state.get("discord_webhook", "")
        gemini_key = st.session_state.get("gemini_key", "")
        if webhook:
            notify_from_csv(
                report_path,
                webhook,
                gemini_key,
                risk_levels={"3", "4"},
                ui_log=st.write,
            )
        st.session_state.log_lines.append(f"Processed {path} -> {report_path}")
        for pct in range(0, 101, 20):
            progress_bar.progress(pct)
            time.sleep(0.05)
    except Exception as exc:  # pragma: no cover - processing errors
        st.session_state.log_lines.append(f"Processing failed {path}: {exc}")


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
    with col2:
        if st.button("Browse", disabled=tk is None or filedialog is None):
            root = tk.Tk()
            root.withdraw()
            selected = filedialog.askdirectory()
            if selected:
                st.session_state.folder_input = selected
                st.session_state.folder = selected
                st.experimental_rerun()
    folder = st.session_state.folder_input
    st.session_state.folder = folder


    st.text_input("Binary model path", key="binary_model")
    st.text_input("Multiclass model path", key="multi_model")
    retention = st.number_input(
        "Auto clear generated files older than (hours, 0=off)",
        min_value=0,
        value=0,
        step=1,
        key="cleanup_hours",
    )
    if st.button("Clear generated files now"):
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
        st.session_state.log_lines.append(f"Monitoring started on {folder}")

    if st.button("Stop monitoring", disabled=stop_disabled):
        st.session_state.observer.stop()
        st.session_state.observer.join()
        st.session_state.observer = None
        st.session_state.handler = None
        st.session_state.log_lines.append("Monitoring stopped")

    log_placeholder = st.empty()
    progress_bar = st.progress(0)

    if st.session_state.observer is not None:
        _process_events(st.session_state.handler, progress_bar)
        _cleanup_generated(retention)
        log_placeholder.text("\n".join(st.session_state.log_lines))
        time.sleep(1)
        st.experimental_rerun()
    else:
        log_placeholder.text("\n".join(st.session_state.log_lines))
