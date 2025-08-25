import os
import time
import gzip
import zipfile
import streamlit as st

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

def _process_events(handler: _FileMonitorHandler, progress_bar) -> None:
    """Process new events, reading only new data and updating UI."""
    new_events = handler.events[len(st.session_state.get("processed_events", [])) :]
    for ev, path in new_events:
        # ensure file has settled for at least 5 seconds
        try:
            if time.time() - os.path.getmtime(path) < 5:
                continue
        except OSError:
            continue
        offset = st.session_state.file_offsets.get(path, 0)
        data = ""
        try:
            if path.endswith(".gz"):
                with gzip.open(path, "rt") as f:
                    text = f.read()
                data = text[offset:]
                st.session_state.file_offsets[path] = len(text)
            elif path.endswith(".zip"):
                with zipfile.ZipFile(path) as z:
                    text = ""
                    for name in z.namelist():
                        with z.open(name) as f:
                            text += f.read().decode("utf-8")
                data = text[offset:]
                st.session_state.file_offsets[path] = len(text)
            else:
                with open(path, "r") as f:
                    f.seek(offset)
                    data = f.read()
                    st.session_state.file_offsets[path] = f.tell()
        except Exception as exc:  # pragma: no cover - file reading errors
            st.session_state.log_lines.append(f"Error reading {path}: {exc}")
            continue
        if data:
            st.session_state.log_lines.append(
                f"{ev}: {path} ({len(data.splitlines())} new lines)"
            )
            for pct in range(0, 101, 20):
                progress_bar.progress(pct)
                time.sleep(0.05)
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
    st.session_state.setdefault("file_offsets", {})
    st.session_state.setdefault("processed_events", [])

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
        log_placeholder.text("\n".join(st.session_state.log_lines))
        time.sleep(1)
        st.experimental_rerun()
    else:
        log_placeholder.text("\n".join(st.session_state.log_lines))
