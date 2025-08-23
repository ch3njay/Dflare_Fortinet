import os
import streamlit as st

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except Exception:  # pragma: no cover - watchdog may not be installed
    Observer = None
    FileSystemEventHandler = object

class _CSVHandler(FileSystemEventHandler):
    def __init__(self):
        self.events = []
    def on_created(self, event):
        if event.src_path.endswith('.csv'):
            self.events.append(('created', event.src_path))
            st.session_state.setdefault('monitored_files', []).append(event.src_path)
    def on_modified(self, event):
        if event.src_path.endswith('.csv'):
            self.events.append(('modified', event.src_path))
            st.session_state.setdefault('monitored_files', []).append(event.src_path)

def app() -> None:
    st.title("Folder Monitor")
    folder = st.text_input("Folder to monitor", ".")
    if 'observer' not in st.session_state:
        st.session_state.observer = None
        st.session_state.handler = None
    if Observer is None:
        st.error("watchdog is not installed")
        return
    if st.button("Start monitoring") and st.session_state.observer is None:
        handler = _CSVHandler()
        observer = Observer()
        observer.schedule(handler, folder, recursive=False)
        observer.start()
        st.session_state.observer = observer
        st.session_state.handler = handler
        st.success("Monitoring started")
    if st.button("Stop monitoring") and st.session_state.observer is not None:
        st.session_state.observer.stop()
        st.session_state.observer = None
        st.session_state.handler = None
        st.success("Monitoring stopped")
    if st.session_state.observer is not None:
        st.write("Recent events:")
        for ev, path in st.session_state.handler.events[-10:]:
            st.write(f"{ev}: {path}")
