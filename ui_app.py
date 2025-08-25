import streamlit as st
from ui_pages import (
    training_ui,
    gpu_etl_ui,
    inference_ui,
    folder_monitor_ui,
    visualization_ui,
)

st.set_page_config(
    page_title="D-FLARE Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)
# Upload size limit configured in .streamlit/config.toml

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = {
    "🧠 Training Pipeline": training_ui.app,
    "⚙️ GPU ETL Pipeline": gpu_etl_ui.app,
    "🔍 Model Inference": inference_ui.app,
    "📂 Folder Monitor": folder_monitor_ui.app,
    "📊 Visualization": visualization_ui.app,
}

PAGE_DESCRIPTIONS = {
    "🧠 Training Pipeline": "Configure and run model training jobs.",
    "⚙️ GPU ETL Pipeline": "Execute ETL processes accelerated by GPUs.",
    "🔍 Model Inference": "Perform inference using trained models.",
    "📂 Folder Monitor": "Watch a directory for CSV/TXT/log files, including compressed variants.",
    "📊 Visualization": "Explore dataset and model outputs through charts.",
}

st.sidebar.title("📚 Navigation")
st.sidebar.markdown(
    "Use **Folder Monitor** to watch CSV/TXT and compressed log files. "
    "Files are processed after 5 seconds of inactivity to avoid partial reads."
)
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
st.sidebar.markdown(PAGE_DESCRIPTIONS.get(selection, ""))
PAGES[selection]()
