import streamlit as st
try:
    from streamlit_option_menu import option_menu
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    option_menu = None

from ui_pages import (
    training_ui,
    gpu_etl_ui,
    inference_ui,
    folder_monitor_ui,
    visualization_ui,
    notifier_app,
)

if "menu_collapse" not in st.session_state:
    st.session_state.menu_collapse = False

sidebar_width = "72px" if st.session_state.menu_collapse else "260px"

st.set_page_config(
    page_title="D-FLARE Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)

st.markdown(
    f"""
    <style>

    .stApp {{
        background-color: #f3f4f6;
    }}
    div[data-testid="stSidebar"] {{
        width: {sidebar_width};
        background-color: #1f2937;
        transition: width 0.3s ease;
    }}
    div[data-testid="stSidebar"] .nav-link {{
        color: #e5e7eb;
    }}
    div[data-testid="stSidebar"] .nav-link:hover {{
        background-color: #374151;
    }}
    div[data-testid="stSidebar"] .nav-link-selected {{
        background-color: #2563eb;
        color: #ffffff;
    }}
    .menu-expanded .nav-link span {{
        display: inline-block;
    }}
    .menu-collapsed .nav-link span {{
        display: none;
    }}
    .menu-collapsed .nav-link {{
        justify-content: center;
    }}
=======
    .stApp {
        background-color: #f5f7fa;
    }
    /* sidebar base style */
    .menu-expanded .nav-link span {
        display: inline-block;
    }
    .menu-collapsed .nav-link span {
        display: none;
    }
    .menu-collapsed .nav-link {
        justify-content: center;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = {
    "Training Pipeline": training_ui.app,
    "GPU ETL Pipeline": gpu_etl_ui.app,
    "Model Inference": inference_ui.app,
    "Folder Monitor": folder_monitor_ui.app,
    "Visualization": visualization_ui.app,
    "Notifications": notifier_app.app,
}

PAGE_ICONS = {
    "Training Pipeline": "cpu",
    "GPU ETL Pipeline": "gpu",
    "Model Inference": "search",
    "Folder Monitor": "folder",
    "Visualization": "bar-chart",
    "Notifications": "bell",
}

PAGE_DESCRIPTIONS = {
    "Training Pipeline": "Configure and run model training jobs.",
    "GPU ETL Pipeline": "Execute ETL processes accelerated by GPUs.",
    "Model Inference": "Perform inference using trained models.",
    "Folder Monitor": "Watch a directory for CSV/TXT/log files, including compressed variants.",
    "Visualization": "Explore dataset and model outputs through charts.",
    "Notifications": "Send Discord alerts with Gemini-generated advice.",
}

with st.sidebar:
    st.title("D-FLARE system")
    st.markdown("整合訓練、ETL、推論與通知的威脅分析平台。")


    if "menu_collapse" not in st.session_state:
        st.session_state.menu_collapse = False

    if option_menu:
        if st.button("☰", key="menu_toggle"):
            st.session_state.menu_collapse = not st.session_state.menu_collapse
        menu_class = "menu-collapsed" if st.session_state.menu_collapse else "menu-expanded"
        with st.container():
            st.markdown(f"<div class='{menu_class}'>", unsafe_allow_html=True)
            selection = option_menu(
                None,
                list(PAGES.keys()),
                icons=[PAGE_ICONS[k] for k in PAGES.keys()],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "0", "background-color": "#1F2937"},
                    "icon": {"color": "white", "font-size": "16px"},
                    "nav-link": {
                        "color": "#d1d5db",
                        "font-size": "16px",
                        "text-align": "left",
                        "margin": "0px",
                        "--hover-color": "#4b5563",
                    },
                    "nav-link-selected": {"background-color": "#111827"},
                },
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                """
                <script>
                const links = window.parent.document.querySelectorAll('.nav-link');
                links.forEach((el) => el.setAttribute('title', el.textContent));
                </script>
                """,
                unsafe_allow_html=True,
            )

    else:  # Fallback to simple radio when option_menu missing
        selection = st.radio("Go to", list(PAGES.keys()))

    st.markdown(PAGE_DESCRIPTIONS.get(selection, ""))

PAGES[selection]()
