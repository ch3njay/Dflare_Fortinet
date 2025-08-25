import streamlit as st
from gpu_etl_pipeliner import run_pipeline

def app() -> None:
    st.title("GPU ETL Pipeline")
    uploaded_files = st.file_uploader(
        "Upload log files",
        type=["csv", "txt", "gz"],
        accept_multiple_files=True,
        help="Max file size: 2GB per file",
    )
    do_clean = st.checkbox("Run cleaning", value=True)
    do_map = st.checkbox("Run mapping", value=True)
    do_fe = st.checkbox("Run feature engineering", value=True)
    out_path = st.text_input("Output CSV path", "engineered_data.csv")
    if st.button("Run ETL"):
        if not uploaded_files:
            st.error("Please upload at least one file")
            return
        paths = []
        for f in uploaded_files:
            tmp = f"uploaded_{f.name}"
            with open(tmp, "wb") as g:
                g.write(f.getbuffer())
            paths.append(tmp)
        try:
            result = run_pipeline(paths, out_path=out_path, do_clean=do_clean, do_map=do_map, do_fe=do_fe, quiet=True)
            st.success(f"ETL completed: {result}")
        except Exception as e:
            st.error(f"ETL failed: {e}")
