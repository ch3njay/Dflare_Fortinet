import streamlit as st
from . import _ensure_module
_ensure_module("numpy", "numpy_stub")
_ensure_module("pandas", "pandas_stub")
import pandas as pd
import matplotlib.pyplot as plt

def app() -> None:
    st.title("Prediction Visualization")
    df = st.session_state.get("prediction_results")
    if df is None:
        uploaded = st.file_uploader(
            "Upload prediction CSV",
            type=["csv"],
            help="Max file size: 2GB",
        )
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.session_state["prediction_results"] = df
    if df is None:
        st.info("No prediction results available")
        return
    st.subheader("is_attack distribution")
    st.bar_chart(df["is_attack"].value_counts())
    if "crlevel" in df.columns:
        counts = df["crlevel"].value_counts()
        st.subheader("crlevel distribution (vertical)")
        st.bar_chart(counts)
        st.subheader("crlevel distribution (horizontal)")
        fig, ax = plt.subplots()
        ax.barh(counts.index.astype(str), counts.values)
        st.pyplot(fig)
