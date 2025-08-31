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
    is_counts = df["is_attack"].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots()
    ax.bar(is_counts.index.astype(str), is_counts.values, color=["green", "red"])
    st.pyplot(fig)

    if "crlevel" in df.columns:
        counts = df["crlevel"].value_counts().reindex([0, 1, 2, 3, 4], fill_value=0)
        st.subheader("crlevel distribution (vertical)")
        fig_v, ax_v = plt.subplots()
        colors = ["green", "yellowgreen", "gold", "orange", "red"]
        ax_v.bar(counts.index.astype(str), counts.values, color=colors)
        st.pyplot(fig_v)

        st.subheader("crlevel distribution (horizontal)")
        fig_h, ax_h = plt.subplots()
        ax_h.barh(counts.index.astype(str), counts.values, color=colors)
        st.pyplot(fig_h)
