import streamlit as st
from . import _ensure_module
_ensure_module("numpy", "numpy_stub")
_ensure_module("pandas", "pandas_stub")
import pandas as pd
from training_pipeline.pipeline_main import TrainingPipeline

def app() -> None:
    st.title("Training Pipeline")
    uploaded_file = st.file_uploader(
        "Upload training CSV",
        type=["csv"],
        help="Max file size: 2GB",
    )
    task_type = st.selectbox("Task type", ["binary", "multiclass"])
    optuna_enabled = st.checkbox("Enable Optuna", value=False)
    optimize_base = st.checkbox("Optimize base models", value=False)
    optimize_ensemble = st.checkbox("Optimize ensemble", value=False)
    use_tuned_for_training = st.checkbox("Use tuned params for training", value=False)
    if st.button("Run training"):
        if uploaded_file is None:
            st.error("Please upload a CSV file")
            return
        tmp_path = f"uploaded_{uploaded_file.name}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pipeline = TrainingPipeline(
            task_type=task_type,
            optuna_enabled=optuna_enabled,
            optimize_base=optimize_base,
            optimize_ensemble=optimize_ensemble,
            use_tuned_for_training=use_tuned_for_training,
        )
        try:
            pipeline.run(tmp_path)
            st.success("Training finished")
        except Exception as e:
            st.error(f"Training failed: {e}")
