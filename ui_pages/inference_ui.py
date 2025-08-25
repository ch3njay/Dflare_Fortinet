import io
import streamlit as st
from . import _ensure_module
_ensure_module("numpy", "numpy_stub")
_ensure_module("pandas", "pandas_stub")
import pandas as pd
import joblib

def app() -> None:
    st.title("Model Inference")
    data_file = st.file_uploader(
        "Upload data CSV",
        type=["csv"],
        help="Max file size: 2GB",
    )
    binary_model = st.file_uploader(
        "Upload binary model",
        type=["pkl", "joblib"],
        help="Max file size: 2GB",
    )
    multi_model = st.file_uploader(
        "Upload multiclass model",
        type=["pkl", "joblib"],
        help="Max file size: 2GB",
    )
    if st.button("Run inference"):
        if data_file is None or binary_model is None or multi_model is None:
            st.error("Please upload data and model files")
            return
        df = pd.read_csv(data_file)
        binary_model.seek(0)
        bin_clf = joblib.load(binary_model)
        bin_pred = bin_clf.predict(df)
        result = pd.DataFrame({"is_attack": bin_pred})
        mask = result["is_attack"] == 1
        if mask.any():
            multi_model.seek(0)
            mul_clf = joblib.load(multi_model)
            cr_pred = mul_clf.predict(df[mask])
            result.loc[mask, "crlevel"] = cr_pred
        st.session_state["prediction_results"] = result
        st.dataframe(result)
