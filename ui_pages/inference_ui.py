import io
import threading
import time
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

        progress = st.progress(0)
        status = st.empty()
        result_holder = {"df": None, "error": None}

        def _run():
            try:
                df = pd.read_csv(data_file)

                df = df.select_dtypes(include=["number", "bool"]).copy()

                binary_model.seek(0)
                bin_clf = joblib.load(binary_model)
                features = getattr(bin_clf, "feature_names_in_", None)
                if features is None and hasattr(bin_clf, "get_booster"):
                    features = bin_clf.get_booster().feature_names
                df = df.reindex(columns=features)
                for col in df.columns:
                    if pd.api.types.is_bool_dtype(df[col].dtype):
                        df[col] = df[col].fillna(False).astype("int8", copy=False)
                    else:
                        df[col] = (
                            pd.to_numeric(df[col], errors="coerce")
                            .fillna(0)
                            .astype("float32", copy=False)
                        )
                bin_pred = bin_clf.predict(df)
                result = pd.DataFrame({"is_attack": bin_pred})
                mask = result["is_attack"] == 1
                if mask.any():
                    multi_model.seek(0)
                    mul_clf = joblib.load(multi_model)
                    cr_pred = mul_clf.predict(df.loc[mask])
                    result.loc[mask, "crlevel"] = cr_pred
                result_holder["df"] = result
            except Exception as exc:  # pragma: no cover - runtime failure
                result_holder["error"] = exc

        thread = threading.Thread(target=_run)
        thread.start()
        pct = 0
        while thread.is_alive():
            if pct < 95:
                pct += 5
            progress.progress(pct)
            status.text(f"Inference in progress... {pct}%")
            time.sleep(0.1)
        thread.join()
        if result_holder["error"] is None:
            progress.progress(100)
            status.text("Inference completed")
            st.session_state["prediction_results"] = result_holder["df"]
            st.dataframe(result_holder["df"])
        else:
            status.text("Inference failed")
            st.error(f"Inference failed: {result_holder['error']}")
