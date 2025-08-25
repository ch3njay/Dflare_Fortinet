"""Streamlit interface for notification utilities."""

import tempfile
from pathlib import Path

import streamlit as st

from notifier import notify_from_csv, send_discord


def app() -> None:
    st.title("🔔 Notification System")

    st.sidebar.header("Settings")
    webhook = st.sidebar.text_input("Discord Webhook URL", key="discord_webhook")
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password", key="gemini_key")
    st.sidebar.text_input("LINE Notify Token", key="line_token")
    st.sidebar.info("LINE notifications currently disabled")

    risk_levels = st.sidebar.multiselect("High-risk levels", [1, 2, 3, 4], default=[3, 4])
    dedupe_strategy = st.sidebar.selectbox(
        "Deduplication strategy", ["Filename + mtime", "File hash"]
    )

    dedupe_cache = st.session_state.setdefault(
        "dedupe_cache", {"strategy": "mtime", "keys": set()}
    )
    dedupe_cache["strategy"] = "hash" if dedupe_strategy == "File hash" else "mtime"

    st.subheader("Actions")
    if st.button("Send Discord test notification"):
        if webhook:
            ok, info = send_discord(webhook, "This is a test notification from D-FLARE.")
            if ok:
                st.success("Test notification sent")
            else:
                st.error(f"Failed to send: {info}")
        else:
            st.warning("Please set the Discord Webhook URL first")

    uploaded = st.file_uploader("Select result CSV", type=["csv"])
    if uploaded is not None:
        temp_dir = tempfile.gettempdir()
        tmp_path = Path(temp_dir) / uploaded.name
        with open(tmp_path, "wb") as fh:
            fh.write(uploaded.getbuffer())

        if st.button("Parse and notify"):
            if not webhook:
                st.warning("Please set the Discord Webhook URL first")
            else:
                results = notify_from_csv(
                    str(tmp_path),
                    webhook,
                    gemini_key,
                    risk_levels={str(r) for r in risk_levels},
                    ui_log=st.write,
                    dedupe_cache=dedupe_cache,
                )
                success = sum(1 for _, ok, _ in results if ok)
                fail = sum(1 for _, ok, _ in results if not ok)
                st.info(f"Succeeded {success}, failed {fail}")

