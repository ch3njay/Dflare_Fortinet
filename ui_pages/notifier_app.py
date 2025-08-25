"""Streamlit interface for notification utilities."""

import tempfile
from pathlib import Path

import streamlit as st

from notifier import notify_from_csv, send_discord


def app() -> None:
    st.title("🔔 通知系統")

    st.sidebar.header("設定")
    webhook = st.sidebar.text_input("Discord Webhook URL", key="discord_webhook")
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password", key="gemini_key")
    st.sidebar.text_input("LINE Notify Token", key="line_token")
    st.sidebar.info("LINE 推播目前停用")

    risk_levels = st.sidebar.multiselect("高風險等級", [1, 2, 3, 4], default=[3, 4])
    dedupe_strategy = st.sidebar.selectbox("去重策略", ["檔名+mtime", "檔案hash"])

    dedupe_cache = st.session_state.setdefault(
        "dedupe_cache", {"strategy": "mtime", "keys": set()}
    )
    dedupe_cache["strategy"] = "hash" if dedupe_strategy == "檔案hash" else "mtime"

    st.header("動作")
    if st.button("發送 Discord 測試通知"):
        if webhook:
            ok, info = send_discord(webhook, "這是來自 D-FLARE 的測試通知。")
            if ok:
                st.success("測試通知已送出")
            else:
                st.error(f"發送失敗: {info}")
        else:
            st.warning("請先設定 Discord Webhook URL")

    uploaded = st.file_uploader("選擇結果 CSV", type=["csv"])
    if uploaded is not None:
        temp_dir = tempfile.gettempdir()
        tmp_path = Path(temp_dir) / uploaded.name
        with open(tmp_path, "wb") as fh:
            fh.write(uploaded.getbuffer())

        if st.button("解析並推播"):
            if not webhook:
                st.warning("請先設定 Discord Webhook URL")
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
                st.info(f"成功 {success} 筆, 失敗 {fail} 筆")

