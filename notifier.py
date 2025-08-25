"""Reusable notification utilities for D-FLARE."""

from __future__ import annotations

import hashlib
import os
from typing import Dict, Iterable, Optional, Set, Tuple

try:  # pragma: no cover - best effort import
    import pandas as pd
except Exception:  # pragma: no cover - fallback
    pd = None
try:  # pragma: no cover - best effort import
    import requests
except Exception:  # pragma: no cover - network disabled
    requests = None  # type: ignore

# Mapping from various severity representations to numeric levels
_CRLEVEL_MAP = {
    "1": 1,
    "low": 1,
    "2": 2,
    "medium": 2,
    "3": 3,
    "high": 3,
    "4": 4,
    "critical": 4,
}

_COLUMN_ALIASES: Dict[str, Iterable[str]] = {
    "crlevel": ["crlevel", "cr_level", "level", "severity"],
    "srcip": ["srcip", "sourceip", "src_ip", "source_ip"],
    "description": ["description", "msg", "event_message", "Description"],
}


def normalize_crlevel(value) -> Optional[int]:
    """Normalize *crlevel* to an integer 1-4.

    Returns ``None`` if *value* cannot be interpreted.
    """

    if isinstance(value, (int, float)):
        val = int(value)
        return val if val in {1, 2, 3, 4} else None
    key = str(value).strip().lower()
    return _CRLEVEL_MAP.get(key)


def send_discord(webhook_url: str, content: str) -> Tuple[bool, str]:
    """Send *content* to a Discord *webhook_url*.

    Returns ``(True, "OK")`` on success or ``(False, error)`` on failure.
    """

    if requests is None:  # pragma: no cover - fallback
        return False, "requests library unavailable"
    try:
        resp = requests.post(webhook_url, json={"content": content}, timeout=10)
        if 200 <= resp.status_code < 300:
            return True, "OK"
        return False, f"{resp.status_code}: {resp.text}"
    except Exception as exc:  # pragma: no cover - network errors
        return False, str(exc)


def ask_gemini(desc: str, api_key: str) -> str:
    """Query Gemini for a two-line Traditional Chinese recommendation.

    Falls back to a fixed message if the API call fails.
    """

    try:  # pragma: no cover - external service
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-pro")
        prompt = (
            "以下是Fortinet事件描述:\n"
            f"{desc}\n"
            "請以繁體中文產生兩段回應：第一段為威脅說明，第二段為立即建議。"
        )
        response = model.generate_content(prompt)
        text = response.text.strip()
        if "\n" not in text:
            text += "\n尚無進一步建議。"
        return text
    except Exception:
        return "威脅說明：無法取得 AI 建議。\n立即建議：請參考內部流程處理。"


def _find_column(columns: Iterable[str], aliases: Iterable[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def notify_from_csv(
    csv_path: str,
    discord_webhook: str,
    gemini_key: str,
    *,
    risk_levels: Iterable = ("3", "4"),
    ui_log=None,
    dedupe_cache: Optional[Dict] = None,
):
    """Read a Fortinet event CSV and push high-risk rows to Discord."""

    if dedupe_cache is not None:
        strategy = dedupe_cache.get("strategy", "mtime")
        cache: Set[str] = dedupe_cache.setdefault("keys", set())
        if strategy == "hash":
            with open(csv_path, "rb") as fh:
                file_hash = hashlib.sha1(fh.read()).hexdigest()
            dedupe_key = f"{csv_path}:{file_hash}"
        else:
            mtime = os.path.getmtime(csv_path)
            dedupe_key = f"{csv_path}:{mtime}"
        if dedupe_key in cache:
            if ui_log:
                ui_log("檔案已處理過，跳過通知。")
            return []
        cache.add(dedupe_key)

    import csv

    try:
        if pd is not None:
            df = pd.read_csv(csv_path)
            rows = df.to_dict("records")
            columns = df.columns
        else:  # fallback parser
            with open(csv_path, newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
                columns = reader.fieldnames or []
    except Exception as exc:
        if ui_log:
            ui_log(f"讀取 CSV 失敗: {exc}")
        return []

    cr_col = _find_column(columns, _COLUMN_ALIASES["crlevel"])
    src_col = _find_column(columns, _COLUMN_ALIASES["srcip"])
    desc_col = _find_column(columns, _COLUMN_ALIASES["description"])
    if not (cr_col and src_col and desc_col):
        if ui_log:
            ui_log("CSV 缺少必要欄位。")
        return []

    risk_ints = {normalize_crlevel(x) for x in risk_levels}
    results = []
    for row in rows:
        cr_int = normalize_crlevel(row.get(cr_col))
        if cr_int is None or cr_int not in risk_ints:
            continue
        cr_text = {1: "low", 2: "medium", 3: "high", 4: "critical"}[cr_int]
        srcip = row.get(src_col)
        desc = row.get(desc_col)
        ai_text = ask_gemini(str(desc), gemini_key)
        lines = ai_text.splitlines()
        reco1 = lines[0] if lines else ""
        reco2 = lines[1] if len(lines) > 1 else ""
        message = (
            "🚨 偵測到高風險事件（Fortinet）\n"
            f"等級：{cr_text}（{cr_int}）\n"
            f"來源 IP：{srcip}\n"
            f"描述：{desc}\n"
            "———— AI 建議 ————\n"
            f"{reco1}\n{reco2}"
        )
        ok, info = send_discord(discord_webhook, message)
        results.append((message, ok, info))
        if ui_log:
            ui_log(f"已送出事件: {srcip} - {'成功' if ok else '失敗'}")

    if not results and ui_log:
        ui_log("沒有符合條件的事件。")
    return results


__all__ = [
    "normalize_crlevel",
    "send_discord",
    "ask_gemini",
    "notify_from_csv",
]

