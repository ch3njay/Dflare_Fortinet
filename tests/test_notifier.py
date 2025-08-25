import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import notifier


def test_normalize_crlevel_mapping():
    assert notifier.normalize_crlevel("low") == 1
    assert notifier.normalize_crlevel(4) == 4
    assert notifier.normalize_crlevel("critical") == 4
    assert notifier.normalize_crlevel("unknown") is None


def test_notify_from_csv(tmp_path, monkeypatch):
    csv = tmp_path / "events.csv"
    csv.write_text("crlevel,srcip,description\n3,1.1.1.1,test desc\n2,2.2.2.2,low\n")

    sent = []

    def fake_send(url, content):
        sent.append(json.dumps({"url": url, "content": content}))
        return True, "OK"

    def fake_ask(desc, key):
        return "建議一\n建議二"

    monkeypatch.setattr(notifier, "send_discord", fake_send)
    monkeypatch.setattr(notifier, "ask_gemini", fake_ask)

    res = notifier.notify_from_csv(
        str(csv),
        "http://hook",
        "key",
        risk_levels={"3"},
        dedupe_cache={"strategy": "mtime", "keys": set()},
    )

    assert len(sent) == 1
    assert "高風險事件" in json.loads(sent[0])["content"]
    assert len(res) == 1
