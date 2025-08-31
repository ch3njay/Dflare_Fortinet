import json
import sys
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
    csv.write_text(
        "is_attack,crlevel,srcip,description\n"
        "1,3,1.1.1.1,test desc\n"
        "0,3,2.2.2.2,low\n"
    )

    sent = []

    def fake_send(url, content):
        sent.append(json.dumps({"url": url, "content": content}))
        return True, "OK"

    def fake_ask(desc, key):
        return "Recommendation 1\nRecommendation 2"

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
    assert "High-risk event detected" in json.loads(sent[0])["content"]
    assert len(res) == 1


def test_notify_from_csv_progress(tmp_path, monkeypatch):
    csv = tmp_path / "events.csv"
    csv.write_text(
        "is_attack,crlevel,srcip,description\n1,3,1.1.1.1,test desc\n"
    )

    monkeypatch.setattr(notifier, "send_discord", lambda url, content: (True, "OK"))
    monkeypatch.setattr(notifier, "ask_gemini", lambda desc, key: "R1\nR2")

    progress_vals = []
    notifier.notify_from_csv(
        str(csv),
        "http://hook",
        "key",
        risk_levels={"3"},
        progress_cb=lambda frac: progress_vals.append(frac),
    )
    assert progress_vals and progress_vals[-1] == 1.0


def test_notify_from_csv_line(tmp_path, monkeypatch):
    csv = tmp_path / "events.csv"
    csv.write_text(
        "is_attack,crlevel,srcip,description\n1,3,1.1.1.1,test desc\n"
    )


    monkeypatch.setattr(notifier, "send_discord", lambda url, content: (True, "OK"))
    monkeypatch.setattr(notifier, "ask_gemini", lambda desc, key: "R1\nR2")

    sent = []

    def fake_line(token, msg, callback=None):
        sent.append((token, msg))
        return True

    monkeypatch.setattr(notifier, "send_line_to_all", fake_line)

    notifier.notify_from_csv(
        str(csv),
        "http://hook",
        "key",
        risk_levels={"3"},
        line_token="TOKEN",
    )
    assert sent and sent[0][0] == "TOKEN"



def test_notify_from_csv_no_gemini(tmp_path, monkeypatch):
    csv = tmp_path / "events.csv"
    csv.write_text(
        "is_attack,crlevel,srcip,description\n1,3,1.1.1.1,test desc\n"
    )

    called = {"ask": False}

    def fake_ask(desc, key):
        called["ask"] = True
        return "should not be called"

    monkeypatch.setattr(notifier, "ask_gemini", fake_ask)
    monkeypatch.setattr(notifier, "send_discord", lambda url, content: (True, "OK"))

    res = notifier.notify_from_csv(
        str(csv),
        "http://hook",
        "",
        risk_levels={"3"},
    )
    assert not called["ask"]
    assert "AI Recommendation" not in res[0][0]


def test_notify_from_csv_filters_is_attack(tmp_path, monkeypatch):
    csv = tmp_path / "events.csv"
    csv.write_text(
        "is_attack,crlevel,srcip,description\n0,3,1.1.1.1,test desc\n"
    )

    monkeypatch.setattr(notifier, "send_discord", lambda url, content: (True, "OK"))
    monkeypatch.setattr(notifier, "ask_gemini", lambda desc, key: "R1\nR2")

    res = notifier.notify_from_csv(
        str(csv),
        "http://hook",
        "key",
        risk_levels={"3"},
    )
    assert res == []

