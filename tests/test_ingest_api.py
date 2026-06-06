"""API tests for the streaming ingestion flow.

`ingest_one` is stubbed so we don't read real files through pymupdf or embed
through Ollama — we only exercise the job lifecycle, SSE progress events, and
the start endpoint's validation."""

from __future__ import annotations

import json


def _drain_ingest(client, timeout: float = 5.0) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    with client.stream("GET", "/ingest/stream", timeout=timeout) as r:
        assert r.status_code == 200
        ev = None
        for line in r.iter_lines():
            if not line:
                ev = None
                continue
            if line.startswith("event:"):
                ev = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and ev is not None:
                events.append((ev, line.split(":", 1)[1].strip()))
                if ev in ("done", "error"):
                    return events
    return events


def _stub_ok(monkeypatch):
    from personal_db.web import app as web_app

    def fake_one(path, stores, **kw):
        return {"name": path.name, "status": "ingested", "chunks": 2, "doc_id": f"x-{path.name}"}

    monkeypatch.setattr(web_app, "ingest_one", fake_one)


def test_ingest_start_single_file_streams_progress(client, monkeypatch):
    _stub_ok(monkeypatch)
    r = client.post(
        "/ingest/start",
        files=[("files", ("a.md", b"# A\n\nbody", "text/markdown"))],
    )
    assert r.status_code == 200
    assert r.json()["total"] == 1

    events = _drain_ingest(client)
    types = [e[0] for e in events]
    assert "start" in types
    # one "processing" + one result = at least two progress events
    assert types.count("progress") >= 2
    assert types[-1] == "done"


def test_ingest_multiple_files_done_summary(client, monkeypatch):
    _stub_ok(monkeypatch)
    files = [("files", (f"d{i}.md", b"# x\n\nbody", "text/markdown")) for i in range(3)]
    r = client.post("/ingest/start", files=files)
    assert r.status_code == 200
    assert r.json()["total"] == 3

    events = _drain_ingest(client)
    done = next(json.loads(d) for t, d in events if t == "done")
    assert done["ingested"] == 3
    assert done["total"] == 3
    assert done.get("failed", 0) == 0


def test_ingest_progress_counts_climb(client, monkeypatch):
    _stub_ok(monkeypatch)
    files = [("files", (f"d{i}.md", b"# x\n\nbody", "text/markdown")) for i in range(2)]
    client.post("/ingest/start", files=files)
    events = _drain_ingest(client)
    dones = [json.loads(d).get("done") for t, d in events if t == "progress"]
    # Should reach the total at the end.
    assert max(dones) == 2


def test_ingest_start_rejects_nothing(client):
    r = client.post("/ingest/start", data={"path": ""})
    assert r.status_code == 400


def test_ingest_start_rejects_missing_path(client):
    r = client.post("/ingest/start", data={"path": "/no/such/dir/zzz-xyz"})
    assert r.status_code == 400


def test_ingest_stream_404_when_idle(client):
    r = client.get("/ingest/stream")
    assert r.status_code == 404


def test_ingest_form_renders(client):
    r = client.get("/ingest")
    assert r.status_code == 200
    assert "ingestForm" in r.text
    assert "/ingest/start" in r.text
