"""Pure helpers in personal_db.chat / personal_db.retrieve — no LLM, no I/O."""

from __future__ import annotations

from types import SimpleNamespace


def _node(text: str, score: float, **meta) -> SimpleNamespace:
    return SimpleNamespace(text=text, score=score, metadata=meta)


def _chunk(text, name="n.md", path="/x/n.md", doc_id="d", score=None, hits=0, origin="semantic"):
    from personal_db.retrieve import RetrievedChunk
    return RetrievedChunk(
        text=text, name=name, path=path, doc_id=doc_id,
        vector_score=score, graph_hits=hits, origin=origin,
    )


def test_filter_nodes_drops_below_floor(monkeypatch):
    from personal_db import config as cfg_mod
    from personal_db.retrieve import _filter_nodes
    monkeypatch.setattr(cfg_mod.config, "min_similarity_score", 0.5)
    nodes = [_node("a", 0.6), _node("b", 0.3), _node("c", 0.55)]
    kept = _filter_nodes(nodes)
    assert [n.text for n in kept] == ["a", "c"]


def test_filter_nodes_keeps_best_when_all_below_floor(monkeypatch):
    from personal_db import config as cfg_mod
    from personal_db.retrieve import _filter_nodes
    monkeypatch.setattr(cfg_mod.config, "min_similarity_score", 0.9)
    nodes = [_node("top", 0.3), _node("bot", 0.1)]
    kept = _filter_nodes(nodes)
    assert len(kept) == 1
    assert kept[0].text == "top"


def test_filter_nodes_empty():
    from personal_db.retrieve import _filter_nodes
    assert _filter_nodes([]) == []


def test_doc_name_uses_basename_when_missing():
    from personal_db.retrieve import _doc_name
    assert _doc_name({"name": "notes.md"}) == "notes.md"
    assert _doc_name({"name": "?", "path": "/x/y/journal.pdf"}) == "journal.pdf"
    assert _doc_name({"path": "/x/y/journal.pdf"}) == "journal.pdf"
    assert _doc_name({}) == "(unnamed)"


def test_format_context_and_citations():
    from personal_db.chat import _format_context
    chunks = [
        _chunk("hello", name="a.md", path="/x/a.md", doc_id="d1", score=0.7),
        _chunk("world", name="b.md", path="/x/b.md", doc_id="d2", score=0.6, hits=4, origin="both"),
    ]
    block, cites = _format_context(chunks)
    assert "[1] a.md\nhello" in block
    assert "[2] b.md\nworld" in block
    assert [c["n"] for c in cites] == [1, 2]
    assert cites[0]["score"] == 0.7
    assert cites[1]["doc_id"] == "d2"
    assert cites[1]["origin"] == "both"
    assert cites[1]["graph_hits"] == 4


def test_format_relations_renders_bullet_arrows():
    from personal_db.chat import _format_relations
    out = _format_relations([("Alex", "founded", "Helix"), ("Helix", "uses", "LiveKit")])
    assert "Alex ─founded→ Helix" in out
    assert "Helix ─uses→ LiveKit" in out


def test_build_messages_includes_summary_only_when_set():
    from personal_db.chat import _build_messages, SYSTEM_PROMPT
    from personal_db.sessions import Message
    from llama_index.core.llms import MessageRole

    history = [Message(role="user", content="prev-u"), Message(role="assistant", content="prev-a")]

    # Without summary: system + 2 hist + final user = 4
    msgs = _build_messages("", history, "what now?", "CTX")
    assert len(msgs) == 4
    assert msgs[0].role == MessageRole.SYSTEM
    assert msgs[0].content == SYSTEM_PROMPT
    assert "CTX" in msgs[-1].content
    assert "what now?" in msgs[-1].content

    # With summary: extra system block at position 1.
    msgs2 = _build_messages("RECAP", history, "what now?", "CTX")
    assert len(msgs2) == 5
    assert msgs2[1].role == MessageRole.SYSTEM
    assert "RECAP" in msgs2[1].content


def test_build_messages_appends_relations_block_when_present():
    from personal_db.chat import _build_messages
    from personal_db.sessions import Message
    msgs = _build_messages("", [], "q?", "CTX", "REL_TRIPLES")
    body = msgs[-1].content
    assert "Context from my personal database" in body
    assert "Known relations from my graph" in body
    assert "REL_TRIPLES" in body
