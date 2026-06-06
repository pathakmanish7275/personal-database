"""Planner — clarify-or-search decision before retrieval."""

from __future__ import annotations

from unittest.mock import MagicMock


def _reply(content: str) -> MagicMock:
    return MagicMock(message=MagicMock(content=content))


def test_parse_json_handles_raw_object():
    from personal_db.planner import _parse_json
    assert _parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_strips_code_fences():
    from personal_db.planner import _parse_json
    assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_extracts_first_block_from_prose():
    from personal_db.planner import _parse_json
    out = _parse_json("Sure, here you go:\n{\"action\": \"search\"}\nthanks!")
    assert out == {"action": "search"}


def test_parse_json_returns_none_for_garbage():
    from personal_db.planner import _parse_json
    assert _parse_json("not json at all") is None


def test_plan_returns_search_with_refined_query(monkeypatch):
    from personal_db import planner as p

    monkeypatch.setattr(p, "safe_chat", lambda msgs: _reply(
        '{"action":"search","query":"helix founder","clarify":null,"reason":"specific"}'
    ))
    out = p.plan("who started Helix?", [])
    assert out.action == "search"
    assert out.query == "helix founder"
    assert "specific" in out.reason


def test_plan_returns_clarify_when_model_asks(monkeypatch):
    from personal_db import planner as p
    monkeypatch.setattr(p, "safe_chat", lambda msgs: _reply(
        '{"action":"clarify","query":null,"clarify":"What topic do you mean?","reason":"vague"}'
    ))
    out = p.plan("what do I do?", [])
    assert out.action == "clarify"
    assert out.clarify == "What topic do you mean?"


def test_plan_falls_through_to_search_on_non_json(monkeypatch):
    from personal_db import planner as p
    monkeypatch.setattr(p, "safe_chat", lambda msgs: _reply("I'm not sure what you mean."))
    out = p.plan("anything", [])
    assert out.action == "search"
    assert out.query == "anything"


def test_plan_falls_through_to_search_on_llm_error(monkeypatch):
    from personal_db import planner as p

    def boom(_msgs):
        raise RuntimeError("network down")

    monkeypatch.setattr(p, "safe_chat", boom)
    out = p.plan("something", [])
    assert out.action == "search"
    assert out.query == "something"


def test_plan_clarify_with_empty_string_falls_to_search(monkeypatch):
    from personal_db import planner as p
    monkeypatch.setattr(p, "safe_chat", lambda msgs: _reply(
        '{"action":"clarify","query":null,"clarify":"","reason":"x"}'
    ))
    out = p.plan("hello", [])
    assert out.action == "search"
    assert out.query == "hello"


def test_plan_anti_loop_overrides_second_clarify(monkeypatch):
    """If the planner already clarified once, the hard guard forces search."""
    from personal_db import planner as p
    monkeypatch.setattr(p, "safe_chat", lambda msgs: _reply(
        '{"action":"clarify","query":null,"clarify":"which one?","reason":"still vague"}'
    ))
    out = p.plan("entire conversation", [], previously_clarified="What topic do you mean?")
    assert out.action == "search"
    assert out.query == "entire conversation"
    assert "anti-loop" in out.reason.lower()


def test_previous_clarification_helper_detects_null_citations():
    from personal_db.chat import _previous_clarification
    from personal_db.sessions import Message
    recent = [
        Message(id=1, role="user", content="hi"),
        Message(id=2, role="assistant", content="What did you mean?", citations=None),
        Message(id=3, role="user", content="follow-up"),
    ]
    assert _previous_clarification(recent) == "What did you mean?"


def test_previous_clarification_helper_skips_real_answers():
    from personal_db.chat import _previous_clarification
    from personal_db.sessions import Message
    recent = [
        Message(id=1, role="user", content="hi"),
        Message(id=2, role="assistant", content="Here is the answer.", citations="[]"),
        Message(id=3, role="user", content="more"),
    ]
    assert _previous_clarification(recent) is None
