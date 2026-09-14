"""Agent loop, tool plumbing, and the streaming wire client.

The wire-format expectations encoded here were measured against FLM v1.0.5 and
Ollama; see llm_stream's docstring. The loop guards each correspond to a
failure actually observed while testing against the live model.
"""

from __future__ import annotations

import pytest

from personal_db.llm_stream import StreamEvent, ToolCall, _reasoning_of, _ToolCallAccumulator
from personal_db.retrieve import RetrievalContext, RetrievedChunk
from personal_db.tools import CitationLedger, format_hits


def _chunk(text="passage", name="notes.md", path="/n/notes.md"):
    return RetrievedChunk(text=text, name=name, path=path, doc_id="d1",
                          vector_score=0.7, origin="semantic")


# ── wire client ────────────────────────────────────────────────────────────

def test_reasoning_field_accepted_under_both_names():
    """FLM calls it reasoning_content; Ollama calls it reasoning."""
    assert _reasoning_of({"reasoning_content": "a"}) == "a"
    assert _reasoning_of({"reasoning": "b"}) == "b"
    assert _reasoning_of({"content": "c"}) is None


def test_tool_call_accumulator_merges_fragmented_arguments():
    """FLM sends a tool call complete in one delta, but Ollama may split the
    arguments string — so we must never depend on getting it whole."""
    acc = _ToolCallAccumulator()
    acc.feed([{"index": 0, "id": "call_1", "function": {"name": "search_kb", "arguments": '{"que'}}])
    acc.feed([{"index": 0, "function": {"arguments": 'ry":"budget"}'}}])
    calls = acc.result()
    assert len(calls) == 1
    assert calls[0].name == "search_kb"
    assert calls[0].parsed_arguments() == {"query": "budget"}


def test_tool_call_accumulator_keeps_parallel_calls_separate():
    acc = _ToolCallAccumulator()
    acc.feed([{"index": 0, "id": "a", "function": {"name": "search_kb", "arguments": '{"query":"x"}'}},
              {"index": 1, "id": "b", "function": {"name": "search_kb", "arguments": '{"query":"y"}'}}])
    calls = acc.result()
    assert [c.parsed_arguments()["query"] for c in calls] == ["x", "y"]


def test_malformed_tool_arguments_degrade_to_empty_dict():
    """A malformed call must become "no usable arguments", not an exception."""
    assert ToolCall(id="1", name="search_kb", arguments="{not json").parsed_arguments() == {}


# ── citations ──────────────────────────────────────────────────────────────

def test_citation_numbers_are_stable_across_repeated_searches():
    """The model may search twice and get overlapping chunks back. Each chunk
    must keep exactly one number, because the UI's [N] chips refer to them."""
    ledger = CitationLedger()
    a, b = _chunk("one", path="/a"), _chunk("two", path="/b")
    assert ledger.number_for(a) == 1
    assert ledger.number_for(b) == 2
    assert ledger.number_for(a) == 1  # same chunk, same number
    assert len(ledger.cites) == 2


def test_format_hits_states_empty_result_explicitly():
    """A blank tool result reads as a malfunction and invites invention."""
    out = format_hits(RetrievalContext(chunks=[]), CitationLedger())
    assert "No matching passages" in out


def test_format_hits_numbers_passages_and_appends_relations():
    ctx = RetrievalContext(chunks=[_chunk("the text")], relations=[("Priya", "owns", "deploys")])
    out = format_hits(ctx, CitationLedger())
    assert "[1] notes.md" in out
    assert "the text" in out
    assert "Priya ─owns→ deploys" in out


# ── agent loop ─────────────────────────────────────────────────────────────

def _fake_stream(script):
    """Build a stream_chat replacement that replays scripted turns in order."""
    turns = list(script)

    def _stream(messages, **kwargs):
        events = turns.pop(0)
        for ev in events:
            yield ev

    return _stream


def _tool_turn(query, call_id="call_1"):
    return [
        StreamEvent(kind="reasoning", text="thinking..."),
        StreamEvent(kind="content", text="\n\n"),
        StreamEvent(kind="tool_calls",
                    tool_calls=[ToolCall(id=call_id, name="search_kb",
                                         arguments='{"query":"%s"}' % query)]),
        StreamEvent(kind="done", finish_reason="tool_calls"),
    ]


def _answer_turn(text):
    return [
        StreamEvent(kind="reasoning", text="thinking..."),
        StreamEvent(kind="content", text="\n\n"),
        StreamEvent(kind="content", text=text),
        StreamEvent(kind="done", finish_reason="stop"),
    ]


def _runner(monkeypatch, script, retrieve=None, **kw):
    from personal_db import agent as agent_mod
    monkeypatch.setattr(agent_mod, "stream_chat", _fake_stream(script))
    return agent_mod.AgentRunner(
        base_url="http://x", model="m",
        retrieve=retrieve or (lambda q: RetrievalContext(chunks=[_chunk()])),
        **kw,
    )


def _collect(runner):
    out = {"tokens": [], "searches": [], "cites": [], "finish": None, "reasoning": 0}
    for ev in runner.run([{"role": "user", "content": "q"}]):
        if ev.kind == "token":
            out["tokens"].append(ev.text)
        elif ev.kind == "searching":
            out["searches"].append(ev.query)
        elif ev.kind == "citations":
            out["cites"] = ev.cites
        elif ev.kind == "reasoning":
            out["reasoning"] += 1
        elif ev.kind == "done":
            out["finish"] = ev.finish_reason
    out["answer"] = "".join(out["tokens"])
    return out


def test_direct_answer_skips_retrieval(monkeypatch):
    """Chit-chat must not trigger a search."""
    calls = []
    r = _runner(monkeypatch, [_answer_turn("Hello!")],
                retrieve=lambda q: calls.append(q) or RetrievalContext())
    out = _collect(r)
    assert calls == []
    assert out["answer"] == "Hello!"
    assert out["finish"] == "stop"


def test_tool_call_then_answer(monkeypatch):
    calls = []
    r = _runner(monkeypatch, [_tool_turn("budget"), _answer_turn("It was cut 12%.")],
                retrieve=lambda q: calls.append(q) or RetrievalContext(chunks=[_chunk()]))
    out = _collect(r)
    assert calls == ["budget"]
    assert out["searches"] == ["budget"]
    assert out["answer"] == "It was cut 12%."
    assert len(out["cites"]) == 1


def test_leading_whitespace_never_reaches_the_answer(monkeypatch):
    """The model reliably opens with "\\n\\n"; that must not render as blank
    lines, and whitespace preceding a tool call must be dropped entirely."""
    r = _runner(monkeypatch, [_answer_turn("Real answer")])
    assert _collect(r)["answer"] == "Real answer"


def test_placeholder_query_is_rejected(monkeypatch):
    """Observed live: a model echoed an unfilled template into the query,
    which would send the retriever hunting for that literal string."""
    seen = []
    r = _runner(monkeypatch,
                [_tool_turn("<a better search query>"), _answer_turn("done")],
                retrieve=lambda q: seen.append(q) or RetrievalContext())
    _collect(r)
    assert seen == []  # retrieval never ran


def test_unknown_tool_does_not_kill_the_turn(monkeypatch):
    from personal_db import agent as agent_mod
    bad = [
        StreamEvent(kind="tool_calls",
                    tool_calls=[ToolCall(id="c", name="launch_missiles", arguments="{}")]),
        StreamEvent(kind="done", finish_reason="tool_calls"),
    ]
    r = _runner(monkeypatch, [bad, _answer_turn("recovered")])
    assert _collect(r)["answer"] == "recovered"


def test_retrieval_failure_is_reported_to_the_model_not_raised(monkeypatch):
    def boom(q):
        raise RuntimeError("qdrant down")
    r = _runner(monkeypatch, [_tool_turn("x"), _answer_turn("answered anyway")],
                retrieve=boom)
    assert _collect(r)["answer"] == "answered anyway"


def test_repeated_query_stops_the_loop(monkeypatch):
    """Re-running an exhausted query cannot produce new context, so the model
    is made to answer instead of burning another NPU round-trip."""
    r = _runner(monkeypatch,
                [_tool_turn("same", "c1"), _tool_turn("same", "c2"), _answer_turn("final")],
                max_tool_turns=3)
    out = _collect(r)
    assert out["searches"] == ["same", "same"]
    assert out["answer"] == "final"


def test_tool_turn_budget_is_enforced(monkeypatch):
    """With max_tool_turns=1 the model gets one search, then must answer."""
    r = _runner(monkeypatch,
                [_tool_turn("a"), _tool_turn("b"), _answer_turn("forced")],
                max_tool_turns=1)
    out = _collect(r)
    assert out["answer"] == "forced"
    assert len(out["searches"]) == 1


# ── fallback ───────────────────────────────────────────────────────────────

def test_falls_back_when_primary_server_unreachable(monkeypatch):
    """The agent talks to the model server directly, so it must carry its own
    fallback — otherwise moving off the LlamaIndex path silently drops the
    resilience the old pipeline had."""
    from personal_db import agent as agent_mod
    seen_hosts = []

    def fake_stream(messages, *, base_url, model, **kw):
        seen_hosts.append(base_url)
        if base_url == "http://primary":
            raise ConnectionError("connection refused")
        for ev in _answer_turn("from fallback"):
            yield ev

    monkeypatch.setattr(agent_mod, "stream_chat", fake_stream)
    r = agent_mod.AgentRunner(
        base_url="http://primary", model="m",
        retrieve=lambda q: RetrievalContext(),
        fallback_base_url="http://fallback", fallback_model="fb",
    )
    out = _collect(r)
    assert out["answer"] == "from fallback"
    assert seen_hosts == ["http://primary", "http://fallback"]


def test_midstream_failure_does_not_fall_back(monkeypatch):
    """Re-issuing after tokens were already shown would replay them to the
    user, so a mid-stream failure must propagate instead."""
    from personal_db import agent as agent_mod

    def fake_stream(messages, *, base_url, model, **kw):
        yield StreamEvent(kind="content", text="partial answer")
        raise ConnectionError("died mid-stream")

    monkeypatch.setattr(agent_mod, "stream_chat", fake_stream)
    r = agent_mod.AgentRunner(
        base_url="http://primary", model="m",
        retrieve=lambda q: RetrievalContext(),
        fallback_base_url="http://fallback", fallback_model="fb",
    )
    with pytest.raises(ConnectionError):
        _collect(r)


def test_no_fallback_configured_propagates(monkeypatch):
    from personal_db import agent as agent_mod

    def fake_stream(messages, **kw):
        raise ConnectionError("refused")
        yield  # pragma: no cover

    monkeypatch.setattr(agent_mod, "stream_chat", fake_stream)
    r = agent_mod.AgentRunner(base_url="http://p", model="m",
                              retrieve=lambda q: RetrievalContext())
    with pytest.raises(ConnectionError):
        _collect(r)


def test_reasoning_is_requested_only_on_the_opening_turn(monkeypatch):
    """Turn 0 decides whether/what to search and benefits from reasoning.
    Later turns already hold the passages and are just synthesising, so paying
    for reasoning there is wasted decode."""
    from personal_db import agent as agent_mod
    thinks = []
    turns = [_tool_turn("q"), _answer_turn("done")]

    def fake_stream(messages, *, base_url, model, think=None, **kw):
        thinks.append(think)
        for ev in turns.pop(0):
            yield ev

    monkeypatch.setattr(agent_mod, "stream_chat", fake_stream)
    r = agent_mod.AgentRunner(base_url="http://x", model="m",
                              retrieve=lambda q: RetrievalContext(chunks=[_chunk()]),
                              think_tools=True, think_answer=False)
    _collect(r)
    assert thinks == [True, False]


def test_tool_turns_are_token_capped_but_the_answer_is_not(monkeypatch):
    """A thinking turn can spiral to the context cap (measured: 4096 tokens /
    245s on one vague question). Tool turns are bounded; the answer needs room."""
    from personal_db import agent as agent_mod
    caps = []
    turns = [_tool_turn("q"), _answer_turn("done")]

    def fake_stream(messages, *, base_url, model, max_tokens=None, **kw):
        caps.append(max_tokens)
        for ev in turns.pop(0):
            yield ev

    monkeypatch.setattr(agent_mod, "stream_chat", fake_stream)
    r = agent_mod.AgentRunner(base_url="http://x", model="m",
                              retrieve=lambda q: RetrievalContext(chunks=[_chunk()]),
                              max_tool_turns=1,
                              tool_turn_max_tokens=768, answer_max_tokens=None)
    _collect(r)
    assert caps == [768, None]


def test_opening_turn_that_burns_its_cap_still_searches(monkeypatch):
    """If reasoning eats the whole budget the turn yields no tool call and no
    text. Rather than return an empty answer, search the user's own words."""
    from personal_db import agent as agent_mod
    seen = []
    turns = [
        [StreamEvent(kind="reasoning", text="spiralling..."),
         StreamEvent(kind="done", finish_reason="length")],
        _answer_turn("recovered answer"),
    ]

    def fake_stream(messages, **kw):
        for ev in turns.pop(0):
            yield ev

    monkeypatch.setattr(agent_mod, "stream_chat", fake_stream)
    r = agent_mod.AgentRunner(
        base_url="http://x", model="m",
        retrieve=lambda q: seen.append(q) or RetrievalContext(chunks=[_chunk()]),
    )
    out = _collect(r)
    assert seen == ["q"]                      # searched the raw user question
    assert out["searches"] == ["q"]
    assert out["answer"] == "recovered answer"


def test_overlong_query_is_truncated_not_rejected(monkeypatch):
    """With reasoning off the model drifts into keyword dumping — one voice turn
    emitted a 500-character query. Truncate: the leading words are the on-topic
    ones, and rejecting would cost another model round-trip."""
    from personal_db import agent as agent_mod
    seen = []
    dump = "LiveKit notes " + " ".join(f"keyword{i}" for i in range(80))
    r = _runner(monkeypatch, [_tool_turn(dump), _answer_turn("ok")],
                retrieve=lambda q: seen.append(q) or RetrievalContext(chunks=[_chunk()]))
    _collect(r)
    assert len(seen) == 1
    assert len(seen[0].split()) == agent_mod.MAX_QUERY_WORDS
    assert seen[0].startswith("LiveKit notes")


def test_short_query_is_left_alone(monkeypatch):
    seen = []
    r = _runner(monkeypatch, [_tool_turn("deployment ownership notes"), _answer_turn("ok")],
                retrieve=lambda q: seen.append(q) or RetrievalContext(chunks=[_chunk()]))
    _collect(r)
    assert seen == ["deployment ownership notes"]


# ── voice ──────────────────────────────────────────────────────────────────

def test_voice_greeting_reflects_whether_the_conversation_is_new(monkeypatch):
    """A call placed on a session that already has turns is a continuation —
    the agent carries that history into every turn, so opening with a
    fresh-start greeting misrepresents the state to the one participant who
    cannot see the screen."""
    from personal_db.voice import bot as bot_mod
    from personal_db.config import config

    monkeypatch.setattr(bot_mod.sess_store, "get_messages", lambda sid: [])
    assert bot_mod._greeting_for("s1") == config.voice_greeting

    monkeypatch.setattr(bot_mod.sess_store, "get_messages", lambda sid: ["a turn"])
    assert bot_mod._greeting_for("s1") == config.voice_resume_greeting


def test_voice_greeting_survives_a_broken_session_read(monkeypatch):
    """A greeting must never be the thing that kills a call."""
    from personal_db.voice import bot as bot_mod
    from personal_db.config import config

    def boom(sid):
        raise RuntimeError("db gone")

    monkeypatch.setattr(bot_mod.sess_store, "get_messages", boom)
    assert bot_mod._greeting_for("s1") == config.voice_greeting


# ── voice state bus ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_voice_state_events_reach_subscribers():
    import asyncio
    from personal_db.voice import events as v

    got = []

    async def reader():
        async for ev in v.subscribe("sess"):
            got.append(ev)
            if len(got) >= 2:
                break

    task = asyncio.create_task(reader())
    await asyncio.sleep(0.05)
    v.publish("sess", "hearing")
    v.publish("sess", "searching", query="q3 budget")
    await asyncio.wait_for(task, 2)
    assert got == [{"state": "hearing"},
                   {"state": "searching", "query": "q3 budget"}]


@pytest.mark.asyncio
async def test_voice_state_replays_current_state_to_a_late_subscriber():
    """A page that connects mid-call must show the current state, not nothing."""
    import asyncio
    from personal_db.voice import events as v

    v.publish("late", "speaking")
    got = []

    async def reader():
        async for ev in v.subscribe("late"):
            got.append(ev)
            break

    await asyncio.wait_for(asyncio.create_task(reader()), 2)
    assert got == [{"state": "speaking"}]
    v.clear("late")


def test_publishing_with_no_subscribers_is_harmless():
    """State publishing is fire-and-forget: it must never break a call."""
    from personal_db.voice import events as v
    v.publish("nobody", "listening")   # must not raise
    v.clear("nobody")
