"""Memory compaction logic (the agent chain)."""

from __future__ import annotations

from unittest.mock import MagicMock


def _seed(s, sid, n_pairs: int, body: str = "x" * 400) -> list[int]:
    """Append n user/assistant pairs of length `body` chars each. Returns msg ids."""
    ids: list[int] = []
    for i in range(n_pairs):
        ids.append(s.add_message(sid, "user", f"u{i}: {body}"))
        ids.append(s.add_message(sid, "assistant", f"a{i}: {body}"))
    return ids


def test_token_estimate_rough():
    from personal_db.memory import _estimate_tokens
    assert _estimate_tokens("") == 1
    # chars/4 heuristic
    assert _estimate_tokens("a" * 40) == 10


def test_plan_under_budget_no_compaction(isolated_paths, monkeypatch):
    """A session with little history should NOT plan a compaction."""
    from personal_db import config as cfg_mod, memory as mem, sessions as s
    monkeypatch.setattr(cfg_mod.config, "memory_token_budget", 100_000)
    sess = s.create_session()
    _seed(s, sess.id, 3, body="hi")
    _, _, _, to_compress, _, compress_turns = mem._plan(sess.id)
    assert to_compress == []
    assert compress_turns == 0
    assert mem.peek_compaction(sess.id) == 0


def test_plan_over_budget_selects_targets(isolated_paths, monkeypatch):
    from personal_db import config as cfg_mod, memory as mem, sessions as s
    monkeypatch.setattr(cfg_mod.config, "memory_token_budget", 50)        # very tight
    monkeypatch.setattr(cfg_mod.config, "memory_keep_recent_turns", 2)    # keep last 2 turns
    sess = s.create_session()
    _seed(s, sess.id, 6, body="x" * 200)
    n = mem.peek_compaction(sess.id)
    assert n > 0
    assert n == 6 - 2  # 6 turns minus the 2 kept verbatim → fold 4 turns


def test_plan_folds_whole_turns_not_split_pairs(isolated_paths, monkeypatch):
    """Compaction must keep user+assistant pairs together (the old bug folded a
    lone user message, producing near-empty summaries)."""
    from personal_db import config as cfg_mod, memory as mem, sessions as s
    monkeypatch.setattr(cfg_mod.config, "memory_token_budget", 10)        # force over budget
    monkeypatch.setattr(cfg_mod.config, "memory_keep_recent_turns", 2)
    sess = s.create_session()
    _seed(s, sess.id, 5, body="word " * 50)
    _, _, _, to_compress, to_keep, compress_turns = mem._plan(sess.id)
    # 5 turns, keep 2 → fold 3 whole turns = 6 messages (3 user + 3 assistant).
    assert compress_turns == 3
    assert len(to_compress) == 6
    roles = [m.role for m in to_compress]
    assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]
    # Kept window is the last 2 whole turns = 4 messages.
    assert len(to_keep) == 4


def test_compact_runs_summarizer_and_persists(isolated_paths, monkeypatch, patch_settings):
    from personal_db import config as cfg_mod, memory as mem, sessions as s
    from llama_index.core import Settings
    monkeypatch.setattr(cfg_mod.config, "memory_token_budget", 20)
    monkeypatch.setattr(cfg_mod.config, "memory_keep_recent_turns", 2)
    sess = s.create_session()
    _seed(s, sess.id, 4, body="hello world " * 30)

    result = mem.compact_if_needed(sess.id)

    assert result.did_compact is True
    assert result.compressed_count > 0
    assert "MOCK_SUMMARY" in result.summary
    # Persisted
    fresh = s.get_session(sess.id)
    assert "MOCK_SUMMARY" in fresh.summary
    assert fresh.summary_up_to_msg_id > 0
    # LLM was invoked exactly once
    assert len(Settings.llm.chat_calls) == 1


def test_compact_when_no_session_is_a_noop(isolated_paths):
    from personal_db import memory as mem
    r = mem.compact_if_needed("doesnotexist")
    assert r.did_compact is False
    assert r.summary == ""
    assert r.recent == []
