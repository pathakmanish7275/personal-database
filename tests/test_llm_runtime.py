"""LLM runtime — primary→fallback wiring for chat and stream_chat."""

from __future__ import annotations

from unittest.mock import MagicMock


def _reset_fallback():
    """Force a fresh fallback Ollama instance per test (cached at module level)."""
    from personal_db import llm_runtime as lr
    lr._fallback_llm = None


def _patch_primary(monkeypatch, llm):
    from llama_index.core import Settings
    monkeypatch.setattr(Settings, "_llm", llm, raising=False)


def test_safe_chat_returns_primary_on_success(monkeypatch):
    from personal_db import config as cfg, llm_runtime as lr
    monkeypatch.setattr(cfg.config, "llm_provider", "gemini")
    _reset_fallback()
    primary = MagicMock()
    primary.chat.return_value = "OK_PRIMARY"
    _patch_primary(monkeypatch, primary)
    assert lr.safe_chat(["m"]) == "OK_PRIMARY"
    primary.chat.assert_called_once()


def test_safe_chat_falls_back_when_primary_raises_and_gemini_enabled(monkeypatch):
    from personal_db import config as cfg, llm_runtime as lr
    monkeypatch.setattr(cfg.config, "llm_provider", "gemini")
    _reset_fallback()
    primary = MagicMock()
    primary.chat.side_effect = RuntimeError("API down")
    _patch_primary(monkeypatch, primary)

    fb = MagicMock()
    fb.chat.return_value = "OK_FALLBACK"
    monkeypatch.setattr(lr, "_get_fallback", lambda: fb)

    assert lr.safe_chat(["m"]) == "OK_FALLBACK"
    fb.chat.assert_called_once()


def test_safe_chat_does_not_fall_back_when_primary_and_fallback_share_host(monkeypatch):
    """provider=ollama with chat LLM on the same host as the fallback: nothing
    to fall back to, so the error propagates."""
    from personal_db import config as cfg, llm_runtime as lr
    monkeypatch.setattr(cfg.config, "llm_provider", "ollama")
    monkeypatch.setattr(cfg.config, "llm_host", "http://localhost:11434")
    monkeypatch.setattr(cfg.config, "fallback_llm_host", "http://localhost:11434")
    _reset_fallback()
    primary = MagicMock()
    primary.chat.side_effect = RuntimeError("ollama down")
    _patch_primary(monkeypatch, primary)

    fb = MagicMock()
    monkeypatch.setattr(lr, "_get_fallback", lambda: fb)

    import pytest
    with pytest.raises(RuntimeError):
        lr.safe_chat(["m"])
    fb.chat.assert_not_called()


def test_safe_chat_falls_back_to_local_ollama_when_flm_primary_down(monkeypatch):
    """provider=ollama with the chat LLM on a separate server (FLM): if the
    primary server dies, fall back to local Ollama."""
    from personal_db import config as cfg, llm_runtime as lr
    monkeypatch.setattr(cfg.config, "llm_provider", "ollama")
    monkeypatch.setattr(cfg.config, "llm_host", "http://localhost:52625")
    monkeypatch.setattr(cfg.config, "fallback_llm_host", "http://localhost:11434")
    _reset_fallback()
    primary = MagicMock()
    primary.chat.side_effect = RuntimeError("connection refused")
    _patch_primary(monkeypatch, primary)

    fb = MagicMock()
    fb.chat.return_value = "OK_LOCAL"
    monkeypatch.setattr(lr, "_get_fallback", lambda: fb)

    assert lr.safe_chat(["m"]) == "OK_LOCAL"
    fb.chat.assert_called_once()


def test_safe_stream_chat_peeks_first_chunk_and_yields_rest(monkeypatch):
    from personal_db import config as cfg, llm_runtime as lr
    monkeypatch.setattr(cfg.config, "llm_provider", "gemini")
    _reset_fallback()

    primary = MagicMock()
    primary.stream_chat.return_value = iter(["a", "b", "c"])
    _patch_primary(monkeypatch, primary)

    out = list(lr.safe_stream_chat(["m"]))
    assert out == ["a", "b", "c"]


def test_safe_stream_chat_falls_back_when_primary_errors_at_first_chunk(monkeypatch):
    from personal_db import config as cfg, llm_runtime as lr
    monkeypatch.setattr(cfg.config, "llm_provider", "gemini")
    _reset_fallback()

    def boom():
        raise RuntimeError("auth failed")
        yield  # pragma: no cover

    primary = MagicMock()
    primary.stream_chat.return_value = boom()
    _patch_primary(monkeypatch, primary)

    fb = MagicMock()
    fb.stream_chat.return_value = iter(["x", "y"])
    monkeypatch.setattr(lr, "_get_fallback", lambda: fb)

    out = list(lr.safe_stream_chat(["m"]))
    assert out == ["x", "y"]
    fb.stream_chat.assert_called_once()


# ── In-band reasoning extraction ────────────────────────────────────────────
# FLM hands back the model's raw reply, so reasoning arrives inside `content`
# in whichever convention the model uses. Both must reduce to just the answer;
# anything that leaks reaches the UI, SQLite history, the planner and summaries.

def test_strip_think_removes_paired_block():
    from personal_db.llm_runtime import _strip_think
    assert _strip_think("<think>reason</think>Answer.") == "Answer."
    assert _strip_think("<thinking>a</thinking>X<reasoning>b</reasoning>Y") == "XY"


def test_strip_think_removes_unterminated_block():
    """qwen spiralling into its output cap emits no closing tag — the paired
    regex matches nothing and the whole reasoning dump would leak."""
    from personal_db.llm_runtime import _strip_think
    assert _strip_think("<think>reason that never closes") == ""
    assert _strip_think("<think>done</think>Real answer<think>cut") == "Real answer"


def test_strip_think_extracts_harmony_final_channel():
    """gpt-oss speaks harmony; FLM does not parse it, so the whole channel
    transcript arrives in content and the answer is the `final` channel."""
    from personal_db.llm_runtime import _strip_think
    raw = (
        "<|start|>assistant<|channel|>analysis<|message|>thinking<|end|>"
        "<|start|>assistant<|channel|>final<|message|>PONG<|end|>"
    )
    assert _strip_think(raw) == "PONG"
    assert _strip_think("<|channel|>final<|message|>Done<|return|>") == "Done"


def test_strip_think_handles_constrained_final_channel():
    """Observed from gpt-oss:20b on FLM: a constrained reply puts a
    <|constrain|> token between the channel header and the message body.
    Requiring them to be adjacent drops the answer and wrongly triggers the
    fallback — every constrained answer would be discarded."""
    from personal_db.llm_runtime import _strip_think
    assert _strip_think(
        "<|channel|>final <|constrain|>answer<|message|>Sam<|end|>"
    ) == "Sam"
    assert _strip_think(
        "<|channel|>analysis<|message|>r<|end|>"
        "<|channel|>final <|constrain|>json<|message|>{\"a\":1}<|return|>"
    ) == '{"a":1}'


def test_strip_think_harmony_without_final_channel_is_empty():
    """Truncated mid-analysis: no answer was reached. Must report empty rather
    than leak the analysis transcript — the caller recovers on that signal."""
    from personal_db.llm_runtime import _strip_think
    assert _strip_think("<|channel|>analysis<|message|>thinking and cut") == ""


def test_strip_think_leaves_plain_text_alone():
    from personal_db.llm_runtime import _strip_think
    assert _strip_think("Just an answer.") == "Just an answer."
    assert _strip_think("") == ""


def test_safe_chat_falls_back_when_primary_returns_only_reasoning(monkeypatch):
    """Primary produced bytes but no answer survived extraction → treat it as a
    failed turn and retry once on the fallback."""
    from personal_db import config as cfg, llm_runtime as lr
    from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole
    monkeypatch.setattr(cfg.config, "llm_provider", "ollama")
    monkeypatch.setattr(cfg.config, "llm_host", "http://localhost:52625")
    monkeypatch.setattr(cfg.config, "fallback_llm_host", "http://localhost:11434")
    _reset_fallback()

    truncated = ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content="<think>no close"),
        raw={"done_reason": "length"},
    )
    primary = MagicMock()
    primary.chat.return_value = truncated
    _patch_primary(monkeypatch, primary)

    answer = ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content="real answer"),
        raw={"done_reason": "stop"},
    )
    fb = MagicMock()
    fb.chat.return_value = answer
    monkeypatch.setattr(lr, "_get_fallback", lambda: fb)

    assert lr.safe_chat(["m"]).message.content == "real answer"
    fb.chat.assert_called_once()


def test_safe_chat_keeps_primary_answer_when_extraction_succeeds(monkeypatch):
    """A normal reasoning+answer reply must NOT trigger the fallback."""
    from personal_db import config as cfg, llm_runtime as lr
    from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole
    monkeypatch.setattr(cfg.config, "llm_provider", "ollama")
    monkeypatch.setattr(cfg.config, "llm_host", "http://localhost:52625")
    monkeypatch.setattr(cfg.config, "fallback_llm_host", "http://localhost:11434")
    _reset_fallback()

    primary = MagicMock()
    primary.chat.return_value = ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content="<think>r</think>Answer."),
        raw={"done_reason": "stop"},
    )
    _patch_primary(monkeypatch, primary)
    fb = MagicMock()
    monkeypatch.setattr(lr, "_get_fallback", lambda: fb)

    assert lr.safe_chat(["m"]).message.content == "Answer."
    fb.chat.assert_not_called()
