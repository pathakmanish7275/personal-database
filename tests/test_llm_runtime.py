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


def test_safe_chat_does_not_fall_back_when_provider_is_ollama(monkeypatch):
    from personal_db import config as cfg, llm_runtime as lr
    monkeypatch.setattr(cfg.config, "llm_provider", "ollama")
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
