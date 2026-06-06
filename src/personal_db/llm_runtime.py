"""LLM call wrappers with provider fallback.

When the configured primary LLM (e.g. Gemini Flash) is up, all chat / stream
calls go through it. If the primary raises (network error, API outage, rate
limit, parse error), we transparently fall back to the local Ollama model so
the assistant keeps answering. Fallback only kicks in when the primary is
provider=gemini — when provider=ollama, there's nothing to fall back to.

Mid-stream failures (an exception thrown after some tokens have already been
yielded) cannot be safely recovered without re-issuing the prompt and replaying
partial output; those propagate to the caller. We do detect failures during
the *first* chunk (which catches most cold-start / auth / quota errors) by
peeking at the iterator before returning it.
"""

from __future__ import annotations

import logging
from typing import Iterable

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

from .config import config

log = logging.getLogger(__name__)

_fallback_llm = None


def _get_fallback() -> Ollama:
    """Lazily build the local Ollama fallback. Cached after first use."""
    global _fallback_llm
    if _fallback_llm is None:
        _fallback_llm = Ollama(
            model=config.llm_model,
            base_url=config.ollama_host,
            request_timeout=600.0,
            context_window=32768,
        )
    return _fallback_llm


def _fallback_enabled() -> bool:
    return (config.llm_provider or "ollama").lower() == "gemini"


def safe_chat(messages):
    """Synchronous chat — used by the memory summarizer. Returns a ChatResponse."""
    try:
        return Settings.llm.chat(messages)
    except Exception as e:  # noqa: BLE001
        if not _fallback_enabled():
            raise
        log.warning("primary LLM chat failed (%s); falling back to Ollama", e)
        return _get_fallback().chat(messages)


def safe_stream_chat(messages) -> Iterable:
    """Streaming chat that survives initial failures of the primary provider.

    We peek the first chunk so auth / quota / network errors are caught before
    we hand the iterator back to the caller. Mid-stream failures still propagate."""
    if not _fallback_enabled():
        return Settings.llm.stream_chat(messages)

    try:
        primary = Settings.llm.stream_chat(messages)
        it = iter(primary)
        first = next(it)

        def merged():
            yield first
            yield from it

        return merged()
    except StopIteration:
        return iter(())
    except Exception as e:  # noqa: BLE001
        log.warning("primary LLM stream_chat failed (%s); falling back to Ollama", e)
        return _get_fallback().stream_chat(messages)
