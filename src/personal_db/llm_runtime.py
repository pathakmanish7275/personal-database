"""LLM call wrappers with provider fallback and global call serialization.

Two invariants live here:

1. **One request at a time to the model server.** A process-wide gate
   (`threading.Lock`) is held for the full duration of every LLM call —
   including the entire consumption of a stream. The job system's asyncio
   run-lock queues *requests* before they start; this gate guarantees that
   even calls from outside the job system can never overlap. The model
   server's own queue (FLM: "NPU busy, request queued") therefore stays
   empty — waiting happens in our system, never on the NPU server.

2. **Fallback.** When the configured primary LLM (e.g. Gemini Flash) is up,
   all chat / stream calls go through it. If the primary raises (network
   error, API outage, rate limit, parse error), we transparently fall back
   to the local Ollama model so the assistant keeps answering. Fallback only
   kicks in when the primary is remote (gemini, or ollama on a separate
   host); there's nothing to fall back to when primary == fallback host.

Mid-stream failures (an exception thrown after some tokens have already been
yielded) cannot be safely recovered without re-issuing the prompt and replaying
partial output; those propagate to the caller. We do detect failures during
the *first* chunk (which catches most cold-start / auth / quota errors) by
peeking at the iterator before returning it.
"""

from __future__ import annotations

import logging
import re
import threading
from contextlib import contextmanager
from typing import Iterable

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

from .config import config

log = logging.getLogger(__name__)

_fallback_llm = None

# Global serialization gate: at most one LLM request in flight process-wide.
# Held across the entire stream consumption, not just stream creation.
_call_gate = threading.Lock()

# In-flight LLM call tracking. Lets the shutdown path wait for active
# generations to finish instead of tearing the connection down mid-call
# (aborted streams can crash the FLM/NPU server).
_active_calls = 0
_active_lock = threading.Lock()

# FLM v1.0.5 ignores `think: false` and emits the model's reasoning as
# think-tag blocks inside the message content. The blocks must be stripped
# (non-greedy, multi-occurrence replace) so chat history, JSON planners and
# session summaries see clean text.
#
# This is a best-effort repair for the in-band case. FLM exposes a structured
# path — thinking separated from content — when `think: true` is sent, so
# LLM_THINKING=true with an FLM primary gets clean content at the source. The
# strip below is kept as defence-in-depth for whatever leaks through.
_THINK_BLOCK_RE = re.compile(
    r"<\s*(think(?:ing)?|reasoning)\b[^>]*>.*?<\s*/\s*\1\s*>",
    re.DOTALL | re.IGNORECASE,
)

# An *unterminated* block. When the model spirals it hits the output cap
# (done_reason="length") mid-reasoning, so no closing tag is ever emitted and
# the paired regex above matches nothing — letting the entire raw reasoning
# dump through into the UI, chat history, planner JSON and summaries.
# Measured: 14.6 KB of reasoning, zero answer. Strip from the tag to the end.
_OPEN_THINK_RE = re.compile(
    r"<\s*(?:think(?:ing)?|reasoning)\b[^>]*>.*\Z",
    re.DOTALL | re.IGNORECASE,
)


# gpt-oss models speak the "harmony" format: the reply is a transcript of
# tagged channels, reasoning in `analysis` and the actual answer in `final`:
#   <|start|>assistant<|channel|>analysis<|message|>...<|end|>
#   <|start|>assistant<|channel|>final<|message|>the answer<|end|>
# FLM does not parse these, so the whole transcript arrives in `content`.
# Extracting `final` is a positive match (take the answer) rather than a strip,
# so it degrades safely: no `final` channel means the model never reached an
# answer, which the caller treats as empty and recovers from.
_HARMONY_MARKER = "<|channel|>"
# The channel header may carry extra control tokens before the message body —
# e.g. a constrained reply is `<|channel|>final <|constrain|>answer<|message|>`.
# Skip anything that is not itself <|message|>, or those answers are dropped.
_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>\s*final\b"
    r"(?:(?!<\|message\|>)[\s\S])*?"
    r"<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|\Z)",
    re.DOTALL,
)


def _strip_think(text: str) -> str:
    """Reduce a raw model reply to just its answer.

    Handles both in-band conventions we see on FLM: gpt-oss harmony channels
    and qwen-style think tags (paired or truncated)."""
    text = text or ""
    if _HARMONY_MARKER in text:
        finals = _HARMONY_FINAL_RE.findall(text)
        # No final channel → only reasoning was emitted; report empty so the
        # caller can recover rather than leaking the analysis transcript.
        return "\n".join(f.strip() for f in finals).strip()
    cleaned = _THINK_BLOCK_RE.sub("", text)
    # Only after paired blocks are gone, so a well-formed block followed by a
    # second truncated one is still handled.
    return _OPEN_THINK_RE.sub("", cleaned).strip()


def _content_of(resp) -> str:
    """Best-effort message text. Returns "" for anything not shaped like a
    ChatResponse (test doubles, odd provider payloads) so callers can branch
    on it without guarding every access."""
    try:
        return str(resp.message.content or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _clean_response(resp):
    """Return a ChatResponse with think blocks removed from message + delta."""
    try:
        orig_content = str(resp.message.content or "")
        clean_content = _strip_think(orig_content)
        clean_delta = None if resp.delta is None else _strip_think(resp.delta)
        if clean_content == orig_content and clean_delta == resp.delta:
            return resp  # nothing to strip — return original untouched
        # `content` is a read-only property over `blocks`, so model_copy with a
        # "content" key does nothing. Rebuild the message via its constructor.
        new_msg = resp.message.__class__(
            role=resp.message.role,
            content=clean_content,
            additional_kwargs=resp.message.additional_kwargs,
        )
        return resp.model_copy(update={"message": new_msg, "delta": clean_delta})
    except Exception:  # noqa: BLE001 — best-effort sanitisation
        return resp


@contextmanager
def call_gate():
    """Hold the process-wide LLM gate for one model request.

    Exported so callers that talk to the model server directly (the agent's
    OpenAI-endpoint client, which the LlamaIndex path cannot serve) are covered
    by the same guarantee: never two requests in flight to the NPU at once."""
    with _call_gate:
        _begin_call()
        try:
            yield
        finally:
            _end_call()


def _begin_call() -> None:
    global _active_calls
    with _active_lock:
        _active_calls += 1


def _end_call() -> None:
    global _active_calls
    with _active_lock:
        _active_calls -= 1


def active_calls() -> int:
    """Number of LLM calls currently in flight (chat or stream)."""
    with _active_lock:
        return _active_calls


def _get_fallback() -> Ollama:
    """Lazily build the local Ollama fallback. Cached after first use."""
    global _fallback_llm
    if _fallback_llm is None:
        _fallback_llm = Ollama(
            model=config.fallback_llm_model,
            base_url=config.fallback_llm_host,
            request_timeout=600.0,
            context_window=32768,
            thinking=config.llm_thinking,
        )
    return _fallback_llm


def _fallback_enabled() -> bool:
    provider = (config.llm_provider or "ollama").lower()
    if provider == "gemini":
        return True
    # provider=ollama with the chat LLM on a separate server (e.g. FLM):
    # fall back to the local Ollama host if the primary server is down.
    return (
        provider == "ollama"
        and config.llm_host.rstrip("/") != config.fallback_llm_host.rstrip("/")
    )


def _primary_chat_with_recovery(messages):
    """One primary `chat()`, with fallback for both failure modes.

    Hard failure (connection refused, auth, timeout) → fall back.
    Soft failure (FLM spiralled into reasoning until it hit the output cap, so
    the answer is empty once the truncated think block is stripped) → fall back
    too: the primary produced no usable answer, only a reasoning dump."""
    try:
        resp = Settings.llm.chat(messages)
    except Exception as e:  # noqa: BLE001
        if not _fallback_enabled():
            raise
        log.warning("primary LLM chat failed (%s); falling back to Ollama", e)
        return _clean_response(_get_fallback().chat(messages))

    cleaned = _clean_response(resp)
    # "Produced bytes, but no answer survived extraction" means the model spent
    # its whole budget reasoning — a truncated think block, or harmony analysis
    # with no final channel. Either way the primary gave us nothing usable.
    produced_only_reasoning = bool(_content_of(resp)) and not _content_of(cleaned)
    if produced_only_reasoning and _fallback_enabled():
        log.warning(
            "primary LLM returned reasoning but no answer (done_reason=%s); "
            "retrying once on the Ollama fallback",
            (resp.raw or {}).get("done_reason") if isinstance(resp.raw, dict) else "?",
        )
        return _clean_response(_get_fallback().chat(messages))
    return cleaned


def safe_chat(messages):
    """Synchronous chat — used by the memory summarizer. Returns a ChatResponse."""
    with _call_gate:
        _begin_call()
        try:
            return _primary_chat_with_recovery(messages)
        finally:
            _end_call()


def safe_stream_chat(messages) -> Iterable:
    """Streaming chat that survives initial failures of the primary provider.

    Returns a lazy generator: the request is issued (and the gate acquired)
    on first iteration and the gate is held until the stream is fully
    consumed. Failures during the first chunk (auth / quota / network /
    cold-start) transparently fall back to Ollama; mid-stream failures still
    propagate."""
    return _safe_stream(messages)


def _primary_stream_is_unsafe() -> bool:
    """True when the primary server's streaming endpoint is known-broken.

    FLM v1.0.5 deadlocks the whole server (any endpoint, stream=true) at
    'Creating checkpoint' — process alive, API dead. Sync chat on the same
    server is stable, so for FLM primaries we emulate streaming with sync
    calls instead of taking the NPU server down mid-conversation."""
    provider = (config.llm_provider or "ollama").lower()
    return (
        provider == "ollama"
        and config.llm_host.rstrip("/") != config.ollama_host.rstrip("/")
    )


def _safe_stream(messages) -> Iterable:
    with _call_gate:
        _begin_call()
        try:
            if _primary_stream_is_unsafe():
                # Emulated stream: one sync call, yielded as a single chunk.
                # The sync response has delta=None (streaming-only field),
                # so surface the cleaned message content as the delta.
                # _primary_chat_with_recovery has already cleaned it and
                # handled both the hard- and output-cap failure modes.
                resp = _primary_chat_with_recovery(messages)
                if resp.delta:
                    yield resp
                else:
                    yield resp.__class__(
                        message=resp.message,
                        raw=resp.raw,
                        delta=str(resp.message.content or ""),
                    )
                return
            if not _fallback_enabled():
                yield from Settings.llm.stream_chat(messages)
                return
            try:
                it = iter(Settings.llm.stream_chat(messages))
                first = next(it)
            except StopIteration:
                return
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "primary LLM stream_chat failed (%s); falling back to Ollama", e
                )
                yield from _get_fallback().stream_chat(messages)
                return
            yield first
            yield from it
        finally:
            _end_call()
