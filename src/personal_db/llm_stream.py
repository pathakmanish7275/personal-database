"""Streaming OpenAI-compatible chat client for FLM and Ollama.

Why this exists instead of the LlamaIndex ``Ollama`` client: tool calling only
works on the **OpenAI-format** endpoint. FLM's Ollama-format ``/api/chat``
silently *drops* the ``tools`` array — the model never learns the tools exist
and apologises that it cannot reach your notes. ``/v1/chat/completions``
returns proper structured ``tool_calls``. (Verified against FLM v1.0.5.)

Wire behaviour this module relies on, all measured rather than assumed:

* ``think: true`` puts reasoning in its **own delta field**, never inside
  ``content`` — so there is no ``<think>`` tag to strip out of the token
  stream. FLM calls the field ``reasoning_content``; Ollama calls it
  ``reasoning``. Both are accepted.
* Reasoning deltas always arrive **before** any content or tool-call delta.
* Content emitted *before* a tool call is whitespace only.
* ``finish_reason`` is ``tool_calls`` when the model wants a tool, ``stop``
  when it has answered, and ``length`` when it ran out of output budget.
* ``reasoning_effort`` has **no effect** on FLM server mode. Neither
  ``reasoning_effort``, ``options.reasoning_effort`` nor a ``Reasoning: low``
  system line changes gpt-oss's reasoning length by more than the run-to-run
  noise (measured n=3: 33-136s spread *within* one condition). FLM documents
  ``/set r-eff`` for CLI mode only, and v1.0.6 does not add a server-mode
  equivalent. Reasoning cost on gpt-oss is therefore fixed.

Tool-call deltas are accumulated by index in the standard OpenAI way. FLM
happens to send each call complete in a single delta, but Ollama may fragment
``arguments`` across deltas, so we never rely on getting it in one piece.
"""

from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Iterator, Literal
from urllib.parse import urlparse

import httpx

from .llm_runtime import call_gate

log = logging.getLogger(__name__)

# Field names for the separated reasoning channel, in preference order.
_REASONING_KEYS = ("reasoning_content", "reasoning")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str = ""

    def parsed_arguments(self) -> dict:
        """Arguments as a dict. Returns {} rather than raising — a malformed
        tool call must degrade into "no usable call", not kill the turn."""
        try:
            out = json.loads(self.arguments or "{}")
            return out if isinstance(out, dict) else {}
        except json.JSONDecodeError:
            log.warning("tool call %s had unparseable arguments: %r", self.name, self.arguments)
            return {}


@dataclass
class StreamEvent:
    """One thing that happened on the wire."""

    kind: Literal["reasoning", "content", "tool_calls", "done"]
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict | None = None


class _ToolCallAccumulator:
    """Merges OpenAI-style streamed tool-call deltas, keyed by index."""

    def __init__(self) -> None:
        self._by_index: dict[int, ToolCall] = {}

    def feed(self, deltas: list[dict]) -> None:
        for d in deltas or []:
            idx = d.get("index", 0)
            fn = d.get("function") or {}
            call = self._by_index.get(idx)
            if call is None:
                call = ToolCall(id=d.get("id") or f"call_{idx}", name=fn.get("name") or "")
                self._by_index[idx] = call
            # Later deltas may fill in or extend any of these.
            if d.get("id"):
                call.id = d["id"]
            if fn.get("name"):
                call.name = fn["name"]
            if fn.get("arguments"):
                call.arguments += fn["arguments"]

    def result(self) -> list[ToolCall]:
        return [self._by_index[i] for i in sorted(self._by_index)]


def _reasoning_of(delta: dict) -> str | None:
    for key in _REASONING_KEYS:
        if delta.get(key):
            return delta[key]
    return None


def _is_local(base_url: str) -> bool:
    """Whether `base_url` points at a server sharing this machine's accelerator."""
    host = urlparse(base_url).hostname or ""
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")


def stream_chat(
    messages: list[dict],
    *,
    base_url: str,
    model: str,
    tools: list[dict] | None = None,
    think: bool | None = None,
    max_tokens: int | None = None,
    timeout: float = 600.0,
    api_key: str | None = None,
    extra_body: dict | None = None,
    serialize: bool | None = None,
) -> Iterator[StreamEvent]:
    """Stream one chat completion, yielding events as they arrive.

    `base_url` is the server root (e.g. http://localhost:52625); "/v1/chat/
    completions" is appended. The final event is always kind="done" and carries
    `finish_reason` plus `usage`, even when the stream ends without either —
    callers can rely on seeing exactly one terminal event.

    `serialize` controls the process-wide call gate, which exists to keep two
    requests off the NPU at once. It defaults to whether `base_url` is local:
    a remote endpoint has its own capacity, so holding the gate for it would
    both serialise calls that could run concurrently and block local chat for
    the duration. Pass it explicitly to override.
    """
    payload: dict = {"model": model, "stream": True, "messages": messages}
    if tools:
        payload["tools"] = tools
    if think is not None:
        payload["think"] = think
    if max_tokens:
        # Caps reasoning + content together, and FLM honours it (`num_predict`
        # is ignored). Needed because a thinking turn can otherwise spiral to
        # the context cap — measured at 4096 tokens / 245s on one vague
        # question, all of it inside the reasoning channel.
        payload["max_tokens"] = max_tokens

    if extra_body:
        payload.update(extra_body)

    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    if serialize is None:
        serialize = _is_local(base_url)
    acc = _ToolCallAccumulator()
    finish_reason: str | None = None
    usage: dict | None = None

    # Held for the whole request, so this client and the LlamaIndex path share
    # one guarantee: never two requests in flight to the NPU at once. A remote
    # endpoint is exempt — see `serialize`.
    with (call_gate() if serialize else nullcontext()):
        with httpx.Client(timeout=timeout, headers=headers) as client:
            with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    body = line[len("data:") :].strip()
                    if body == "[DONE]":
                        break
                    try:
                        chunk = json.loads(body)
                    except json.JSONDecodeError:
                        log.warning("undecodable SSE chunk: %r", body[:200])
                        continue

                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}

                    reasoning = _reasoning_of(delta)
                    if reasoning:
                        yield StreamEvent(kind="reasoning", text=reasoning)
                    if delta.get("content"):
                        yield StreamEvent(kind="content", text=delta["content"])
                    if delta.get("tool_calls"):
                        acc.feed(delta["tool_calls"])
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

    calls = acc.result()
    if calls:
        yield StreamEvent(kind="tool_calls", tool_calls=calls)
    yield StreamEvent(kind="done", finish_reason=finish_reason, usage=usage)
