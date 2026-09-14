"""The agentic chat loop.

Replaces the old fixed plan→retrieve→synthesise pipeline. The model is given
`search_kb` as a real tool and decides for itself whether to search, how to
phrase the query, whether to search again, or whether to answer directly
(chit-chat and follow-ups skip retrieval entirely). Asking the user a
clarifying question needs no special path any more — it is simply the model
answering without calling the tool.

Streaming contract, measured against FLM v1.0.5 (see llm_stream):

* reasoning arrives in its own delta field and never enters `content`
* reasoning deltas always precede content and tool-call deltas
* content emitted *before* a tool call is whitespace only

That last point is why content is buffered until the turn is classified: we
must not emit a stray newline into the UI and then discover the turn was a
tool call. Once a turn is known to be an answer, tokens stream through live.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterator

from .llm_stream import ToolCall, stream_chat
from .retrieve import RetrievalContext
from .tools import SEARCH_KB, TOOL_SCHEMAS, CitationLedger, format_hits

log = logging.getLogger(__name__)


@dataclass
class AgentEvent:
    """What the caller should do about it."""

    kind: str  # "reasoning" | "token" | "searching" | "citations" | "done"
    text: str = ""
    query: str = ""
    cites: list[dict] = field(default_factory=list)
    finish_reason: str | None = None


def _last_user_text(messages: list[dict]) -> str:
    """The most recent user message — used as a fallback search query."""
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            return str(m["content"]).strip()
    return ""


# Retrieval is semantic: a focused phrase embeds well, a long keyword dump
# embeds to mush. Models drift into dumping when reasoning is off — one voice
# turn produced a 500-character run of loosely related terms. Truncating beats
# rejecting: the leading words are the on-topic ones, so a bounded query still
# retrieves, where a rejection costs another model round-trip.
MAX_QUERY_WORDS = 24


def _bound_query(q: str) -> str:
    words = q.split()
    if len(words) <= MAX_QUERY_WORDS:
        return q
    trimmed = " ".join(words[:MAX_QUERY_WORDS])
    log.warning(
        "search query was %d words; truncated to %d: %r",
        len(words), MAX_QUERY_WORDS, trimmed,
    )
    return trimmed


def _looks_like_placeholder(q: str) -> bool:
    """Reject a query the model copied out of its own instructions.

    Observed in testing: a model handed a `<a better search query>` example
    echoed the placeholder verbatim, which would send the retriever hunting for
    that literal string."""
    return "<" in q and ">" in q


class AgentRunner:
    """Drives the tool loop for one user turn.

    `retrieve` is injected (rather than imported) so the loop is testable
    without Qdrant/Kuzu, and so the caller controls threading."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        retrieve: Callable[[str], RetrievalContext],
        max_tool_turns: int = 2,
        think_tools: bool | None = True,
        think_answer: bool | None = False,
        fallback_base_url: str | None = None,
        fallback_model: str | None = None,
        tool_turn_max_tokens: int | None = 768,
        answer_max_tokens: int | None = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.retrieve = retrieve
        self.max_tool_turns = max_tool_turns
        self.think_tools = think_tools
        self.think_answer = think_answer
        self.fallback_base_url = fallback_base_url
        self.fallback_model = fallback_model
        self.tool_turn_max_tokens = tool_turn_max_tokens
        self.answer_max_tokens = answer_max_tokens
        self.ledger = CitationLedger()
        self._using_fallback = False

    def _stream(self, messages: list[dict], *, think: bool | None, max_tokens: int | None):
        """Stream one request, falling back if the primary server is unreachable.

        Fallback only applies when the primary fails *before* emitting anything.
        A mid-stream failure cannot be recovered without re-issuing the prompt
        and replaying tokens the caller has already shown, so it propagates.
        Once we have fallen back we stay there for the rest of the turn —
        switching servers mid-conversation would split the tool-call history
        across two models."""
        kwargs = dict(tools=TOOL_SCHEMAS, think=think, max_tokens=max_tokens)
        if not self._using_fallback:
            emitted = False
            try:
                for ev in stream_chat(
                    messages, base_url=self.base_url, model=self.model, **kwargs
                ):
                    emitted = True
                    yield ev
                return
            except Exception as e:  # noqa: BLE001
                if emitted or not (self.fallback_base_url and self.fallback_model):
                    raise
                log.warning(
                    "primary model server failed (%s); falling back to %s",
                    e, self.fallback_base_url,
                )
                self._using_fallback = True

        yield from stream_chat(
            messages,
            base_url=self.fallback_base_url,
            model=self.fallback_model,
            **kwargs,
        )

    # ── one streamed turn ──────────────────────────────────────────────────

    def _run_turn(self, messages: list[dict], *, think: bool | None, max_tokens: int | None = None):
        """Stream one model turn.

        Yields ("reasoning"|"token", text) as they become safe to show, and
        finally returns via `self._last_turn` the tool calls / finish reason.
        Content is withheld until the turn is classified as an answer."""
        pending: list[str] = []
        classified = False
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        finish_reason: str | None = None

        for ev in self._stream(messages, think=think, max_tokens=max_tokens):
            if ev.kind == "reasoning":
                yield ("reasoning", ev.text)
            elif ev.kind == "content":
                content_parts.append(ev.text)
                if classified:
                    yield ("token", ev.text)
                else:
                    pending.append(ev.text)
                    # Anything non-whitespace means this is a real answer, not
                    # the whitespace that precedes a tool call. Flush and go live.
                    if "".join(pending).strip():
                        classified = True
                        # lstrip only this first flush: the model reliably opens
                        # with "\n\n", which would render as blank lines in the UI.
                        yield ("token", "".join(pending).lstrip())
                        pending.clear()
            elif ev.kind == "tool_calls":
                # Turn was a tool call after all — drop the buffered whitespace.
                tool_calls = ev.tool_calls
                pending.clear()
            elif ev.kind == "done":
                finish_reason = ev.finish_reason

        # A turn that produced only whitespace and no tool call would otherwise
        # vanish into the buffer. Nothing useful to show, but the caller must
        # still see an accurate content record to detect the empty-answer case.
        if pending and not tool_calls:
            leftover = "".join(pending).strip()
            if leftover:
                yield ("token", leftover)

        self._last_turn = (tool_calls, finish_reason, "".join(content_parts))

    # ── the loop ───────────────────────────────────────────────────────────

    def run(self, messages: list[dict]) -> Iterator[AgentEvent]:
        """Run the turn to completion, yielding AgentEvents."""
        messages = list(messages)
        previous_queries: set[str] = set()

        for turn in range(self.max_tool_turns + 1):
            is_last = turn == self.max_tool_turns
            # Reasoning is worth paying for only on the opening turn, where the
            # model decides whether to search and how to word the query. From
            # turn 1 on it has retrieved passages in hand and is synthesising,
            # which mostly spends decode on reasoning nobody reads. (Measured:
            # leaving it on for turn 1 cost ~290 reasoning chunks per answer.)
            think = self.think_tools if turn == 0 else self.think_answer
            # Tool turns are bounded; only the answer may run long.
            budget = self.tool_turn_max_tokens if not is_last else self.answer_max_tokens

            for kind, text in self._run_turn(messages, think=think, max_tokens=budget):
                yield AgentEvent(kind=kind, text=text)

            tool_calls, finish_reason, content = self._last_turn

            if not tool_calls:
                # Spent the whole budget reasoning and produced nothing usable.
                # Rather than return an empty turn, search the user's own words:
                # a mediocre query beats no answer, and guarantees progress.
                if finish_reason == "length" and not content.strip() and turn == 0:
                    log.warning(
                        "opening turn hit its token cap while reasoning; "
                        "falling back to a search on the raw question"
                    )
                    fallback_query = _last_user_text(messages)
                    if fallback_query:
                        yield AgentEvent(kind="searching", query=fallback_query)
                        result = self._search(fallback_query)
                        messages.append(
                            {"role": "user", "content": f"Search results:\n\n{result}"}
                        )
                        if self.ledger.cites:
                            yield AgentEvent(kind="citations", cites=list(self.ledger.cites))
                        continue
                yield AgentEvent(kind="done", finish_reason=finish_reason)
                return

            if is_last:
                # Out of budget: the model wanted to search again but we stop
                # here. It has already seen results; make it answer with them.
                log.info("tool-turn budget exhausted; forcing an answer")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Answer now using the search results you already have. "
                            "Do not search again."
                        ),
                    }
                )
                for kind, text in self._run_turn(
                    messages, think=self.think_answer, max_tokens=self.answer_max_tokens
                ):
                    yield AgentEvent(kind=kind, text=text)
                _, fr, _ = self._last_turn
                yield AgentEvent(kind="done", finish_reason=fr)
                return

            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tc in tool_calls:
                yield AgentEvent(kind="searching", query=emitted_query(tc))
                result = self._dispatch(tc)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )

            if self.ledger.cites:
                yield AgentEvent(kind="citations", cites=list(self.ledger.cites))

            # Loop guard: if the model repeats a query it already ran, more
            # turns cannot produce new context.
            queries = {tc.parsed_arguments().get("query", "") for tc in tool_calls}
            if queries and queries <= previous_queries:
                log.info("model repeated an exhausted query; forcing an answer")
                messages.append(
                    {
                        "role": "user",
                        "content": "Answer now using the results above. Do not search again.",
                    }
                )
                for kind, text in self._run_turn(
                    messages, think=self.think_answer, max_tokens=self.answer_max_tokens
                ):
                    yield AgentEvent(kind=kind, text=text)
                _, fr, _ = self._last_turn
                yield AgentEvent(kind="done", finish_reason=fr)
                return
            previous_queries |= queries

    def _dispatch(self, tc: ToolCall) -> str:
        """Execute one tool call, returning the text handed back to the model.

        Every failure path returns a sentence the model can act on rather than
        raising: a dead tool must not kill the turn."""
        if tc.name != SEARCH_KB:
            log.warning("model called unknown tool %r", tc.name)
            return f"Error: no tool named {tc.name!r} exists."

        query = (tc.parsed_arguments().get("query") or "").strip()
        if not query:
            return "Error: search_kb requires a non-empty 'query' argument."
        if _looks_like_placeholder(query):
            log.warning("rejected placeholder-looking query %r", query)
            return (
                "Error: that query looks like an unfilled template. "
                "Send a real search query built from the user's question."
            )
        query = _bound_query(query)

        return self._search(query)

    def _search(self, query: str) -> str:
        """Run one retrieval and render it for the model.

        Never raises: a dead retriever must degrade into a sentence the model
        can act on, not kill the turn."""
        try:
            ctx = self.retrieve(query)
        except Exception:  # noqa: BLE001
            log.exception("search_kb failed for query %r", query)
            return "Error: the knowledge base search failed. Answer without it if you can."
        # Numbering happens here so citation [N] stays stable and unique across
        # every search the model runs this turn.
        return format_hits(ctx, self.ledger)


def emitted_query(tc: ToolCall) -> str:
    """The query a tool call asked for, for UI status text."""
    return (tc.parsed_arguments().get("query") or "").strip()


__all__ = ["AgentRunner", "AgentEvent", "emitted_query"]
