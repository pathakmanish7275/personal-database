"""Pre-retrieval planner — decides clarify-or-search before any chunks are pulled.

Every turn, before the vector + graph retrieval fires, an LLM call inspects the
user's question (with the recent conversation as context) and emits one of:

  search   — the question is specific enough to retrieve on. The planner
             rewrites it into a short, keyword-rich query stripped of pronouns
             and filler — that becomes the retrieval input.
  clarify  — the question is genuinely ambiguous. The planner writes ONE short
             follow-up question, which becomes the assistant's response for the
             turn (no retrieval runs, no synthesis happens).

Defaults to `search` on any planner failure (bad JSON, network error, etc.) so
the chat is never silently blocked.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

from llama_index.core.llms import ChatMessage, MessageRole

from . import sessions as sess_store
from .llm_runtime import safe_chat

log = logging.getLogger(__name__)

PLANNER_SYSTEM = """You plan the first step of a personal-knowledge-base agent that retrieves from Alex's notes and a knowledge graph.

For the next user message, decide ONE of:
- "search":  the question is specific enough to retrieve on. Restate it as a precise, keyword-rich query — short, no pronouns, no filler.
- "clarify": the question is genuinely ambiguous. Write ONE short follow-up question that would make it answerable.

Strongly prefer "search". Only ask for clarification when:
- a pronoun (it/they/this/that) lacks an antecedent in recent turns,
- intent is vague ("what should I do?") with no recoverable topic in recent turns,
- two reasonable interpretations exist AND the pick changes what to retrieve.

Hard rules:
- NEVER clarify twice in a row. If the "previously asked" line below is present,
  YOU already asked that question on the prior turn — the user's current
  message is their REPLY. Treat the reply as the answer and output action="search".
  Use the user's reply (possibly combined with the prior question's topic) as
  the refined query.
- Questions ABOUT the conversation itself ("what have we covered?",
  "summarize our chat", "what did you just say?") are searchable — output
  action="search" with a query like "conversation summary" or similar.

Respond with STRICT JSON only, no prose, no code fences:
{
  "action":  "search" | "clarify",
  "query":   <string or null>,
  "clarify": <string or null>,
  "reason":  <one sentence>
}"""

USER_TEMPLATE = """Recent conversation (most recent last):
{history}
{previously_block}
Current user message:
{question}

Output the JSON object only."""


@dataclass
class Plan:
    action: Literal["search", "clarify"]
    query: str | None = None
    clarify: str | None = None
    reason: str = ""


def _format_history(recent: list[sess_store.Message]) -> str:
    if not recent:
        return "(no prior turns)"
    lines = []
    # Last 8 messages (4 turns) is plenty of context for disambiguation.
    for m in recent[-8:]:
        snippet = (m.content or "").strip().replace("\n", " ")[:300]
        lines.append(f"[{m.role.upper()}] {snippet}")
    return "\n".join(lines)


def _parse_json(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    # Strip code fences a model might add despite instructions.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                return None
    return None


def plan(
    user_text: str,
    recent: list[sess_store.Message],
    previously_clarified: str | None = None,
) -> Plan:
    """Decide search-vs-clarify. Falls through to search-with-raw-query on any error.

    `previously_clarified`: when set, the planner's own last turn was a
    clarification. Pass it in so the model knows the user's current message is
    a *reply* — and a hard guard below forces search even if the LLM ignores
    the rule (Flash-Lite sometimes does)."""
    previously_block = (
        f'\nYou previously asked the user: "{previously_clarified.strip()}"\n'
        if previously_clarified
        else ""
    )
    msgs = [
        ChatMessage(role=MessageRole.SYSTEM, content=PLANNER_SYSTEM),
        ChatMessage(
            role=MessageRole.USER,
            content=USER_TEMPLATE.format(
                history=_format_history(recent),
                previously_block=previously_block,
                question=user_text,
            ),
        ),
    ]
    try:
        resp = safe_chat(msgs)
        raw = (resp.message.content or "").strip()
    except Exception:  # noqa: BLE001
        log.exception("planner call failed; defaulting to search")
        return Plan(action="search", query=user_text, reason="planner error — using raw query")

    parsed = _parse_json(raw)
    if not parsed:
        return Plan(action="search", query=user_text, reason="non-JSON planner output — using raw query")

    action = (parsed.get("action") or "search").strip().lower()
    reason = (parsed.get("reason") or "").strip()

    if action == "clarify":
        # Hard anti-loop: if we just clarified on the previous turn, the user's
        # current message is the answer. Don't ask again.
        if previously_clarified:
            return Plan(
                action="search",
                query=user_text,
                reason="anti-loop override — already asked once",
            )
        q = (parsed.get("clarify") or "").strip()
        if not q:
            return Plan(action="search", query=user_text, reason="empty clarify — using raw query")
        return Plan(action="clarify", clarify=q, reason=reason)

    refined = (parsed.get("query") or "").strip() or user_text
    return Plan(action="search", query=refined, reason=reason)
