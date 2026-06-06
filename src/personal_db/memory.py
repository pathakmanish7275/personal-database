"""Memory agent chain: maintain a compact running summary per chat session.

Pipeline, run before every chat turn:

  1. estimate  — token-count the un-summarized history
  2. decide    — is the un-summarized budget exceeded?
  3. select    — which *turns* fold into the summary vs. stay verbatim?
  4. compress  — call gpt-oss:20b to refresh the summary
  5. apply     — persist the new summary + marker

Compaction is turn-aware: a "turn" is a user message plus the assistant
reply that follows it. We never split a pair (folding a lone question with no
answer produced near-empty summaries). The token budget is the *trigger*; the
recent keep-window is a small coherence floor of whole turns kept verbatim.
Everything older than that window folds into the summary in one pass.

Result returned to the chat layer:

  (summary_text, recent_messages_to_pass_verbatim)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from llama_index.core.llms import ChatMessage, MessageRole

from . import sessions as sess_store
from .config import config
from .llm_runtime import safe_chat

log = logging.getLogger(__name__)

SUMMARIZER_SYSTEM = """You maintain a compact running memory of a conversation between Alex (the user) and his personal-database assistant.

Keep the summary tight ({max_chars} characters max). It must preserve:
- decisions Alex made (with rationale, briefly)
- preferences and constraints he stated (style, tools, "don't do X")
- key facts about him, his projects, people, and any named entities raised
- unresolved questions or threads he intends to come back to
- corrections he gave the assistant

Drop:
- conversational filler and pleasantries
- detailed reasoning the assistant produced (keep only the conclusion)
- retrieved document content (the RAG layer handles that, don't restate it)
- topics fully resolved with nothing actionable

Write in third person, as a memo to a future instance of the assistant. Use short
sentences or bullets. No preamble, no closing remark, summary text only."""

SUMMARIZER_USER_TEMPLATE = """Existing summary:
{existing_summary}

New turns to fold in:
{new_turns}

Produce the refreshed summary now."""


@dataclass
class CompactedHistory:
    summary: str
    recent: list[sess_store.Message]
    did_compact: bool = False
    compressed_count: int = 0


def _estimate_tokens(text: str) -> int:
    """Cheap token estimate. Tiktoken would be more accurate but adds latency;
    a char/4 heuristic is good enough for thresholding."""
    return max(1, len(text) // 4)


def _messages_tokens(msgs: list[sess_store.Message]) -> int:
    return sum(_estimate_tokens(m.content) + 4 for m in msgs)


def _group_turns(msgs: list[sess_store.Message]) -> list[list[sess_store.Message]]:
    """Group a flat message list into turns. A new turn starts at each user
    message; the assistant reply (and any follow-on messages) attach to it. The
    final turn may be a lone user message (the question currently being answered)."""
    turns: list[list[sess_store.Message]] = []
    current: list[sess_store.Message] = []
    for m in msgs:
        if m.role == "user" and current:
            turns.append(current)
            current = []
        current.append(m)
    if current:
        turns.append(current)
    return turns


def _format_turns_for_summary(msgs: list[sess_store.Message]) -> str:
    lines = []
    for m in msgs:
        role = m.role.upper()
        lines.append(f"[{role}] {m.content.strip()}")
    return "\n\n".join(lines)


def _run_summarizer(existing: str, to_compress: list[sess_store.Message]) -> str:
    """Call gpt-oss:20b synchronously to produce the new summary."""
    if not to_compress:
        return existing
    existing_block = existing.strip() if existing.strip() else "(none yet)"
    user = SUMMARIZER_USER_TEMPLATE.format(
        existing_summary=existing_block,
        new_turns=_format_turns_for_summary(to_compress),
    )
    system = SUMMARIZER_SYSTEM.format(max_chars=config.memory_summary_max_chars)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system),
        ChatMessage(role=MessageRole.USER, content=user),
    ]
    resp = safe_chat(messages)
    new_summary = (resp.message.content or "").strip()
    if len(new_summary) > config.memory_summary_max_chars * 2:
        new_summary = new_summary[: config.memory_summary_max_chars * 2]
    return new_summary


def _plan(session_id: str):
    """Compute the work to do (if any) WITHOUT running the LLM.

    Returns (session, summary_text, unsummarized_msgs, to_compress, to_keep,
    compress_turn_count). When no compaction is needed, to_compress is empty,
    to_keep is the full un-summarized history, and the turn count is 0."""
    session = sess_store.get_session(session_id)
    if session is None:
        return None, "", [], [], [], 0

    unsummarized = sess_store.get_messages(
        session_id, since_id=session.summary_up_to_msg_id
    )
    turns = _group_turns(unsummarized)
    keep_n = max(1, config.memory_keep_recent_turns)
    over_budget = _messages_tokens(unsummarized) > config.memory_token_budget

    # Only fold when we're over budget AND there are whole turns older than the
    # recent keep-window. The budget is the trigger; keep_n is the coherence floor.
    if not over_budget or len(turns) <= keep_n:
        return session, session.summary, unsummarized, [], unsummarized, 0

    compress_turns = turns[:-keep_n]
    keep_turns = turns[-keep_n:]
    to_compress = [m for t in compress_turns for m in t]
    to_keep = [m for t in keep_turns for m in t]
    return session, session.summary, unsummarized, to_compress, to_keep, len(compress_turns)


def peek_compaction(session_id: str) -> int:
    """How many older *turns* WILL be folded into the summary on the next call?

    Returns 0 if no compaction is needed. Cheap — does not call the LLM."""
    *_, compress_turn_count = _plan(session_id)
    return compress_turn_count


def compact_if_needed(session_id: str) -> CompactedHistory:
    """Run the memory chain when the un-summarized history exceeds the budget."""
    session, summary, unsummarized, to_compress, to_keep, compress_turns = _plan(session_id)
    if session is None:
        return CompactedHistory(summary="", recent=[], did_compact=False)
    if not to_compress:
        return CompactedHistory(summary=summary, recent=to_keep, did_compact=False)

    marker_id = to_compress[-1].id or 0
    log.info(
        "compacting session=%s: compressing %d turns / %d msgs (~%d tokens), keeping %d msgs verbatim",
        session_id, compress_turns, len(to_compress), _messages_tokens(to_compress), len(to_keep),
    )
    new_summary = _run_summarizer(summary, to_compress)
    sess_store.update_summary(session_id, new_summary, marker_id)
    return CompactedHistory(
        summary=new_summary,
        recent=to_keep,
        did_compact=True,
        compressed_count=compress_turns,
    )
