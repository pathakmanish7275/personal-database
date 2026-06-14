"""Chat generation: retrieve from Qdrant, build a prompt, stream from Ollama.

The actual streaming runs as a background asyncio task via personal_db.jobs.
This module exports `run_generation(job, stores)` which the jobs runner calls.
HTTP clients subscribe to the job's events for SSE delivery."""

from __future__ import annotations

import asyncio
import json

from llama_index.core.llms import ChatMessage, MessageRole

from . import jobs as jobs_mod
from . import memory as memory_agent
from . import sessions as sess_store
from . import planner as planner_mod
from .llm_runtime import safe_stream_chat
from .retrieve import RetrievedChunk, hybrid_retrieve
from .stores import Stores, configure_llama_index

SYSTEM_PROMPT = """You are Alex's local personal-database assistant.

Two sources of context arrive with every question:

1. Retrieved chunks — labeled  [N] <filename>\\n<text>.  The filename matters:
   it tells you which of Alex's documents the chunk came from. Retrieval
   errs on the side of recall, so some chunks may be tangential. Read each
   one, then SILENTLY DISCARD any chunk whose content does not directly help
   answer the question, even if it shares keywords. Never mix facts from
   different documents unless explicitly asked to compare.

2. Known relations from the graph — a short list of structured facts written as
   entity ─predicate→ entity.  These come from connections recorded across
   Alex's notes by a non-LLM extractor, so some can be noisy. Use them to
   fill in links the chunks alone don't state, but don't quote them verbatim
   and don't cite them.

Citation rules:
- Cite chunks inline as [1], [2] matching only the chunks you actually used.
- Do NOT cite a chunk you discarded.
- Do NOT cite the relations block.

If, after discarding tangential chunks, no useful context remains, say so
plainly ("I don't see that in your notes") rather than guessing.

For general-knowledge questions answer normally and prefix the answer with
[general knowledge].

Style: concise, direct, no preamble like "Sure!" or "Here's the answer:".
Do not use markdown formatting — no asterisks, no bold, no italics, no bullet
symbols, no headers. Write in plain prose."""


# ─── prompt formatting (kept pure for unit tests) ──────────────────────────


def _format_context(chunks: list[RetrievedChunk]) -> tuple[str, list[dict]]:
    """Turn the fused chunk list into the [N] context block + a citations list."""
    blocks = []
    cites = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[{i}] {c.name}\n{c.text.strip()}")
        cites.append(
            {
                "n": i,
                "name": c.name,
                "path": c.path,
                "doc_id": c.doc_id,
                "score": c.vector_score,
                "origin": c.origin,
                "graph_hits": c.graph_hits or None,
            }
        )
    return "\n\n".join(blocks), cites


def _format_relations(rels: list[tuple[str, str, str]]) -> str:
    """Render the graph triples as compact, non-cited bullet lines."""
    return "\n".join(f"- {h} ─{p}→ {t}" for h, p, t in rels)


def _previous_clarification(recent: list[sess_store.Message]) -> str | None:
    """Return the text of the most recent assistant clarification, if any.

    Clarifications are saved with citations=NULL while real answers always carry
    a JSON citations payload (even when empty), so the column doubles as a
    marker — no migration required."""
    for m in reversed(recent):
        if m.role != "assistant":
            continue
        if m.citations is None:
            return m.content or ""
        return None  # last assistant turn was a real answer; no clarification active
    return None


def _build_messages(
    summary: str,
    recent: list[sess_store.Message],
    user_text: str,
    context_block: str,
    relations_block: str = "",
) -> list[ChatMessage]:
    msgs: list[ChatMessage] = [ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT)]
    if summary.strip():
        msgs.append(
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "Memory of earlier turns in this session (compressed):\n\n"
                    f"{summary.strip()}"
                ),
            )
        )
    for m in recent:
        role = MessageRole.USER if m.role == "user" else MessageRole.ASSISTANT
        msgs.append(ChatMessage(role=role, content=m.content))

    parts = []
    if context_block:
        parts.append(f"Context from my personal database:\n\n{context_block}")
    if relations_block:
        parts.append(f"Known relations from my graph:\n\n{relations_block}")
    body = "\n\n".join(parts) if parts else "(no context retrieved)"
    msgs.append(
        ChatMessage(
            role=MessageRole.USER,
            content=f"{body}\n\n---\n\nQuestion: {user_text}",
        )
    )
    return msgs


# ─── the async runner used by jobs.start ──────────────────────────────────


async def run_generation(job: jobs_mod.GenJob, stores: Stores) -> None:
    """Do the full chat turn: compaction → retrieval → LLM stream → save."""
    configure_llama_index()

    # 1) Save the user message + rename session BEFORE any heavy work, so the
    #    user's question is always durable and the title sticks.
    await asyncio.to_thread(sess_store.add_message, job.session_id, "user", job.user_text)
    s = await asyncio.to_thread(sess_store.get_session, job.session_id)
    if s and s.title == "New chat":
        await asyncio.to_thread(sess_store.rename_session, job.session_id, job.user_text[:60])

    # 2) Memory compaction (gpt-oss:20b — slow). Emit a status BEFORE running so
    #    the user sees the "summarizing N earlier turns" pill, not silence.
    pending = await asyncio.to_thread(memory_agent.peek_compaction, job.session_id)
    if pending:
        await jobs_mod.emit(
            job,
            {"type": "status", "data": {"phase": "summarizing", "text": f"summarizing {pending} earlier turns"}},
        )
    compacted = await asyncio.to_thread(memory_agent.compact_if_needed, job.session_id)
    if compacted.did_compact:
        await jobs_mod.emit(
            job,
            {
                "type": "memory",
                "data": {
                    "compacted": True,
                    "compressed_count": compacted.compressed_count,
                    "summary_chars": len(compacted.summary),
                },
            },
        )

    # 3) Plan — figure out if the question is searchable, or if we should ask
    #    a clarifying question first. On any planner failure we fall through to
    #    a search on the raw question.
    await jobs_mod.emit(
        job, {"type": "status", "data": {"phase": "planning", "text": "understanding your question…"}}
    )
    # If the prior assistant turn was a clarification (citations stored as NULL),
    # this user message is the reply — tell the planner so it doesn't re-ask.
    prior_clarify = _previous_clarification(compacted.recent)
    plan = await asyncio.to_thread(
        planner_mod.plan, job.user_text, compacted.recent, prior_clarify
    )
    if plan.action == "clarify":
        # Skip retrieval + synthesis; the clarification IS this turn's response.
        await jobs_mod.emit(
            job,
            {"type": "clarify", "data": {"reason": plan.reason}},
        )
        await jobs_mod.emit(job, {"type": "token", "data": plan.clarify or ""})
        await asyncio.to_thread(
            sess_store.add_message, job.session_id, "assistant", plan.clarify or "", None
        )
        return

    retrieval_query = plan.query or job.user_text
    if retrieval_query != job.user_text:
        await jobs_mod.emit(
            job,
            {"type": "status", "data": {"phase": "planning", "text": f"refined: {retrieval_query[:80]}"}},
        )

    # 4) Hybrid retrieval — vector + graph fire together inside the worker thread.
    await jobs_mod.emit(
        job, {"type": "status", "data": {"phase": "retrieving", "text": "searching docs + graph"}}
    )
    ctx = await asyncio.to_thread(hybrid_retrieve, retrieval_query, stores)
    context_block, cites = _format_context(ctx.chunks)
    relations_block = _format_relations(ctx.relations)
    await jobs_mod.emit(job, {"type": "citations", "data": cites})
    # Always emit a dedicated `graph` event when the KG is enabled — even when
    # the graph didn't contribute — so the UI can show that it was consulted.
    from .config import config as _cfg
    if _cfg.kg_enabled and stores.graph is not None:
        await jobs_mod.emit(
            job,
            {
                "type": "graph",
                "data": {
                    "entities": ctx.query_entities,
                    "chunks": ctx.graph_chunk_count,
                    "relations": len(ctx.relations),
                    "semantic_chunks": ctx.semantic_chunk_count,
                },
            },
        )

    # 4) Build the chat messages and stream the LLM response.
    messages = _build_messages(
        compacted.summary, compacted.recent, job.user_text, context_block, relations_block
    )
    await jobs_mod.emit(
        job, {"type": "status", "data": {"phase": "thinking", "text": "thinking"}}
    )

    response_stream = await asyncio.to_thread(safe_stream_chat, messages)
    full_text_parts: list[str] = []
    iterator = iter(response_stream)

    def _next_chunk():
        try:
            return next(iterator)
        except StopIteration:
            return None

    while True:
        chunk = await asyncio.to_thread(_next_chunk)
        if chunk is None:
            break
        delta = chunk.delta or ""
        if delta:
            full_text_parts.append(delta)
            await jobs_mod.emit(job, {"type": "token", "data": delta})

    # 5) Persist the assistant message (citations included).
    full_text = "".join(full_text_parts)
    if full_text:
        await asyncio.to_thread(
            sess_store.add_message,
            job.session_id,
            "assistant",
            full_text,
            json.dumps(cites),
        )
