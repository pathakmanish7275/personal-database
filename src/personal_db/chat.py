"""Chat generation: retrieve from Qdrant, build a prompt, stream from Ollama.

The actual streaming runs as a background asyncio task via personal_db.jobs.
This module exports `run_generation(job, stores)` which the jobs runner calls.
HTTP clients subscribe to the job's events for SSE delivery."""

from __future__ import annotations

import asyncio
import json
import logging

from llama_index.core.llms import ChatMessage, MessageRole

from . import jobs as jobs_mod
from . import memory as memory_agent
from . import sessions as sess_store
from .agent import AgentRunner
from .config import config
from .retrieve import RetrievedChunk, hybrid_retrieve
from .stores import Stores, configure_llama_index

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Alex's local personal-database assistant.

You have one tool, search_kb, which searches Alex's own documents — his
notes, journals, PDFs and books.

When to search:
- Search whenever the question concerns Alex's own material: his notes,
  projects, decisions, history, or anything he has written down.
- Do NOT search for greetings, small talk, or questions you can answer from
  general knowledge alone.
- Questions about Alex himself — who he is, what he does, what he is working
  on — are knowledge-base questions. Search for them. You know nothing about
  him beyond what the documents say; his name appearing above is not knowledge
  about him.
- Do NOT search to answer a question about the conversation you are already
  having.
- If the question is too vague to search usefully, ask Alex a short
  clarifying question instead of guessing a query.

Search results arrive as passages labeled  [N] <filename>\\n<text>, sometimes
followed by relations from the knowledge graph written as
entity ─predicate→ entity. Retrieval errs on the side of recall, so some
passages will be tangential: read each one, then SILENTLY DISCARD any that does
not directly help, even if it shares keywords. Never mix facts from different
documents unless asked to compare. The graph relations come from a non-LLM
extractor and can be noisy — use them to fill in links the passages don't
state, but don't quote them and don't cite them.

Citation rules:
- Cite passages inline as [1], [2], using the numbers they were given.
- Cite only passages you actually used; never cite one you discarded.
- Never cite the relations block.

If a search returns nothing useful, say so plainly ("I don't see that in your
notes") rather than guessing. You may search once more with different terms if
the first query was poorly chosen, but do not repeat a search that already
failed.

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


def _build_agent_messages(
    summary: str,
    recent: list[sess_store.Message],
    user_text: str,
) -> list[dict]:
    """Build the OpenAI-format message list for the agent loop.

    No context block: retrieval is a tool the model calls, not something we
    pre-stuff into the prompt. Plain dicts rather than ChatMessage because the
    agent talks to /v1/chat/completions directly — the LlamaIndex Ollama client
    targets /api/chat, which silently drops the tools array."""
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if summary.strip():
        msgs.append(
            {
                "role": "system",
                "content": (
                    "Memory of earlier turns in this session (compressed):\n\n"
                    f"{summary.strip()}"
                ),
            }
        )
    for m in recent:
        msgs.append(
            {"role": "user" if m.role == "user" else "assistant", "content": m.content}
        )
    msgs.append({"role": "user", "content": user_text})
    return msgs


VOICE_SYSTEM_PROMPT = """You are Alex's local personal-database assistant, speaking aloud.

You have one tool, search_kb, which searches Alex's own documents — his notes,
journals, PDFs and books. Search whenever the question concerns his own
material — including questions about Alex himself, such as who he is or what
he works on, which you can only answer from his documents. Do not search for
greetings, small talk, or general knowledge.

You are being heard, not read. This changes how you answer:
- Two or three sentences. Never more unless asked to go on.
- Plain spoken English. No markdown, no bullet points, no headings, no
  asterisks — they get read out as noise.
- Never speak citation markers like [1]. Name the document instead, naturally:
  "your architecture notes say…".
- Spell out anything that would be unclear when heard: say "twelve percent",
  not "12%".
- If a search finds nothing useful, say so in one sentence.
- If the question is too vague to search, ask one short clarifying question.

Do not narrate what you are about to do. Answer."""


def build_voice_messages(
    summary: str,
    recent: list[sess_store.Message],
    user_text: str,
) -> list[dict]:
    """Agent messages for a spoken turn.

    Identical to the text path except for the system prompt, so voice and text
    share one conversation history — a voice turn is visible in the chat
    transcript and can be followed up in text, and vice versa."""
    msgs = _build_agent_messages(summary, recent, user_text)
    msgs[0] = {"role": "system", "content": VOICE_SYSTEM_PROMPT}
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

    # 3) Agentic turn. The model is handed `search_kb` as a real tool and
    #    decides for itself whether to search, how to phrase the query, whether
    #    to search again, or to answer directly. Asking a clarifying question
    #    needs no special case — it is just the model answering without
    #    calling the tool.
    await jobs_mod.emit(
        job, {"type": "status", "data": {"phase": "thinking", "text": "thinking"}}
    )

    messages = _build_agent_messages(compacted.summary, compacted.recent, job.user_text)

    def _retrieve(query: str):
        return hybrid_retrieve(query, stores)

    runner = AgentRunner(
        base_url=config.llm_host,
        model=config.llm_model,
        retrieve=_retrieve,
        max_tool_turns=config.max_tool_turns,
        think_tools=config.think_tool_turns,
        think_answer=config.think_answer,
        fallback_base_url=config.fallback_llm_host,
        fallback_model=config.fallback_llm_model,
        tool_turn_max_tokens=config.tool_turn_max_tokens,
        answer_max_tokens=config.answer_max_tokens or None,
    )

    # From here on a stream to the model server may be open. Cancel must be
    # cooperative: hard-cancelling would drop the HTTP stream mid-generation.
    job.streaming = True
    events = runner.run(messages)
    full_text_parts: list[str] = []
    cites: list[dict] = []
    cancelled = False

    def _next_event():
        try:
            return next(events)
        except StopIteration:
            return None

    try:
        while True:
            ev = await asyncio.to_thread(_next_event)
            if ev is None:
                break
            if job.cancel_requested and not cancelled:
                # Stop emitting/persisting but keep draining, so the model
                # server finishes its generation cleanly.
                cancelled = True
                full_text_parts.clear()
                log.info("job %s cancelled mid-turn; draining", job.session_id)
            if cancelled:
                continue

            if ev.kind == "token":
                full_text_parts.append(ev.text)
                await jobs_mod.emit(job, {"type": "token", "data": ev.text})
            elif ev.kind == "reasoning":
                # Reasoning arrives on its own channel and is never part of the
                # answer; surface it as status only so the UI can show life.
                await jobs_mod.emit(
                    job, {"type": "reasoning", "data": ev.text}
                )
            elif ev.kind == "searching":
                await jobs_mod.emit(
                    job,
                    {
                        "type": "status",
                        "data": {"phase": "retrieving", "text": f"searching: {ev.query[:80]}"},
                    },
                )
            elif ev.kind == "citations":
                cites = ev.cites
                await jobs_mod.emit(job, {"type": "citations", "data": cites})
            elif ev.kind == "done" and ev.finish_reason == "length":
                log.warning(
                    "job %s hit the model's output cap", job.session_id
                )
    finally:
        # The agent holds the process-wide LLM gate for the life of each
        # request. If we leave by any path other than exhausting the generator,
        # close it explicitly — leaving it to GC would gate every future LLM
        # call in the process on refcount timing.
        job.streaming = False
        try:
            await asyncio.to_thread(events.close)
        except Exception:  # noqa: BLE001
            # Already-exhausted is a no-op; "generator already executing" means
            # a worker thread still holds it and will release the gate itself.
            # Never let cleanup mask the exception that brought us here.
            log.debug("closing agent stream for job %s failed", job.session_id, exc_info=True)

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
