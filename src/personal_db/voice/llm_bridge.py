"""Pipecat LLM service backed by the project's own agent.

This is the piece that makes voice and text *one* assistant rather than two.
Pipecat normally drives its own LLM and its own tool loop; that would mean a
second agent implementation with its own guards, its own conversation state and
no shared history. Instead this service replaces pipecat's LLM stage entirely
and delegates to `AgentRunner` — the same loop, the same `search_kb` tool, the
same loop guards, the same session rows, and crucially the same process-wide NPU
call gate, so a voice turn and a text turn can never hit the model server at
once.

Frame contract (pipecat 1.3.0):
  in   LLMContextFrame     — the conversation so far
  out  LLMFullResponseStartFrame
       LLMThoughtTextFrame — reasoning; routed here so it is never spoken
       TTSSpeakFrame       — the "checking your notes" filler, see below
       LLMTextFrame        — answer tokens, which TTS speaks as they arrive
       LLMFullResponseEndFrame

Latency note. A search-backed answer needs a tool round-trip before the first
answer token, which is a long silence in a voice call. The agent already emits a
`searching` event at exactly that moment, so we speak a short filler then — the
caller hears something while retrieval and the second model turn run.
"""

from __future__ import annotations

import asyncio
import logging

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    LLMThoughtTextFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings

from .. import memory as memory_agent
from .. import sessions as sess_store
from ..agent import AgentRunner
from ..config import config
from ..retrieve import hybrid_retrieve
from ..stores import Stores
from . import events as vevents

log = logging.getLogger(__name__)

# Spoken while the knowledge base is searched. Kept short and non-committal;
# it is interrupted as soon as real answer tokens start flowing.
SEARCH_FILLER = "Let me check your notes."


class PersonalDBLLMService(LLMService):
    """Runs the project's agent inside a pipecat pipeline."""

    def __init__(self, *, stores: Stores, session_id: str, **kwargs):
        # Every LLMSettings field must be initialised or the base class logs an
        # error about NOT_GIVEN fields at pipeline start. None means "this
        # service does not support that knob" — which is accurate here: model,
        # sampling and token limits are decided by the agent and its config,
        # one layer down, not by pipecat.
        kwargs.setdefault(
            "settings",
            LLMSettings(
                model=config.llm_model,
                extra={},
                system_instruction=None,
                temperature=None,
                max_tokens=None,
                top_p=None,
                top_k=None,
                frequency_penalty=None,
                presence_penalty=None,
                seed=None,
                filter_incomplete_user_turns=None,
                user_turn_completion_config=None,
            ),
        )
        super().__init__(**kwargs)
        self._stores = stores
        self._session_id = session_id

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            await self._run_turn(frame)
        else:
            await self.push_frame(frame, direction)

    # ──────────────────────────────────────────────────────────────────────

    def _latest_user_text(self, context) -> str:
        try:
            for m in reversed(context.get_messages()):
                if m.get("role") == "user":
                    content = m.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
                    # Multimodal content arrives as a list of parts.
                    if isinstance(content, list):
                        text = " ".join(
                            p.get("text", "") for p in content if isinstance(p, dict)
                        ).strip()
                        if text:
                            return text
        except Exception:  # noqa: BLE001
            log.exception("could not read user text from pipecat context")
        return ""

    async def _run_turn(self, frame: LLMContextFrame) -> None:
        user_text = self._latest_user_text(frame.context)
        if not user_text:
            log.debug("voice turn with no user text; ignoring")
            return

        # The transcript is the one thing the caller cannot verify by ear, so
        # show what was actually heard before anything is done with it.
        vevents.publish(self._session_id, "thinking", heard=user_text)
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            await self._generate(user_text)
        except Exception:  # noqa: BLE001 — a failed turn must not kill the call
            log.exception("voice turn failed")
            vevents.publish(self._session_id, "error", detail="that turn failed")
            await self.push_frame(
                LLMTextFrame("Sorry, something went wrong on my side.")
            )
        finally:
            await self.push_frame(LLMFullResponseEndFrame())

    async def _generate(self, user_text: str) -> None:
        # Same durability order as the text path: persist the question first.
        await asyncio.to_thread(
            sess_store.add_message, self._session_id, "user", user_text
        )
        await self._maybe_rename_session(user_text)

        compacted = await asyncio.to_thread(
            memory_agent.compact_if_needed, self._session_id
        )

        # Imported here to avoid a circular import at module load: chat imports
        # the agent, and the voice package is imported from the web app.
        from ..chat import build_voice_messages

        messages = build_voice_messages(
            compacted.summary, compacted.recent, user_text
        )

        runner = AgentRunner(
            base_url=config.llm_host,
            model=config.llm_model,
            retrieve=lambda q: hybrid_retrieve(q, self._stores),
            max_tool_turns=config.max_tool_turns,
            # Reasoning is the single largest latency cost and helps least when
            # the answer has to be short and spoken. Off for both turns here.
            think_tools=config.voice_think,
            think_answer=config.voice_think,
            fallback_base_url=config.fallback_llm_host,
            fallback_model=config.fallback_llm_model,
            tool_turn_max_tokens=config.tool_turn_max_tokens,
            answer_max_tokens=config.voice_answer_max_tokens or None,
        )

        events = runner.run(messages)
        spoken: list[str] = []
        cites: list[dict] = []

        def _next():
            try:
                return next(events)
            except StopIteration:
                return None

        try:
            while True:
                ev = await asyncio.to_thread(_next)
                if ev is None:
                    break
                if ev.kind == "token":
                    if not spoken:
                        vevents.publish(self._session_id, "speaking")
                    spoken.append(ev.text)
                    await self.push_frame(LLMTextFrame(ev.text))
                elif ev.kind == "reasoning":
                    # Routed as a thought so it is never sent to TTS.
                    await self.push_frame(LLMThoughtTextFrame(ev.text))
                elif ev.kind == "searching":
                    vevents.publish(self._session_id, "searching", query=ev.query)
                    await self.push_frame(TTSSpeakFrame(SEARCH_FILLER))
                elif ev.kind == "citations":
                    cites = ev.cites
                    # Spoken answers cannot carry citation markers, so the
                    # sources go to the screen instead.
                    vevents.publish(
                        self._session_id,
                        "searching",
                        sources=[
                            {"name": c.get("name"), "path": c.get("path")}
                            for c in (cites or [])[:6]
                        ],
                    )
        finally:
            try:
                await asyncio.to_thread(events.close)
            except Exception:  # noqa: BLE001
                log.debug("closing voice agent stream failed", exc_info=True)

        text = "".join(spoken).strip()
        if text:
            import json

            await asyncio.to_thread(
                sess_store.add_message,
                self._session_id,
                "assistant",
                text,
                json.dumps(cites),
            )

    async def _maybe_rename_session(self, user_text: str) -> None:
        s = await asyncio.to_thread(sess_store.get_session, self._session_id)
        if s and s.title == "New chat":
            await asyncio.to_thread(
                sess_store.rename_session, self._session_id, user_text[:60]
            )
