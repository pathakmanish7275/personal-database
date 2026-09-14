"""Per-session voice state, published to the browser over SSE.

The page can already poll for finished messages, but a call is mostly *waiting*
— and silence with no explanation reads as a hang. These events let the UI say
which kind of waiting it is: listening, hearing you, searching your notes,
thinking, speaking.

Deliberately fire-and-forget. A dropped state event must never block or fail the
call, so publishing to a full queue discards the oldest rather than waiting, and
a subscriber that goes away is simply forgotten.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

log = logging.getLogger(__name__)

# session_id -> set of subscriber queues
_subscribers: dict[str, set[asyncio.Queue]] = {}

# Last state per session, replayed to a new subscriber so a page that connects
# mid-call shows the current state instead of nothing.
_last: dict[str, dict] = {}

_MAX_QUEUE = 32


def publish(session_id: str, state: str, **data) -> None:
    """Publish a state change. Safe to call from anywhere, never raises."""
    event = {"state": state, **data}
    _last[session_id] = event
    for q in list(_subscribers.get(session_id, ())):
        try:
            if q.full():
                # Drop the oldest: a slow page must not stall the call.
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(event)
        except Exception:  # noqa: BLE001
            log.debug("dropping voice event for a dead subscriber", exc_info=True)


async def subscribe(session_id: str) -> AsyncIterator[dict]:
    """Yield state events for a session until the caller stops iterating."""
    q: asyncio.Queue = asyncio.Queue(maxsize=_MAX_QUEUE)
    _subscribers.setdefault(session_id, set()).add(q)
    try:
        last = _last.get(session_id)
        if last:
            yield last
        while True:
            yield await q.get()
    finally:
        subs = _subscribers.get(session_id)
        if subs:
            subs.discard(q)
            if not subs:
                _subscribers.pop(session_id, None)


def clear(session_id: str) -> None:
    """Forget the remembered state when a call ends."""
    _last.pop(session_id, None)


def make_state_observer(session_id: str):
    """A pipeline processor that turns pipecat frames into UI state events.

    Sits in the pipeline purely to watch traffic — every frame is passed through
    untouched. Kept separate from the LLM bridge because these states come from
    the transport and TTS stages rather than from the agent.

    Imported lazily so this module stays importable without pipecat."""
    from pipecat.frames.frames import (
        BotStoppedSpeakingFrame,
        Frame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    class _VoiceStateObserver(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, UserStartedSpeakingFrame):
                publish(session_id, "hearing")
            elif isinstance(frame, UserStoppedSpeakingFrame):
                publish(session_id, "transcribing")
            elif isinstance(frame, BotStoppedSpeakingFrame):
                # Back to waiting for the caller once the reply finishes.
                publish(session_id, "listening")
            await self.push_frame(frame, direction)

    return _VoiceStateObserver()
