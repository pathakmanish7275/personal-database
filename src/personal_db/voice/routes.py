"""Voice routes, mounted into the main app so there is one service and one UI.

`POST /voice/offer` is WebRTC signalling; the browser's call button talks to
the same origin as the chat page. The session id travels with the offer, so a
call continues whichever conversation is open on screen.
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ..config import config
from . import events as vevents

log = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])

_handler = None
# Keyed by session, not a bare set: a page reload or a second Call press sends
# a fresh offer, and without tearing the previous bot down both pipelines stay
# subscribed to the same audio. Observed live — two aggregators, every utterance
# transcribed twice, and two agent turns racing for the NPU.
_bots: dict[str, asyncio.Task] = {}


async def _retire_existing_bot(session_id: str) -> None:
    task = _bots.pop(session_id, None)
    if task is None or task.done():
        return
    log.info("replacing the running voice bot for session %s", session_id)
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass


def _get_handler():
    """Built lazily: importing pipecat's transport is slow and only needed if
    someone actually places a call."""
    global _handler
    if _handler is None:
        from pipecat.transports.smallwebrtc.request_handler import (
            SmallWebRTCRequestHandler,
        )

        _handler = SmallWebRTCRequestHandler()
    return _handler


@router.get("/status")
async def voice_status():
    """Whether a call can currently be placed, and why not if it can't.

    The browser uses this to enable or explain the call button, rather than
    letting the user click and hit a dead connection."""

    async def _serving(base: str, route: str) -> bool:
        """True only if the *right* service is on that port.

        A liveness check is not enough: port 8000 on this machine hosts an
        unrelated API whose /health also returns 200, which made the call
        button claim to be ready when no speech server existed. Probing the
        audio route itself discriminates — an existing POST-only route answers
        405 to a GET, while a different service answers 404."""
        url = base.rstrip("/") + route
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(url)
                return r.status_code != 404
        except httpx.HTTPError:
            return False

    stt_ok, tts_ok = await asyncio.gather(
        _serving(config.stt_base_url, "/audio/transcriptions"),
        _serving(config.tts_base_url, "/audio/speech"),
    )
    ready = bool(config.voice_enabled and stt_ok and tts_ok)
    reasons = []
    if not config.voice_enabled:
        reasons.append("voice disabled (VOICE_ENABLED=false)")
    if not stt_ok:
        reasons.append(f"no speech-to-text service at {config.stt_base_url}")
    if not tts_ok:
        reasons.append(f"no text-to-speech service at {config.tts_base_url}")
    return JSONResponse({"ready": ready, "reasons": reasons})


@router.get("/{session_id}/events")
async def voice_events(session_id: str):
    """Stream what the voice agent is doing, for the call bar in the UI.

    A call is mostly waiting, and unexplained silence reads as a hang — this
    says which kind of waiting it is."""
    from sse_starlette.sse import EventSourceResponse

    async def gen():
        async for ev in vevents.subscribe(session_id):
            yield {"event": "state", "data": json.dumps(ev)}

    return EventSourceResponse(gen())


@router.post("/{session_id}/say")
async def voice_say(session_id: str, request: Request):
    """Send typed text into a call that is already running.

    Speech is the lossy channel: names and spellings arrive mangled ("MuseTalk"
    came through as "new stock" and "MUSP C-A-L-T"), and repeating them louder
    does not help. Typing gives the caller an exact-input path without dropping
    the call, and the reply still comes back as speech.
    """
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")

    from .bot import say_text

    if not await say_text(session_id, text):
        raise HTTPException(status_code=409, detail="no call is running")
    return {"ok": True}


@router.post("/offer")
async def voice_offer(request: Request):
    """WebRTC offer/answer. Body carries the offer plus `session_id`."""
    if not config.voice_enabled:
        raise HTTPException(503, "voice is disabled")

    body = await request.json()
    session_id = (body.get("session_id") or "").strip()
    if not session_id:
        raise HTTPException(400, "session_id is required")

    from pipecat.transports.smallwebrtc.request_handler import SmallWebRTCRequest

    # Imported here, not at module scope: pulls in torch/silero and would make
    # every app start pay for it even when voice is never used.
    from ..web.app import get_stores
    from .bot import run_voice_bot

    stores = get_stores()

    # One bot per session: retire any bot still running for it before starting
    # another, so a reload or a second Call press replaces rather than doubles.
    await _retire_existing_bot(session_id)

    async def _on_connection(connection):
        task = asyncio.create_task(
            run_voice_bot(connection, stores=stores, session_id=session_id)
        )
        _bots[session_id] = task

        def _cleanup(t: asyncio.Task) -> None:
            if _bots.get(session_id) is t:
                _bots.pop(session_id, None)

        task.add_done_callback(_cleanup)

    # from_dict() is a plain `cls(**data)`, so any extra key raises TypeError.
    # `request_data` is the field meant for application payload, so session_id
    # travels there and the rest of the body is filtered to what the dataclass
    # actually accepts.
    import dataclasses

    accepted = {f.name for f in dataclasses.fields(SmallWebRTCRequest)}
    payload = {k: v for k, v in body.items() if k in accepted}
    payload["request_data"] = {
        **(payload.get("request_data") or {}),
        "session_id": session_id,
    }

    answer = await _get_handler().handle_web_request(
        SmallWebRTCRequest.from_dict(payload), _on_connection
    )
    return answer
