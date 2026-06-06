"""Background generation jobs decoupled from HTTP request lifecycle.

One job per session, AND one active LLM run at a time globally (a process-wide
asyncio.Lock serializes runners). Jobs queued behind the lock emit a "queued"
status event so the UI can show it.

The job task survives any client disconnect — navigation never cancels work.
Subscribers attach by calling `subscribe(job)` which replays history and yields
new events as they happen.

Lifecycle:

    POST /chat/{sid}/ask    →  jobs.start(sid, q)  →  spawns task
    GET  /chat/{sid}/stream →  jobs.get(sid) + jobs.subscribe(job)
    POST /chat/{sid}/delete →  jobs.cancel(sid)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable

log = logging.getLogger(__name__)

# Process-wide lock: at most one runner doing real work (LLM calls, retrieval,
# compaction) at a time. Ollama can only do one inference at a time anyway —
# this just makes the queueing explicit and lets the UI show a "queued" state.
#
# Lazy-create per running event loop so the lock can't be bound to a stale
# loop (matters for test isolation, and is harmless in production).
_run_locks: dict[int, asyncio.Lock] = {}


def _get_run_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    key = id(loop)
    lock = _run_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _run_locks[key] = lock
    return lock


@dataclass
class GenJob:
    session_id: str
    user_text: str
    events: list[dict] = field(default_factory=list)
    done: bool = False
    error: str | None = None
    task: asyncio.Task | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    # Set whenever events is appended or done flips True. Subscribers wait on this.
    pulse: asyncio.Event = field(default_factory=asyncio.Event)


# session_id -> GenJob
_jobs: dict[str, GenJob] = {}
# Retain done jobs briefly so a late subscriber can still replay the final events.
_KEEP_DONE_SECONDS = 30.0


def get(session_id: str) -> GenJob | None:
    """Return the current job for a session, if any (including recently-done)."""
    return _jobs.get(session_id)


def get_active(session_id: str) -> GenJob | None:
    """Return the job only if it is still running."""
    j = _jobs.get(session_id)
    return j if (j and not j.done) else None


async def emit(job: GenJob, evt: dict) -> None:
    """Append an event to the job and wake any subscribers."""
    job.events.append(evt)
    job.pulse.set()


async def mark_error(job: GenJob, msg: str) -> None:
    job.error = msg
    job.events.append({"type": "error", "data": msg})
    job.pulse.set()


async def mark_done(job: GenJob) -> None:
    job.done = True
    job.finished_at = time.time()
    # 'done' event lets streaming clients close cleanly.
    job.events.append({"type": "done", "data": {}})
    job.pulse.set()


def _gc_jobs() -> None:
    """Drop done jobs older than _KEEP_DONE_SECONDS so the registry doesn't grow."""
    now = time.time()
    stale = [
        sid for sid, j in _jobs.items()
        if j.done and j.finished_at and (now - j.finished_at) > _KEEP_DONE_SECONDS
    ]
    for sid in stale:
        _jobs.pop(sid, None)


async def start(
    session_id: str,
    user_text: str,
    runner: Callable[[GenJob], Awaitable[None]],
) -> GenJob:
    """Register a new job and kick off `runner(job)` as a background task.

    Raises RuntimeError if a job is already in flight for this session."""
    _gc_jobs()
    existing = _jobs.get(session_id)
    if existing and not existing.done:
        raise RuntimeError("a generation is already in progress for this session")

    job = GenJob(session_id=session_id, user_text=user_text)
    _jobs[session_id] = job

    async def _wrapped() -> None:
        try:
            lock = _get_run_lock()
            if lock.locked():
                await emit(
                    job,
                    {"type": "status", "data": {"phase": "queued", "text": "queued — waiting for current generation"}},
                )
            async with lock:
                await runner(job)
        except asyncio.CancelledError:
            log.info("job %s cancelled", session_id)
            await emit(job, {"type": "status", "data": {"phase": "cancelled", "text": "cancelled"}})
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("job %s failed", session_id)
            await mark_error(job, str(e))
        finally:
            await mark_done(job)

    job.task = asyncio.create_task(_wrapped(), name=f"genjob-{session_id}")
    return job


async def cancel(session_id: str) -> bool:
    """Cancel the active job for this session, if any. Returns True if cancelled."""
    job = _jobs.get(session_id)
    if not job or job.done or job.task is None:
        return False
    job.task.cancel()
    try:
        await job.task
    except (asyncio.CancelledError, Exception):
        pass
    return True


async def subscribe(job: GenJob) -> AsyncIterator[dict]:
    """Yield SSE events for this job. Replays history, then streams new events
    until the job is done. Multiple subscribers can attach concurrently or
    sequentially; each sees the full event log from index 0."""
    sent = 0
    while True:
        while sent < len(job.events):
            evt = job.events[sent]
            sent += 1
            yield evt
        if job.done:
            return
        # Wait for the next emit/mark_done.
        job.pulse.clear()
        try:
            await job.pulse.wait()
        except asyncio.CancelledError:
            return
