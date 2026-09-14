"""Background ingestion job — decoupled from the HTTP request lifecycle.

Ingesting a batch of files (parse → chunk → embed → Qdrant) can take minutes,
especially for PDFs. If that work is tied to the POST request, navigating away
mid-ingest cancels it and only the files processed so far survive. So ingestion
runs as a standalone asyncio.Task: it keeps going whether or not anyone is
watching, and the page subscribes to its progress over SSE.

Only one ingestion runs at a time (it's heavy, and it shares the global run
lock with chat generation so the two never fight over Ollama). Progress events:

    {"type": "start",    "data": {"total": N}}
    {"type": "progress", "data": {"done": i, "total": N, "name": ..., "status": ...}}
    {"type": "done",     "data": {"ingested": k, "failed": m, "total": N}}
    {"type": "error",    "data": "message"}
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable

log = logging.getLogger(__name__)


@dataclass
class IngestJob:
    targets: list[Path]
    events: list[dict] = field(default_factory=list)
    done: bool = False
    error: str | None = None
    task: asyncio.Task | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    # Set whenever events is appended or done flips True. Subscribers wait on it.
    pulse: asyncio.Event = field(default_factory=asyncio.Event)


# At most one ingestion at a time, process-wide.
_current: IngestJob | None = None
# Keep a finished job around briefly so a late subscriber still sees the result.
_KEEP_DONE_SECONDS = 60.0


def get() -> IngestJob | None:
    """Return the current ingestion job, if any (including recently-finished)."""
    return _current


def get_active() -> IngestJob | None:
    """Return the job only if it is still running."""
    return _current if (_current and not _current.done) else None


async def emit(job: IngestJob, evt: dict) -> None:
    job.events.append(evt)
    job.pulse.set()


async def _terminate(job: IngestJob, *, summary: dict | None = None, error: str | None = None) -> None:
    if error is not None:
        job.error = error
        job.events.append({"type": "error", "data": error})
    else:
        job.events.append({"type": "done", "data": summary or {}})
    job.done = True
    job.finished_at = time.time()
    job.pulse.set()


async def start(
    targets: list[Path],
    runner: Callable[[IngestJob], Awaitable[dict]],
) -> IngestJob:
    """Register a new ingestion job and kick off `runner(job)` as a background task.

    `runner` should emit progress events and return a summary dict; this wrapper
    handles the terminal done/error event. Raises RuntimeError if one is already
    in flight."""
    global _current
    if _current and not _current.done:
        raise RuntimeError("an ingestion is already in progress")

    # Drop a stale finished job so we don't leak it.
    if _current and _current.done:
        _current = None

    job = IngestJob(targets=list(targets))
    _current = job

    async def _wrapped() -> None:
        try:
            summary = await runner(job)
            await _terminate(job, summary=summary)
        except asyncio.CancelledError:
            log.info("ingestion cancelled")
            await _terminate(job, error="cancelled")
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("ingestion failed")
            await _terminate(job, error=str(e))

    job.task = asyncio.create_task(_wrapped(), name="ingest-job")
    return job


async def cancel() -> bool:
    """Cancel the active ingestion, if any."""
    job = _current
    if not job or job.done or job.task is None:
        return False
    job.task.cancel()
    try:
        await job.task
    except (asyncio.CancelledError, Exception):
        pass
    return True


async def drain_all(timeout: float = 300.0) -> None:
    """Wait for the active ingestion (if any) to finish, up to `timeout`.

    Used on shutdown so ingestion state stays consistent and any Ollama
    embedding call in progress is never abandoned mid-request."""
    task = _current.task if _current else None
    if task and not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except (asyncio.TimeoutError, Exception):
            pass


async def subscribe(job: IngestJob) -> AsyncIterator[dict]:
    """Yield SSE events for this job, replaying history first then streaming new
    ones until done. Reconnecting clients always see the full log from index 0."""
    sent = 0
    while True:
        while sent < len(job.events):
            evt = job.events[sent]
            sent += 1
            yield evt
        if job.done:
            return
        job.pulse.clear()
        try:
            await job.pulse.wait()
        except asyncio.CancelledError:
            return
