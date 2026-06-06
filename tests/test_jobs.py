"""Background job registry: subscribe, replay, global serial lock, cancel."""

from __future__ import annotations

import asyncio
import pytest


@pytest.mark.asyncio
async def test_start_emits_and_completes(isolated_paths):
    from personal_db import jobs

    async def runner(job):
        await jobs.emit(job, {"type": "token", "data": "a"})
        await jobs.emit(job, {"type": "token", "data": "b"})

    job = await jobs.start("sid-1", "q", runner)
    await job.task

    seq = [e["data"] for e in job.events if e["type"] == "token"]
    assert seq == ["a", "b"]
    assert job.done is True
    assert any(e["type"] == "done" for e in job.events)


@pytest.mark.asyncio
async def test_subscribe_replays_history_and_streams(isolated_paths):
    from personal_db import jobs

    async def runner(job):
        await jobs.emit(job, {"type": "token", "data": "x"})
        await asyncio.sleep(0.02)
        await jobs.emit(job, {"type": "token", "data": "y"})

    job = await jobs.start("sid-2", "q", runner)
    # Subscribe AFTER the first emit so we exercise replay.
    await asyncio.sleep(0.01)
    collected: list[dict] = []
    async for evt in jobs.subscribe(job):
        collected.append(evt)
    assert {"x", "y"}.issubset({e.get("data") for e in collected if e["type"] == "token"})
    assert collected[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_global_lock_serializes(isolated_paths):
    """Two jobs started concurrently must run serially."""
    from personal_db import jobs

    observed: list[str] = []

    async def runner(name):
        async def _r(job):
            observed.append(f"start:{name}")
            await asyncio.sleep(0.05)
            observed.append(f"end:{name}")
        return _r

    a = await jobs.start("A", "q", await runner("A"))
    b = await jobs.start("B", "q", await runner("B"))
    await asyncio.gather(a.task, b.task)

    # Either A then B, or B then A — but NEVER interleaved.
    assert observed[:2] in (["start:A", "end:A"], ["start:B", "end:B"])
    assert observed[2:] in (["start:B", "end:B"], ["start:A", "end:A"])

    # The second one should have been told it was queued.
    second = b if observed[0].endswith("A") else a
    phases = [
        e["data"]["phase"] for e in second.events
        if e["type"] == "status" and isinstance(e.get("data"), dict)
    ]
    assert "queued" in phases


@pytest.mark.asyncio
async def test_duplicate_start_for_same_session_raises(isolated_paths):
    from personal_db import jobs

    async def slow(job):
        await asyncio.sleep(0.1)

    j = await jobs.start("sid-dup", "q", slow)
    with pytest.raises(RuntimeError):
        await jobs.start("sid-dup", "q2", slow)
    await j.task


@pytest.mark.asyncio
async def test_cancel_aborts_runner(isolated_paths):
    from personal_db import jobs

    started = asyncio.Event()

    async def long(job):
        started.set()
        await asyncio.sleep(5)

    j = await jobs.start("sid-cancel", "q", long)
    await started.wait()
    cancelled = await jobs.cancel("sid-cancel")
    assert cancelled is True
    assert j.done is True


@pytest.mark.asyncio
async def test_runner_exception_is_captured(isolated_paths):
    from personal_db import jobs

    async def boom(job):
        raise ValueError("boom")

    j = await jobs.start("sid-err", "q", boom)
    await j.task
    assert j.error and "boom" in j.error
    assert any(e["type"] == "error" for e in j.events)
    assert j.done is True
