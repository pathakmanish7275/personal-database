"""FastAPI web UI.

Architecture:

  POST /chat/{sid}/ask       — starts a background generation task. Returns 200
                                with the job snapshot. The task survives any
                                client disconnect.
  GET  /chat/{sid}/stream    — subscribes to the active job's events via SSE.
                                Replays any past events so reconnecting clients
                                see what they missed.
  POST /chat/{sid}/cancel    — cancels the active job for this session.
  POST /chat/{sid}/delete    — cancels + deletes the session.

All endpoints are async so SSE streams don't tie up the request thread pool.
The heavy LLM / DB work happens in asyncio.to_thread inside the background task.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .. import chat as chat_engine
from .. import ingest_jobs
from .. import jobs as jobs_mod
from .. import llm_runtime
from .. import sessions as sess_store
from ..config import config
from ..retrieve import hybrid_retrieve, vector_retrieve
from ..stores import configure_llama_index
from ..ingest import (
    SUPPORTED_SUFFIXES,
    existing_doc_ids,
    extract_file_to_graph,
    graph_backfill_targets,
    ingest_one,
    iter_files,
    list_documents,
)
from ..stores import init_stores

WEB_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))

def _asset_version() -> str:
    """Newest mtime across static assets, as a cache-busting query value.

    Without this the browser keeps serving a cached stylesheet after an edit —
    a CSS fix can look like it simply did not work, which cost real debugging
    time once already."""
    newest = 0.0
    static = WEB_DIR / "static"
    if static.is_dir():
        for f in static.rglob("*"):
            if f.is_file():
                newest = max(newest, f.stat().st_mtime)
    return str(int(newest))


# Recomputed per render so edits show up without a restart in development.
templates.env.globals["asset_v"] = _asset_version

# How long shutdown waits for in-flight generations before giving up. Chat
# generations typically finish in well under a minute; the cap just bounds a
# hung model server.
SHUTDOWN_DRAIN_SECONDS = float(os.getenv("SHUTDOWN_DRAIN_SECONDS", "300"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Safe shutdown: never abandon an in-flight LLM call. Dropping a stream
    # mid-generation can crash the FLM/NPU server, so wait for active jobs to
    # finish and their streams to close cleanly before the process exits.
    try:
        await jobs_mod.drain_all(timeout=SHUTDOWN_DRAIN_SECONDS)
        await ingest_jobs.drain_all(timeout=SHUTDOWN_DRAIN_SECONDS)
        # Belt-and-braces: any LLM calls outside the job system (shouldn't be
        # any, but the counter is cheap insurance).
        deadline = time.monotonic() + 30.0
        while llm_runtime.active_calls() > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.2)
    except Exception:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).exception("error during shutdown drain")


app = FastAPI(title="Personal Database", docs_url="/docs", redoc_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

# Voice shares this app on purpose: one process means one NPU call gate covering
# both channels, and one origin for the browser. The router is thin — pipecat
# itself is imported lazily inside it.
from ..voice.routes import router as voice_router  # noqa: E402

app.include_router(voice_router)

_stores = None


def get_stores():
    global _stores
    if _stores is None:
        _stores = init_stores()
    return _stores


# ─── pages ──────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return RedirectResponse(url="/chat")


@app.get("/chat", response_class=HTMLResponse)
async def chat_index(request: Request):
    sessions, docs = await asyncio.gather(
        asyncio.to_thread(sess_store.list_sessions),
        asyncio.to_thread(list_documents, get_stores()),
    )
    stats = {
        "sessions": len(sessions),
        "docs": len(docs),
        "chunks": sum(d.get("chunks", 0) for d in docs),
    }
    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "sessions": sessions,
            "current": None,
            "messages": [],
            "active_job": False,
            "stats": stats,
        },
    )


@app.post("/chat/new")
async def chat_new():
    s = await asyncio.to_thread(sess_store.create_session)
    return RedirectResponse(url=f"/chat/{s.id}", status_code=303)


@app.get("/chat/{sid}", response_class=HTMLResponse)
async def chat_view(request: Request, sid: str):
    session = await asyncio.to_thread(sess_store.get_session, sid)
    if not session:
        raise HTTPException(404, "session not found")
    sessions, messages = await asyncio.gather(
        asyncio.to_thread(sess_store.list_sessions),
        asyncio.to_thread(sess_store.get_messages, sid),
    )
    msgs = []
    for m in messages:
        cites = []
        if m.citations:
            try:
                cites = json.loads(m.citations)
            except Exception:
                cites = []
        msgs.append({"role": m.role, "content": m.content, "citations": cites})

    active = jobs_mod.get_active(sid)
    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "sessions": sessions,
            "current": session,
            "messages": msgs,
            "summary": session.summary,
            "summary_up_to_msg_id": session.summary_up_to_msg_id,
            "active_job": bool(active),
        },
    )


@app.post("/chat/{sid}/delete")
async def chat_delete(sid: str):
    await jobs_mod.cancel(sid)
    await asyncio.to_thread(sess_store.delete_session, sid)
    return RedirectResponse(url="/chat", status_code=303)


# ─── generation (background task pattern) ──────────────────────────────────


class AskBody(BaseModel):
    q: str


@app.post("/chat/{sid}/ask")
async def chat_ask(sid: str, body: AskBody):
    """Start a background generation job for this session. Returns immediately
    with a small payload — the actual response is consumed via /stream."""
    session = await asyncio.to_thread(sess_store.get_session, sid)
    if not session:
        raise HTTPException(404, "session not found")
    q = (body.q or "").strip()
    if not q:
        raise HTTPException(400, "empty question")

    stores = get_stores()

    async def runner(job: jobs_mod.GenJob) -> None:
        await chat_engine.run_generation(job, stores)

    try:
        job = await jobs_mod.start(sid, q, runner)
    except RuntimeError as e:
        # Already a job in flight for this session.
        raise HTTPException(409, str(e))
    return JSONResponse({"session_id": sid, "started_at": job.started_at})


@app.get("/chat/{sid}/stream")
async def chat_stream(sid: str):
    job = jobs_mod.get(sid)
    if not job:
        raise HTTPException(404, "no active or recent generation for this session")

    async def event_gen():
        async for evt in jobs_mod.subscribe(job):
            yield {"event": evt["type"], "data": json.dumps(evt.get("data"))}

    return EventSourceResponse(event_gen())


@app.get("/chat/{sid}/messages")
async def chat_messages(sid: str, after: int = 0):
    """Messages for a session, as JSON.

    Used by the voice UI: a spoken turn is persisted by the same code path as a
    typed one, so the page polls this during a call to show the transcript in
    the same thread. `after` is the number of messages the client already has."""
    session = await asyncio.to_thread(sess_store.get_session, sid)
    if not session:
        raise HTTPException(404, "session not found")
    msgs = await asyncio.to_thread(sess_store.get_messages, sid)
    out = []
    for m in msgs[after:]:
        out.append(
            {
                "role": m.role,
                "content": m.content or "",
                "citations": json.loads(m.citations) if m.citations else None,
            }
        )
    return JSONResponse({"total": len(msgs), "messages": out})


@app.post("/chat/{sid}/cancel")
async def chat_cancel(sid: str):
    cancelled = await jobs_mod.cancel(sid)
    return JSONResponse({"cancelled": cancelled})


# ─── library / ingest (unchanged behavior, now async) ──────────────────────


@app.get("/library", response_class=HTMLResponse)
async def library(request: Request):
    docs = await asyncio.to_thread(list_documents, get_stores())
    total_chunks = sum(d.get("chunks", 0) for d in docs)
    return templates.TemplateResponse(
        request,
        "library.html",
        {"docs": docs, "total_chunks": total_chunks},
    )


@app.get("/ingest", response_class=HTMLResponse)
async def ingest_form(request: Request):
    return templates.TemplateResponse(
        request, "ingest.html", {"active": bool(ingest_jobs.get_active())}
    )


async def _run_ingest(job: ingest_jobs.IngestJob, stores) -> dict:
    """Ingest the job's target files one at a time, emitting progress events.

    Each file is processed under the shared run lock so ingestion never competes
    with chat generation for Ollama. The lock is acquired per-file (not for the
    whole batch) so a chat can interleave between files. Already-ingested files
    (matching doc_id) are skipped rather than re-embedded."""
    targets = job.targets
    total = len(targets)
    await ingest_jobs.emit(job, {"type": "start", "data": {"total": total}})

    # One scan of the vector store at the start; updated in-place as new docs land.
    have = await asyncio.to_thread(existing_doc_ids, stores)

    run_lock = jobs_mod._get_run_lock()
    ingested = 0
    skipped = 0
    failed = 0
    for i, p in enumerate(targets):
        await ingest_jobs.emit(
            job,
            {"type": "progress", "data": {"done": i, "total": total, "name": p.name, "status": "processing"}},
        )
        async with run_lock:
            res = await asyncio.to_thread(ingest_one, p, stores, known_doc_ids=have)
        status = res.get("status")
        if status in ("ingested", "graph_only"):
            ingested += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1
        if res.get("doc_id"):
            have.add(res["doc_id"])
        await ingest_jobs.emit(
            job,
            {
                "type": "progress",
                "data": {
                    "done": i + 1,
                    "total": total,
                    "name": res.get("name") or p.name,
                    "status": status,
                    "chunks": res.get("chunks"),
                    "entities": res.get("entities"),
                    "relations": res.get("relations"),
                    "error": res.get("error"),
                },
            },
        )
    return {"ingested": ingested, "skipped": skipped, "failed": failed, "total": total}


@app.post("/ingest/start")
async def ingest_start(
    path: str = Form(default=""),
    files: list[UploadFile] = File(default=[]),
):
    """Save any uploads, resolve the target file list, and launch a background
    ingestion job. Returns immediately; progress is consumed via /ingest/stream."""
    if ingest_jobs.get_active():
        raise HTTPException(409, "an ingestion is already in progress")

    targets: list[Path] = []

    for f in files:
        if not f or not f.filename:
            continue
        dest = config.raw_dir / Path(f.filename).name
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        if dest.suffix.lower() in SUPPORTED_SUFFIXES:
            targets.append(dest)

    if path:
        p = Path(path).expanduser()
        if not p.exists():
            raise HTTPException(400, f"path not found: {path}")
        targets.extend(iter_files(p))

    # De-dupe while preserving order (a path + an upload could collide).
    seen_paths: set[str] = set()
    unique: list[Path] = []
    for t in targets:
        key = str(t.resolve())
        if key not in seen_paths:
            seen_paths.add(key)
            unique.append(t)

    if not unique:
        raise HTTPException(400, "no supported files to ingest")

    stores = get_stores()

    async def runner(job: ingest_jobs.IngestJob) -> dict:
        return await _run_ingest(job, stores)

    try:
        job = await ingest_jobs.start(unique, runner)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return JSONResponse({"total": len(unique), "started_at": job.started_at})


@app.get("/ingest/stream")
async def ingest_stream():
    job = ingest_jobs.get()
    if not job:
        raise HTTPException(404, "no ingestion running")

    async def event_gen():
        async for evt in ingest_jobs.subscribe(job):
            yield {"event": evt["type"], "data": json.dumps(evt.get("data"))}

    return EventSourceResponse(event_gen())


# ─── knowledge graph ────────────────────────────────────────────────────────


@app.get("/graph", response_class=HTMLResponse)
async def graph_view(request: Request, e: str = "", q: str = ""):
    stores = get_stores()
    counts = await asyncio.to_thread(stores.graph.counts)
    if q.strip():
        entities = await asyncio.to_thread(stores.graph.search_entities, q.strip(), 40)
    else:
        entities = await asyncio.to_thread(stores.graph.top_entities, 40)
    detail = await asyncio.to_thread(stores.graph.entity_detail, e) if e.strip() else None
    targets = await asyncio.to_thread(graph_backfill_targets, stores)
    return templates.TemplateResponse(
        request,
        "graph.html",
        {
            "counts": counts,
            "entities": entities,
            "detail": detail,
            "selected": e,
            "query": q,
            "backfill_pending": len(targets),
            "active": bool(ingest_jobs.get_active()),
        },
    )


async def _run_backfill(job: ingest_jobs.IngestJob, stores) -> dict:
    """Build the graph for files already in the vector store, one at a time."""
    targets = job.targets
    total = len(targets)
    await ingest_jobs.emit(job, {"type": "start", "data": {"total": total}})

    run_lock = jobs_mod._get_run_lock()
    ok = 0
    failed = 0
    for i, p in enumerate(targets):
        await ingest_jobs.emit(
            job,
            {"type": "progress", "data": {"done": i, "total": total, "name": p.name, "status": "processing"}},
        )
        async with run_lock:
            res = await asyncio.to_thread(extract_file_to_graph, p, stores)
        if res.get("status") == "ingested":
            ok += 1
        else:
            failed += 1
        await ingest_jobs.emit(
            job,
            {
                "type": "progress",
                "data": {
                    "done": i + 1,
                    "total": total,
                    "name": res.get("name") or p.name,
                    "status": res.get("status"),
                    "entities": res.get("entities"),
                    "relations": res.get("relations"),
                    "error": res.get("error"),
                },
            },
        )
    return {"ingested": ok, "failed": failed, "total": total}


@app.get("/api/graph/data")
async def graph_data(limit: int = 120):
    """Nodes + edges for the graph visualisation."""
    stores = get_stores()
    if not stores.graph:
        return JSONResponse({"nodes": [], "edges": [], "enabled": False})
    snap = await asyncio.to_thread(stores.graph.graph_snapshot, limit)
    snap["enabled"] = True
    return JSONResponse(snap)


@app.post("/graph/build")
async def graph_build():
    """Backfill the graph from the existing corpus (reuses the ingestion job slot)."""
    if ingest_jobs.get_active():
        raise HTTPException(409, "an ingestion is already in progress")
    stores = get_stores()
    targets = await asyncio.to_thread(graph_backfill_targets, stores)
    if not targets:
        raise HTTPException(400, "graph is already up to date with the corpus")

    async def runner(job: ingest_jobs.IngestJob) -> dict:
        return await _run_backfill(job, stores)

    try:
        job = await ingest_jobs.start(targets, runner)
    except RuntimeError as ex:
        raise HTTPException(409, str(ex))
    return JSONResponse({"total": len(targets), "started_at": job.started_at})


# ─── search API ─────────────────────────────────────────────────────────────


class SearchBody(BaseModel):
    q: str
    mode: str = "hybrid"  # "hybrid" | "vector"
    top_k: int | None = None


def _search_response(q: str, mode: str, top_k: int | None) -> dict:
    """Run retrieval and return a serialisable dict."""
    configure_llama_index()
    stores = get_stores()

    if mode == "vector":
        from ..retrieve import RetrievalContext
        chunks = vector_retrieve(q, stores)
        if top_k:
            chunks = chunks[:top_k]
        ctx = RetrievalContext(chunks=chunks, semantic_chunk_count=len(chunks))
    else:
        ctx = hybrid_retrieve(q, stores)
        if top_k:
            ctx.chunks = ctx.chunks[:top_k]

    return {
        "query": q,
        "mode": mode,
        "semantic_count": ctx.semantic_chunk_count,
        "graph_count": ctx.graph_chunk_count,
        "query_entities": ctx.query_entities,
        "chunks": [
            {
                "rank": i + 1,
                "text": c.text,
                "name": c.name,
                "path": c.path,
                "doc_id": c.doc_id,
                "vector_score": c.vector_score,
                "graph_hits": c.graph_hits,
                "origin": c.origin,
            }
            for i, c in enumerate(ctx.chunks)
        ],
        "relations": [
            {"head": h, "predicate": p, "tail": t} for h, p, t in ctx.relations
        ],
    }


@app.get("/api/search")
async def search_get(q: str, mode: str = "hybrid", top_k: int | None = None):
    """Semantic (or hybrid) search over the corpus.

    - **q**: search query
    - **mode**: `hybrid` (vector + graph, default) or `vector` (vector only)
    - **top_k**: max chunks to return (defaults to server config)

    Example: `curl "http://localhost:8765/api/search?q=WebRTC+signalling"`
    """
    if not q.strip():
        raise HTTPException(400, "q must not be empty")
    result = await asyncio.to_thread(_search_response, q.strip(), mode, top_k)
    return JSONResponse(result)


@app.post("/api/search")
async def search_post(body: SearchBody):
    """Same as GET /api/search but accepts a JSON body — easier for scripts.

    ```json
    {"q": "WebRTC signalling", "mode": "hybrid", "top_k": 5}
    ```
    """
    if not body.q.strip():
        raise HTTPException(400, "q must not be empty")
    result = await asyncio.to_thread(_search_response, body.q.strip(), body.mode, body.top_k)
    return JSONResponse(result)
