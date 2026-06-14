"""Lightweight search-only API server.

Exposes only the retrieval endpoints — no web UI, no chat, no sessions.
Boots faster and can run independently of the main app.

Run:
    uvicorn personal_db.search_app:app --host 0.0.0.0 --port 8766
or:
    ./run_search.sh
"""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .retrieve import RetrievalContext, hybrid_retrieve, vector_retrieve
from .stores import configure_llama_index, init_stores

app = FastAPI(title="Personal Database — Search API", version="1.0")

_stores = None


def _get_stores():
    global _stores
    if _stores is None:
        configure_llama_index()
        _stores = init_stores()
    return _stores


# ─── shared retrieval logic ──────────────────────────────────────────────────


def _run_search(q: str, mode: str, top_k: int | None) -> dict:
    stores = _get_stores()

    if mode == "vector":
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


# ─── endpoints ──────────────────────────────────────────────────────────────


class SearchBody(BaseModel):
    q: str
    mode: str = "hybrid"  # "hybrid" | "vector"
    top_k: int | None = None


@app.get("/search")
async def search_get(q: str, mode: str = "hybrid", top_k: int | None = None):
    """Search the corpus.

    - **q**: query string
    - **mode**: `hybrid` (vector + graph, default) or `vector` (vector only)
    - **top_k**: max chunks to return (default: server config)

    Example: `curl "http://localhost:8766/search?q=WebRTC+signalling"`
    """
    if not q.strip():
        raise HTTPException(400, "q must not be empty")
    result = await asyncio.to_thread(_run_search, q.strip(), mode, top_k)
    return JSONResponse(result)


@app.post("/search")
async def search_post(body: SearchBody):
    """Same as GET /search but accepts a JSON body.

    ```json
    {"q": "WebRTC signalling", "mode": "hybrid", "top_k": 5}
    ```
    """
    if not body.q.strip():
        raise HTTPException(400, "q must not be empty")
    result = await asyncio.to_thread(_run_search, body.q.strip(), body.mode, body.top_k)
    return JSONResponse(result)


@app.get("/health")
async def health():
    return {"status": "ok"}


def serve() -> None:
    """Entry point for `pdb-search` CLI command and ./run_search.sh."""
    import os
    import uvicorn

    uvicorn.run(
        "personal_db.search_app:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8766")),
        log_level="warning",
    )
