"""Hybrid retrieval: vector (Qdrant) + graph (Kuzu) → fused context for the LLM.

Every chat turn fires both retrievals and combines them:

  vector  — top-K semantic matches over the chunk embeddings (good at phrasing).
  graph   — entities GLiNER finds in the question → chunks that mention them
            via Kuzu, plus the structured relations around those entities
            (good at connections and aliases the vector pipeline misses).

The two chunk lists are fused with reciprocal-rank fusion (RRF), so a chunk
that lands in *both* lists rises to the top. The relations are passed to the
LLM as a separate, non-cited structured block so the model can use the
connections without inventing them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever

from .config import config
from .stores import Stores

RRF_K = 60


@dataclass
class RetrievedChunk:
    text: str
    name: str
    path: str
    doc_id: str | None = None
    vector_score: float | None = None
    graph_hits: int = 0
    origin: str = "semantic"  # "semantic" | "graph" | "both"


@dataclass
class RetrievalContext:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    relations: list[tuple[str, str, str]] = field(default_factory=list)
    query_entities: list[str] = field(default_factory=list)
    # Raw counts pre-fusion, surfaced to the UI so the user can see what each
    # side actually contributed.
    graph_chunk_count: int = 0
    semantic_chunk_count: int = 0


def _doc_name(meta: dict) -> str:
    name = meta.get("name")
    if name and name != "?":
        return name
    p = meta.get("path") or ""
    if p:
        return Path(p).name or "(unnamed)"
    return "(unnamed)"


def _filter_nodes(nodes: list) -> list:
    """Apply the similarity floor, keeping the single best hit if all are below."""
    if not nodes:
        return []
    floor = config.min_similarity_score
    kept = [n for n in nodes if (getattr(n, "score", None) or 0.0) >= floor]
    if not kept:
        kept = [nodes[0]]
    return kept


def vector_retrieve(query: str, stores: Stores) -> list[RetrievedChunk]:
    """Pure-semantic retrieval over the Qdrant vector store."""
    index = VectorStoreIndex.from_vector_store(vector_store=stores.vector_store)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=config.top_k_vector)
    nodes = _filter_nodes(retriever.retrieve(query))
    out: list[RetrievedChunk] = []
    for n in nodes:
        m = n.metadata or {}
        out.append(
            RetrievedChunk(
                text=(n.text or "").strip(),
                name=_doc_name(m),
                path=m.get("path") or "",
                doc_id=m.get("pdb_doc_id") or m.get("ref_doc_id"),
                vector_score=getattr(n, "score", None),
                origin="semantic",
            )
        )
    return out


def graph_retrieve(
    query: str, stores: Stores
) -> tuple[list[RetrievedChunk], list[tuple[str, str, str]], list[str]]:
    """Graph-side retrieval: GLiNER on the query → Kuzu lookup → chunks + relations.

    Returns (chunks, relations, query_entity_labels). All empty if the KG is
    disabled, the extractor is unavailable, or no query entity matches the graph."""
    if not config.kg_enabled or stores.graph is None:
        return [], [], []

    # Local import keeps torch out of the import graph for code paths that
    # don't use the graph (e.g. KG_ENABLED=false).
    from .extract.entities import extract_entities

    # Lower threshold for queries — they're short and the model has less signal.
    raw = extract_entities(query, threshold=0.35)
    if not raw:
        return [], [], []

    resolved = stores.graph.resolve_query_entities([r[0] for r in raw])
    if not resolved:
        return [], [], [r[0] for r in raw]

    keys = [r["key"] for r in resolved]
    rows = stores.graph.chunks_for_entities(keys, limit=config.top_k_graph)
    relations = stores.graph.relations_for_entities(keys, limit=25)

    chunks = [
        RetrievedChunk(
            text=r["text"],
            name=r["name"],
            path=r["path"],
            doc_id=r["doc_id"],
            graph_hits=r["hits"],
            origin="graph",
        )
        for r in rows
    ]
    return chunks, relations, [r["label"] for r in resolved]


def _rrf_fuse(vec: list[RetrievedChunk], gph: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
    """Reciprocal-rank fusion of two ranked lists. Items present in both
    accumulate score and get their origin flipped to 'both'."""

    def key(c: RetrievedChunk) -> tuple[str, str]:
        # Doc_id alone may be None for old-format payloads; combine with a text
        # prefix so different chunks of the same doc don't collide.
        return (c.doc_id or "", c.text[:200])

    scores: dict[tuple[str, str], float] = {}
    items: dict[tuple[str, str], RetrievedChunk] = {}

    def push(lst: list[RetrievedChunk], source: str) -> None:
        for i, c in enumerate(lst):
            k = key(c)
            scores[k] = scores.get(k, 0.0) + 1.0 / (RRF_K + i + 1)
            if k in items:
                existing = items[k]
                if source == "graph":
                    existing.graph_hits = max(existing.graph_hits, c.graph_hits)
                else:
                    existing.vector_score = max(existing.vector_score or 0.0, c.vector_score or 0.0)
                existing.origin = "both"
            else:
                items[k] = c

    push(vec, "vector")
    push(gph, "graph")

    ranked = sorted(items.values(), key=lambda c: scores[key(c)], reverse=True)
    return ranked[:top_n]


def hybrid_retrieve(query: str, stores: Stores) -> RetrievalContext:
    """Run vector + graph retrieval and return a fused context.

    Runs synchronously in this thread — the caller (chat.run_generation) wraps
    it in asyncio.to_thread, so the event loop is never blocked."""
    vec = vector_retrieve(query, stores)
    gph, rels, q_ents = graph_retrieve(query, stores)
    top_n = config.top_k_vector + max(1, config.top_k_graph // 2)
    fused = _rrf_fuse(vec, gph, top_n=top_n)
    return RetrievalContext(
        chunks=fused,
        relations=rels,
        query_entities=q_ents,
        graph_chunk_count=len(gph),
        semantic_chunk_count=len(vec),
    )
