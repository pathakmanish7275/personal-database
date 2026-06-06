"""Ingestion: turn files into chunks indexed in Qdrant + a Kuzu knowledge graph."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

from .config import config
from .stores import Stores, configure_llama_index, init_stores

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}


def _read_md_or_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    import pymupdf4llm
    return pymupdf4llm.to_markdown(str(path))


def _read(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in {".md", ".markdown", ".txt"}:
        return _read_md_or_text(path)
    raise ValueError(f"unsupported file type: {suffix}")


def _doc_id_for(path: Path, text: str) -> str:
    h = hashlib.sha1()
    h.update(str(path.resolve()).encode())
    h.update(text[:4096].encode("utf-8", errors="replace"))
    return h.hexdigest()[:16]


def _build_document(path: Path, text: str) -> tuple[str, Document]:
    doc_id = _doc_id_for(path, text)
    doc = Document(
        text=text,
        metadata={
            # Namespaced so it doesn't collide with LlamaIndex's auto-assigned
            # `doc_id`/`document_id` payload keys when chunks are stored in Qdrant.
            "pdb_doc_id": doc_id,
            "path": str(path.resolve()),
            "name": path.name,
            "source": path.suffix.lstrip(".").lower(),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "chars": len(text),
        },
    )
    return doc_id, doc


def iter_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target] if target.suffix.lower() in SUPPORTED_SUFFIXES else []
    return [p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]


def ingest_path(target: Path, stores: Stores | None = None) -> dict:
    """Ingest one file or all supported files under a directory. Returns a summary dict."""
    configure_llama_index()
    if stores is None:
        stores = init_stores()

    files = iter_files(target)
    if not files:
        return {"ingested": 0, "files": [], "skipped_reason": "no supported files"}

    docs: list[Document] = []
    seen: list[dict] = []
    for f in files:
        try:
            text = _read(f)
        except Exception as e:
            seen.append({"path": str(f), "status": "error", "error": str(e)})
            continue
        if not text.strip():
            seen.append({"path": str(f), "status": "empty"})
            continue
        doc_id, doc = _build_document(f, text)
        docs.append(doc)
        seen.append({"path": str(f), "status": "queued", "doc_id": doc_id, "chars": len(text)})

    if not docs:
        return {"ingested": 0, "files": seen}

    splitter = SentenceSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    storage_context = StorageContext.from_defaults(vector_store=stores.vector_store)
    VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=False,
    )
    return {"ingested": len(docs), "files": seen}


def existing_doc_ids(stores: Stores) -> set[str]:
    """The doc_ids currently present in the vector store. Used as a dedup gate."""
    return {d["doc_id"] for d in list_documents(stores) if d.get("doc_id")}


def ingest_one(
    path: Path,
    stores: Stores | None = None,
    *,
    known_doc_ids: set[str] | None = None,
) -> dict:
    """Ingest a single file and return a per-file result dict.

    Skips a file whose doc_id is already in the vector store (no re-embedding).
    If the vectors exist but the graph is missing the doc, runs a graph-only
    pass instead. Pass `known_doc_ids` to amortize the existence check across a
    batch; otherwise we scan the vector store per call."""
    configure_llama_index()
    if stores is None:
        stores = init_stores()

    try:
        text = _read(path)
    except Exception as e:  # noqa: BLE001
        return {"path": str(path), "name": path.name, "status": "error", "error": str(e)}
    if not text.strip():
        return {"path": str(path), "name": path.name, "status": "empty"}

    doc_id, doc = _build_document(path, text)

    # ─── dedup gate ───────────────────────────────────────────────────────
    have = known_doc_ids if known_doc_ids is not None else existing_doc_ids(stores)
    if doc_id in have:
        # Vector copy already exists. If the graph is missing this doc, do a
        # graph-only pass so /graph stays in sync without re-embedding.
        if (
            config.kg_enabled
            and stores.graph is not None
            and doc_id not in stores.graph.document_ids()
        ):
            splitter = SentenceSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
            nodes = splitter.get_nodes_from_documents([doc])
            kg = _extract_to_graph(stores.graph, _doc_record(doc_id, doc), nodes)
            return {
                "path": str(path),
                "name": path.name,
                "status": "graph_only",
                "doc_id": doc_id,
                "chunks": len(nodes),
                "entities": kg["entities"],
                "relations": kg["relations"],
            }
        return {
            "path": str(path),
            "name": path.name,
            "status": "skipped",
            "doc_id": doc_id,
            "reason": "already ingested",
        }

    splitter = SentenceSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    nodes = splitter.get_nodes_from_documents([doc])
    storage_context = StorageContext.from_defaults(vector_store=stores.vector_store)
    VectorStoreIndex(nodes, storage_context=storage_context, show_progress=False)

    result = {
        "path": str(path),
        "name": path.name,
        "status": "ingested",
        "doc_id": doc_id,
        "chars": len(text),
        "chunks": len(nodes),
    }

    if config.kg_enabled and stores.graph is not None:
        try:
            kg = _extract_to_graph(stores.graph, _doc_record(doc_id, doc), nodes)
            result["entities"] = kg["entities"]
            result["relations"] = kg["relations"]
        except Exception as e:  # noqa: BLE001
            log.exception("graph population failed for %s", path)
            result["kg_error"] = str(e)
    return result


def _doc_record(doc_id: str, doc: Document) -> dict:
    m = doc.metadata
    return {
        "doc_id": doc_id,
        "name": m.get("name"),
        "path": m.get("path"),
        "source": m.get("source"),
        "ingested_at": m.get("ingested_at"),
    }


def _extract_to_graph(graph, doc_record: dict, nodes) -> dict:
    """Extract entities + relations from each chunk and write them to the graph."""
    from .extract import extract_for_chunk

    chunks = []
    for n in nodes:
        content = n.get_content()
        entities, relations = extract_for_chunk(content)
        chunks.append(
            {
                "chunk_id": n.node_id,
                "text": content,
                "entities": entities,
                "relations": relations,
            }
        )
    return graph.populate_document(doc_record, chunks)


def extract_file_to_graph(path: Path, stores: Stores) -> dict:
    """Re-read a file already in the vector store and (re)build its graph entry.

    Used by the backfill that builds the graph for a corpus ingested before the
    graph existed. Does NOT re-embed — only parses + extracts + populates Kuzu."""
    configure_llama_index()
    try:
        text = _read(path)
    except Exception as e:  # noqa: BLE001
        return {"path": str(path), "name": path.name, "status": "error", "error": str(e)}
    if not text.strip():
        return {"path": str(path), "name": path.name, "status": "empty"}

    doc_id, doc = _build_document(path, text)
    splitter = SentenceSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    nodes = splitter.get_nodes_from_documents([doc])
    kg = _extract_to_graph(stores.graph, _doc_record(doc_id, doc), nodes)
    return {
        "path": str(path),
        "name": path.name,
        "status": "ingested",
        "doc_id": doc_id,
        "chunks": len(nodes),
        "entities": kg["entities"],
        "relations": kg["relations"],
    }


def graph_backfill_targets(stores: Stores) -> list[Path]:
    """Files in the vector store that still exist on disk but aren't in the graph yet.

    Keys on file *path* (stable across schema changes) rather than doc_id —
    pre-rename corpora report their LlamaIndex-internal UUID as doc_id, which
    wouldn't match the content-hash doc_id the graph uses."""
    have_paths = stores.graph.document_paths() if stores.graph else set()
    targets: list[Path] = []
    seen: set[str] = set()
    for d in list_documents(stores):
        p = d.get("path")
        if not p or p in seen:
            continue
        seen.add(p)
        if p in have_paths:
            continue
        if Path(p).exists():
            targets.append(Path(p))
    return targets


def list_documents(stores: Stores | None = None) -> list[dict]:
    """Pull a deduplicated list of ingested documents from Qdrant payloads."""
    if stores is None:
        stores = init_stores()

    client = stores.qdrant_client
    coll = config.qdrant_collection
    try:
        client.get_collection(coll)
    except Exception:
        return []

    seen: dict[str, dict] = {}
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=coll,
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=next_offset,
        )
        for p in points:
            payload = p.payload or {}
            meta = payload.get("metadata") or payload
            # Prefer our namespaced key; fall back to LlamaIndex's parent-doc
            # UUID so corpora ingested before the rename still show up.
            doc_id = (
                meta.get("pdb_doc_id")
                or payload.get("pdb_doc_id")
                or meta.get("ref_doc_id")
                or payload.get("ref_doc_id")
            )
            if not doc_id:
                continue
            entry = seen.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "name": meta.get("name") or "?",
                    "path": meta.get("path"),
                    "source": meta.get("source"),
                    "ingested_at": meta.get("ingested_at"),
                    "chars": meta.get("chars"),
                    "chunks": 0,
                },
            )
            entry["chunks"] += 1
        if not next_offset:
            break
    return sorted(seen.values(), key=lambda x: x.get("ingested_at") or "", reverse=True)
