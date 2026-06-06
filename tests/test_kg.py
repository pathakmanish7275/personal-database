"""Kuzu graph layer — schema, upserts, dedup, and queries.

No torch here: we feed synthetic entities/relations straight into the graph,
so this runs fast and verifies the Cypher works on the installed Kuzu."""

from __future__ import annotations

import kuzu
import pytest


@pytest.fixture
def graph(tmp_path):
    from personal_db.kg import Graph
    db = kuzu.Database(str(tmp_path / "g.db"))
    return Graph(db)


def _doc(doc_id="d1", name="notes.md"):
    return {"doc_id": doc_id, "name": name, "path": f"/x/{name}", "source": "md", "ingested_at": "2026-01-01"}


def test_populate_and_counts(graph):
    chunks = [
        {
            "chunk_id": "c1",
            "text": "Alex works at Northwind Labs on LiveKit.",
            "entities": [("Alex", "person"), ("Northwind Labs", "organization"), ("LiveKit", "project")],
            "relations": [("Alex", "works at", "Northwind Labs"), ("Alex", "works on", "LiveKit")],
        }
    ]
    summary = graph.populate_document(_doc(), chunks)
    assert summary["entities"] == 3
    assert summary["relations"] == 2

    counts = graph.counts()
    assert counts["documents"] == 1
    assert counts["chunks"] == 1
    assert counts["entities"] == 3
    assert counts["relations"] == 2
    assert counts["mentions"] == 3


def test_entity_dedup_across_chunks(graph):
    chunks = [
        {"chunk_id": "c1", "text": "Alex.", "entities": [("Alex", "person")], "relations": []},
        {"chunk_id": "c2", "text": "alex again.", "entities": [("alex", "person")], "relations": []},
    ]
    graph.populate_document(_doc(), chunks)
    # Case-insensitive dedup → one Entity, mention_count 2.
    assert graph.counts()["entities"] == 1
    top = graph.top_entities()
    assert len(top) == 1
    assert top[0]["mentions"] == 2


def test_entity_detail_relations_and_docs(graph):
    chunks = [
        {
            "chunk_id": "c1",
            "text": "x",
            "entities": [("Alex", "person"), ("Helix", "project")],
            "relations": [("Alex", "builds", "Helix")],
        }
    ]
    graph.populate_document(_doc(doc_id="d9", name="helix.md"), chunks)
    detail = graph.entity_detail("alex")
    assert detail is not None
    assert detail["name"] == "Alex"
    assert any(r["target"] == "Helix" and "build" in r["predicate"] for r in detail["out_relations"])
    assert {d["name"] for d in detail["documents"]} == {"helix.md"}

    # Incoming relation shows up on the target.
    tdetail = graph.entity_detail("Helix")
    assert any(r["source"] == "Alex" for r in tdetail["in_relations"])


def test_document_ids_and_idempotent_repopulate(graph):
    chunks = [{"chunk_id": "c1", "text": "x", "entities": [("A", "concept")], "relations": []}]
    graph.populate_document(_doc(doc_id="dA"), chunks)
    graph.populate_document(_doc(doc_id="dA"), chunks)  # same doc again
    assert graph.document_ids() == {"dA"}
    # Chunk merged, not duplicated.
    assert graph.counts()["chunks"] == 1


def test_resolve_chunks_relations_for_query(graph):
    chunks = [
        {
            "chunk_id": "c1",
            "text": "Alex founded Helix, an AI consultancy.",
            "entities": [("Alex", "person"), ("Helix", "project")],
            "relations": [("Alex", "founded", "Helix")],
        },
        {
            "chunk_id": "c2",
            "text": "Helix runs on LiveKit and a Vertel SIP gateway.",
            "entities": [("Helix", "project"), ("LiveKit", "tool"), ("Vertel", "organization")],
            "relations": [("Helix", "uses", "LiveKit")],
        },
    ]
    graph.populate_document(_doc(name="overview.md"), chunks)

    resolved = graph.resolve_query_entities(["helix"])
    assert any(r["label"] == "Helix" for r in resolved)

    keys = [r["key"] for r in resolved]
    rows = graph.chunks_for_entities(keys, limit=5)
    # Both chunks mention Helix → both come back.
    texts = [r["text"] for r in rows]
    assert any("founded Helix" in t for t in texts)
    assert any("LiveKit" in t for t in texts)

    rels = graph.relations_for_entities(keys, limit=10)
    pairs = [(h, t) for h, _, t in rels]
    assert ("Alex", "Helix") in pairs
    assert ("Helix", "LiveKit") in pairs


def test_resolve_partial_match_when_no_exact(graph):
    graph.populate_document(
        _doc(),
        [{"chunk_id": "c1", "text": "x", "entities": [("Northwind Labs", "organization")], "relations": []}],
    )
    resolved = graph.resolve_query_entities(["northwind"])  # substring, no exact key
    assert resolved and resolved[0]["label"] == "Northwind Labs"


def test_useless_entities_skipped(graph):
    chunks = [{"chunk_id": "c1", "text": "x", "entities": [("a", "concept"), ("!", "concept"), ("ok", "concept")], "relations": []}]
    s = graph.populate_document(_doc(), chunks)
    # "a" (1 char) and "!" (no alnum) are skipped; only "ok" survives.
    assert s["entities"] == 1
