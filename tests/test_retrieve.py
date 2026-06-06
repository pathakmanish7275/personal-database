"""Hybrid retrieval — RRF fusion + graph_retrieve path.

The vector half is exercised by the existing API/chat tests against a real
Qdrant + MockEmbedding; here we focus on graph_retrieve (with the entity
extractor stubbed) and the fusion math."""

from __future__ import annotations


def _rc(text, doc_id="d", score=None, hits=0, origin="semantic"):
    from personal_db.retrieve import RetrievedChunk
    return RetrievedChunk(text=text, name="n.md", path="/x/n.md", doc_id=doc_id,
                          vector_score=score, graph_hits=hits, origin=origin)


def test_rrf_fuse_boosts_chunks_in_both_lists():
    from personal_db.retrieve import _rrf_fuse
    vec = [_rc("only-vec", doc_id="v"), _rc("shared", doc_id="s", score=0.9)]
    gph = [_rc("shared", doc_id="s", hits=3, origin="graph"), _rc("only-graph", doc_id="g", hits=2, origin="graph")]
    fused = _rrf_fuse(vec, gph, top_n=10)
    # "shared" appears in both at rank 2 and rank 1 → highest RRF score.
    assert fused[0].text == "shared"
    assert fused[0].origin == "both"
    assert fused[0].graph_hits == 3
    # All three unique chunks present.
    assert {c.text for c in fused} == {"shared", "only-vec", "only-graph"}


def test_rrf_fuse_top_n_caps_output():
    from personal_db.retrieve import _rrf_fuse
    vec = [_rc(f"v{i}", doc_id=f"v{i}") for i in range(8)]
    gph = [_rc(f"g{i}", doc_id=f"g{i}", origin="graph") for i in range(8)]
    fused = _rrf_fuse(vec, gph, top_n=5)
    assert len(fused) == 5


def test_graph_retrieve_returns_chunks_and_relations(isolated_paths, monkeypatch):
    from personal_db import config as cfg_mod
    from personal_db import retrieve
    from personal_db.extract import entities as ents_mod
    from personal_db.stores import init_stores

    monkeypatch.setattr(cfg_mod.config, "kg_enabled", True)
    monkeypatch.setattr(ents_mod, "extract_entities", lambda text, threshold=None: [("Acme", "project")])

    stores = init_stores()
    stores.graph.populate_document(
        {"doc_id": "d1", "name": "helix.md", "path": "/x/helix.md", "source": "md", "ingested_at": ""},
        [
            {
                "chunk_id": "c1",
                "text": "Acme is the AI consultancy Alex founded.",
                "entities": [("Acme", "project"), ("Alex", "person")],
                "relations": [("Alex", "founded", "Acme")],
            }
        ],
    )

    chunks, rels, q_ents = retrieve.graph_retrieve("what is helix?", stores)
    assert q_ents == ["Acme"]
    assert len(chunks) == 1
    assert chunks[0].origin == "graph"
    assert "Acme is the AI consultancy" in chunks[0].text
    assert chunks[0].graph_hits >= 1
    assert ("Alex", "founded", "Acme") in rels


def test_graph_retrieve_disabled_returns_empty(isolated_paths, monkeypatch):
    from personal_db import config as cfg_mod
    from personal_db import retrieve
    from personal_db.stores import init_stores

    monkeypatch.setattr(cfg_mod.config, "kg_enabled", False)
    stores = init_stores()
    chunks, rels, q_ents = retrieve.graph_retrieve("anything", stores)
    assert (chunks, rels, q_ents) == ([], [], [])


def test_graph_retrieve_unknown_entity_returns_labels_only(isolated_paths, monkeypatch):
    from personal_db import config as cfg_mod
    from personal_db import retrieve
    from personal_db.extract import entities as ents_mod
    from personal_db.stores import init_stores

    monkeypatch.setattr(cfg_mod.config, "kg_enabled", True)
    monkeypatch.setattr(ents_mod, "extract_entities", lambda text, threshold=None: [("Atlantis", "place")])
    stores = init_stores()
    chunks, rels, q_ents = retrieve.graph_retrieve("where is atlantis?", stores)
    assert chunks == []
    assert rels == []
    assert q_ents == ["Atlantis"]
