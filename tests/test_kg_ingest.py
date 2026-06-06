"""Graph population through ingestion + the /graph routes.

The extractor is stubbed so these run without GLiNER/REBEL/torch — we only
verify the wiring: ingestion writes to Kuzu, and the routes render/guard."""

from __future__ import annotations


def _fake_extract(text):
    return (
        [("Alex", "person"), ("Acme", "project")],
        [("Alex", "builds", "Acme")],
    )


def _use_mock_embedding():
    """Real BaseEmbedding so VectorStoreIndex builds without Ollama."""
    from llama_index.core import Settings
    from llama_index.core.embeddings import MockEmbedding

    Settings.embed_model = MockEmbedding(embed_dim=8)


def test_ingest_one_populates_graph(isolated_paths, patch_settings, monkeypatch):
    from personal_db import extract as extract_mod
    from personal_db import ingest as ing
    from personal_db.stores import init_stores

    monkeypatch.setattr(extract_mod, "extract_for_chunk", _fake_extract)
    _use_mock_embedding()

    stores = init_stores()
    f = isolated_paths / "raw" / "note.md"
    f.write_text("# Note\n\nAlex builds Acme, a project he cares about.")

    res = ing.ingest_one(f, stores)
    assert res["status"] == "ingested"
    assert res["entities"] >= 2
    assert res["relations"] >= 1

    counts = stores.graph.counts()
    assert counts["documents"] == 1
    assert counts["entities"] >= 2
    assert counts["relations"] >= 1

    detail = stores.graph.entity_detail("Acme")
    assert detail is not None
    assert any(r["source"] == "Alex" for r in detail["in_relations"])


def test_extract_file_to_graph_no_reembed(isolated_paths, patch_settings, monkeypatch):
    from personal_db import extract as extract_mod
    from personal_db import ingest as ing
    from personal_db.stores import init_stores

    monkeypatch.setattr(extract_mod, "extract_for_chunk", _fake_extract)
    stores = init_stores()
    f = isolated_paths / "raw" / "doc.md"
    f.write_text("Alex builds Acme.")

    res = ing.extract_file_to_graph(f, stores)
    assert res["status"] == "ingested"
    assert stores.graph.counts()["entities"] >= 2
    # Backfill targets should now exclude this doc (it's in the graph).
    assert ing.graph_backfill_targets(stores) == []


def test_kg_disabled_skips_graph(isolated_paths, patch_settings, monkeypatch):
    from personal_db import config as cfg_mod
    from personal_db import extract as extract_mod
    from personal_db import ingest as ing
    from personal_db.stores import init_stores

    monkeypatch.setattr(cfg_mod.config, "kg_enabled", False)
    called = {"n": 0}

    def spy(text):
        called["n"] += 1
        return ([], [])

    monkeypatch.setattr(extract_mod, "extract_for_chunk", spy)
    _use_mock_embedding()
    stores = init_stores()
    f = isolated_paths / "raw" / "n.md"
    f.write_text("Alex builds Acme.")
    res = ing.ingest_one(f, stores)
    assert res["status"] == "ingested"
    assert "entities" not in res
    assert called["n"] == 0
    assert stores.graph.counts()["entities"] == 0


def test_ingest_one_skips_known_doc_id(isolated_paths, patch_settings, monkeypatch):
    from personal_db import extract as extract_mod
    from personal_db import ingest as ing
    from personal_db.stores import init_stores

    monkeypatch.setattr(extract_mod, "extract_for_chunk", _fake_extract)
    _use_mock_embedding()
    stores = init_stores()
    f = isolated_paths / "raw" / "n.md"
    f.write_text("Hello world content")

    first = ing.ingest_one(f, stores)
    assert first["status"] == "ingested"

    # Re-ingest with an explicit known set → must skip without re-embedding.
    again = ing.ingest_one(f, stores, known_doc_ids={first["doc_id"]})
    assert again["status"] == "skipped"
    assert again["doc_id"] == first["doc_id"]

    # And without passing the set: ingest_one scans the vector store itself.
    auto = ing.ingest_one(f, stores)
    assert auto["status"] == "skipped"


def test_ingest_one_graph_only_when_vectors_present_but_graph_empty(
    isolated_paths, patch_settings, monkeypatch
):
    from personal_db import config as cfg_mod
    from personal_db import extract as extract_mod
    from personal_db import ingest as ing
    from personal_db.stores import init_stores

    monkeypatch.setattr(extract_mod, "extract_for_chunk", _fake_extract)
    _use_mock_embedding()
    stores = init_stores()
    f = isolated_paths / "raw" / "g.md"
    f.write_text("Alex builds Acme.")

    # First pass: KG disabled, so only the vector store gets populated.
    monkeypatch.setattr(cfg_mod.config, "kg_enabled", False)
    first = ing.ingest_one(f, stores)
    assert first["status"] == "ingested"
    assert stores.graph.counts()["documents"] == 0

    # Second pass: KG enabled. The gate detects vectors exist + graph missing,
    # so it does a graph-only pass — no re-embedding.
    monkeypatch.setattr(cfg_mod.config, "kg_enabled", True)
    second = ing.ingest_one(f, stores)
    assert second["status"] == "graph_only"
    assert stores.graph.counts()["documents"] == 1
    assert second["entities"] >= 2


def test_graph_page_renders(client):
    r = client.get("/graph")
    assert r.status_code == 200
    assert "Knowledge graph" in r.text


def test_graph_build_nothing_to_do_returns_400(client):
    r = client.post("/graph/build")
    assert r.status_code == 400
