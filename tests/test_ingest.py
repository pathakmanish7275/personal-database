"""Ingestion helpers (parsing, dedup, file walking)."""

from __future__ import annotations

from pathlib import Path


def test_iter_files_walks_supported_only(tmp_path):
    from personal_db.ingest import iter_files
    (tmp_path / "a.md").write_text("# A")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4\n%fake")
    (tmp_path / "c.txt").write_text("hello")
    (tmp_path / "d.docx").write_bytes(b"unsupported")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.md").write_text("# nested")

    files = sorted(p.name for p in iter_files(tmp_path))
    assert files == ["a.md", "b.pdf", "c.txt", "nested.md"]


def test_iter_files_single_file(tmp_path):
    from personal_db.ingest import iter_files
    p = tmp_path / "x.md"
    p.write_text("hi")
    assert iter_files(p) == [p]
    bad = tmp_path / "x.docx"
    bad.write_text("nope")
    assert iter_files(bad) == []


def test_doc_id_is_stable(tmp_path):
    from personal_db.ingest import _doc_id_for
    p = tmp_path / "n.md"
    p.write_text("body")
    a = _doc_id_for(p, "body")
    b = _doc_id_for(p, "body")
    assert a == b
    # different text → different id
    c = _doc_id_for(p, "other")
    assert a != c


def test_read_md_round_trip(tmp_path):
    from personal_db.ingest import _read
    p = tmp_path / "x.md"
    p.write_text("# Hello\n\nbody")
    assert "Hello" in _read(p)


def test_read_unsupported_raises(tmp_path):
    from personal_db.ingest import _read
    p = tmp_path / "x.docx"
    p.write_bytes(b"data")
    import pytest
    with pytest.raises(ValueError):
        _read(p)
