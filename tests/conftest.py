"""Shared test fixtures.

Strategy:
- Every test gets an isolated data directory (Qdrant, Kuzu, SQLite sessions).
- The Ollama-backed LLM and embedding are stubbed so tests run without a
  running Ollama process.
- A `client` fixture provides a FastAPI TestClient for API tests.
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock

import pytest


# ─── Isolated config ──────────────────────────────────────────────────────


@pytest.fixture
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point all storage paths at a tmp directory."""
    from personal_db import config as cfg_mod

    data = tmp_path / "data"
    data.mkdir()
    (data / "raw").mkdir()
    (data / "qdrant").mkdir()
    (data / "kuzu").mkdir()

    monkeypatch.setattr(cfg_mod.config, "data_dir", data)
    monkeypatch.setattr(cfg_mod.config, "qdrant_path", data / "qdrant")
    monkeypatch.setattr(cfg_mod.config, "kuzu_path", data / "kuzu" / "p.db")
    monkeypatch.setattr(cfg_mod.config, "raw_dir", data / "raw")
    return data


# ─── LLM and embedding stubs ──────────────────────────────────────────────


class _FakeDelta:
    def __init__(self, text: str) -> None:
        self.delta = text
        self.message = MagicMock(content=text)


class FakeLLM:
    """Stand-in for LlamaIndex's Ollama LLM. Records calls and yields canned tokens."""

    def __init__(self) -> None:
        self.chat_calls: list = []
        self.stream_calls: list = []
        self.summary = "MOCK_SUMMARY: condensed prior turns about projects, decisions and preferences."
        self.tokens = ["Mock", " answer", " grounded", " in", " context", " [1]", "."]

    def chat(self, messages):
        self.chat_calls.append(messages)
        return MagicMock(message=MagicMock(content=self.summary))

    def stream_chat(self, messages):
        self.stream_calls.append(messages)
        for t in self.tokens:
            yield _FakeDelta(t)


class FakeEmbedding:
    """Deterministic 8-dim embeddings — enough for Qdrant cosine sim."""

    def __init__(self) -> None:
        self.calls: list = []

    def _vec(self, text: str) -> list[float]:
        # Hash-based deterministic vector so equal text → equal vector.
        import hashlib
        h = hashlib.sha1(text.encode()).digest()
        return [(b - 128) / 128.0 for b in h[:8]]

    def get_text_embedding(self, text: str) -> list[float]:
        self.calls.append(("one", text))
        return self._vec(text)

    def get_text_embedding_batch(self, texts, **kw) -> list[list[float]]:
        self.calls.append(("batch", len(texts)))
        return [self._vec(t) for t in texts]

    def get_agg_embedding_from_queries(self, queries) -> list[float]:
        return self._vec(" ".join(queries))

    # LlamaIndex sometimes probes these:
    @property
    def model_name(self) -> str:
        return "fake-embed"


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_embed() -> FakeEmbedding:
    return FakeEmbedding()


@pytest.fixture
def patch_settings(fake_llm: FakeLLM, fake_embed: FakeEmbedding, monkeypatch: pytest.MonkeyPatch):
    """Swap LlamaIndex global Settings to use our fakes; also short-circuit
    `configure_llama_index` so it doesn't reach for Ollama."""
    from llama_index.core import Settings
    monkeypatch.setattr(Settings, "_llm", fake_llm, raising=False)
    monkeypatch.setattr(Settings, "_embed_model", fake_embed, raising=False)

    # Re-import to grab the live module references for patching.
    from personal_db import stores as stores_mod
    monkeypatch.setattr(stores_mod, "configure_llama_index", lambda: None)
    yield fake_llm, fake_embed


# ─── Per-test event loop hygiene ──────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_jobs_state():
    """Drop any retained jobs/locks between tests so module state doesn't leak."""
    from personal_db import jobs as j
    from personal_db import ingest_jobs as ij
    j._jobs.clear()
    j._run_locks.clear()
    ij._current = None
    yield
    j._jobs.clear()
    j._run_locks.clear()
    ij._current = None


# ─── FastAPI test client ──────────────────────────────────────────────────


@pytest.fixture
def client(isolated_paths, patch_settings, monkeypatch):
    """A TestClient with all storage isolated and the LLM stubbed.

    Replaces the run_generation runner with a fast fake that emits a couple
    of canned events so the API surface can be exercised without a real LLM."""
    # Reset module-cached stores so a fresh per-test directory is used.
    from personal_db.web import app as web_app

    web_app._stores = None  # force re-init under the isolated paths

    # Stub run_generation so tests don't depend on the full LLM/retrieval path.
    from personal_db import chat as chat_mod
    from personal_db import jobs as jobs_mod

    async def fake_run_generation(job: jobs_mod.GenJob, stores) -> None:
        from personal_db import sessions as sess_store
        await asyncio.to_thread(sess_store.add_message, job.session_id, "user", job.user_text)
        s = await asyncio.to_thread(sess_store.get_session, job.session_id)
        if s and s.title == "New chat":
            await asyncio.to_thread(sess_store.rename_session, job.session_id, job.user_text[:60])
        await jobs_mod.emit(job, {"type": "status", "data": {"phase": "thinking", "text": "thinking"}})
        await jobs_mod.emit(job, {"type": "citations", "data": []})
        await jobs_mod.emit(job, {"type": "token", "data": "Hello"})
        await jobs_mod.emit(job, {"type": "token", "data": " world."})
        await asyncio.to_thread(sess_store.add_message, job.session_id, "assistant", "Hello world.", "[]")

    monkeypatch.setattr(chat_mod, "run_generation", fake_run_generation)

    # Stub get_stores to avoid opening real Qdrant/Kuzu where possible. The
    # isolated paths still keep things scoped per-test if it does init.
    from starlette.testclient import TestClient
    with TestClient(web_app.app) as c:
        yield c
