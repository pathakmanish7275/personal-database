"""Initialize Qdrant (embedded), Kuzu, and LlamaIndex Settings."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import kuzu

log = logging.getLogger(__name__)
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from .config import config
from .kg import Graph


@dataclass
class Stores:
    qdrant_client: QdrantClient
    vector_store: QdrantVectorStore
    kuzu_db: kuzu.Database
    graph: Graph


def _ollama_llm() -> Ollama:
    return Ollama(
        model=config.llm_model,
        base_url=config.llm_host,
        request_timeout=600.0,
        context_window=32768,
        thinking=config.llm_thinking,
    )


def configure_llama_index() -> None:
    """Point LlamaIndex Settings at the local models.

    Chat goes to `llm_host` — FLM on the NPU when configured, otherwise the
    local Ollama. Embeddings always go to `ollama_host`: FLM serves one model
    at a time, and a query embedding is milliseconds on CPU anyway.

    There is no cloud provider branch. Everything runs on this machine, which
    is the point of the project; a remote fallback would quietly undo it."""
    Settings.llm = _ollama_llm()
    Settings.embed_model = OllamaEmbedding(
        model_name=config.embed_model,
        base_url=config.ollama_host,
    )


def init_stores() -> Stores:
    """Create Qdrant (embedded) + Kuzu + LlamaIndex store adapters."""
    config.ensure_dirs()

    qclient = QdrantClient(path=str(config.qdrant_path))
    vstore = QdrantVectorStore(
        collection_name=config.qdrant_collection,
        client=qclient,
    )

    kdb = kuzu.Database(str(config.kuzu_path))
    graph = Graph(kdb)

    return Stores(
        qdrant_client=qclient,
        vector_store=vstore,
        kuzu_db=kdb,
        graph=graph,
    )
