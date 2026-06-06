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
        base_url=config.ollama_host,
        request_timeout=600.0,
        context_window=32768,
    )


def configure_llama_index() -> None:
    """Point LlamaIndex Settings at the configured LLM provider.

    LLM is swappable (ollama | gemini). When provider=gemini fails *at
    construction* (bad key, no network at startup), we silently fall through
    to Ollama so the assistant still works. Run-time failures on a successfully
    constructed Gemini are handled per-call by llm_runtime.safe_*.

    Embeddings always stay local via Ollama — cheap on CPU and keeps the
    corpus private."""
    provider = (config.llm_provider or "ollama").lower()
    llm = None
    if provider == "gemini" and config.gemini_api_key:
        try:
            from llama_index.llms.google_genai import GoogleGenAI

            llm = GoogleGenAI(
                model=config.gemini_model,
                api_key=config.gemini_api_key,
                context_window=131072,
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "Gemini LLM init failed (%s); falling back to Ollama for this session", e
            )
            llm = None
    elif provider == "gemini":
        log.warning("LLM_PROVIDER=gemini but GEMINI_API_KEY is empty; using Ollama")
    elif provider == "openai" and config.openai_api_key:
        try:
            from llama_index.llms.openai import OpenAI

            llm = OpenAI(model=config.openai_model, api_key=config.openai_api_key)
        except Exception as e:  # noqa: BLE001
            log.warning("OpenAI LLM init failed (%s); falling back to Ollama for this session", e)
            llm = None
    elif provider == "openai":
        log.warning("LLM_PROVIDER=openai but OPENAI_API_KEY is empty; using Ollama")
    elif provider == "anthropic" and config.anthropic_api_key:
        try:
            from llama_index.llms.anthropic import Anthropic

            llm = Anthropic(model=config.anthropic_model, api_key=config.anthropic_api_key)
        except Exception as e:  # noqa: BLE001
            log.warning(
                "Anthropic LLM init failed (%s); falling back to Ollama for this session", e
            )
            llm = None
    elif provider == "anthropic":
        log.warning("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty; using Ollama")

    Settings.llm = llm or _ollama_llm()
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
