"""Configuration loaded from .env."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _path(env_key: str, default: str) -> Path:
    raw = os.getenv(env_key, default)
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


class Config(BaseModel):
    # LLM provider: "ollama" (local) or "gemini" / "openai" / "anthropic" (cloud).
    # Embeddings stay local via Ollama regardless of provider.
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama").lower())
    ollama_host: str = Field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen3.5:4b"))
    embed_model: str = Field(default_factory=lambda: os.getenv("EMBED_MODEL", "nomic-embed-text"))
    # Controls thinking for Ollama models that support it (e.g. qwen3.5).
    # "auto" → False for ≥4b (disable thinking), None for :2b (omit the flag,
    # because 2b stops using tools when thinking is fully off).
    # Explicitly set LLM_THINKING=true|false to override.
    llm_thinking: Optional[bool] = Field(default=None)

    @model_validator(mode="after")
    def _resolve_thinking(self) -> "Config":
        raw = os.getenv("LLM_THINKING", "auto").lower()
        if raw == "auto":
            # Only touch the think param for models that actually support it.
            # For everything else (gpt-oss, llama, mistral, gemma…) leave None
            # so Ollama never sees the flag.
            _thinking_families = ("qwen3", "qwq", "deepseek-r1")
            is_thinking_model = any(f in self.llm_model.lower() for f in _thinking_families)
            if is_thinking_model:
                # 2b variant loses tool-use when thinking is fully off — omit the flag.
                self.llm_thinking = None if ":2b" in self.llm_model else False
        elif raw in ("true", "1", "yes"):
            self.llm_thinking = True
        elif raw in ("false", "0", "no"):
            self.llm_thinking = False
        # else: leave as None (omit the param entirely)
        return self

    # Gemini (free tier covers gemini-2.5-flash and gemini-2.5-flash-lite).
    gemini_model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"))
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    # OpenAI (any model on your account; gpt-4o-mini is the cheap/fast tier).
    openai_model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # Anthropic (claude-3-5-haiku-20241022 is the fast/cheap tier).
    anthropic_model: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"))
    anthropic_api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    data_dir: Path = Field(default_factory=lambda: _path("DATA_DIR", "./data"))
    qdrant_path: Path = Field(default_factory=lambda: _path("QDRANT_PATH", "./data/qdrant"))
    kuzu_path: Path = Field(default_factory=lambda: _path("KUZU_PATH", "./data/kuzu/personal.db"))
    raw_dir: Path = Field(default_factory=lambda: _path("RAW_DIR", "./data/raw"))

    top_k_vector: int = Field(default_factory=lambda: int(os.getenv("TOP_K_VECTOR", "6")))
    min_similarity_score: float = Field(default_factory=lambda: float(os.getenv("MIN_SIMILARITY_SCORE", "0.45")))
    top_k_graph: int = Field(default_factory=lambda: int(os.getenv("TOP_K_GRAPH", "10")))
    rerank_top_n: int = Field(default_factory=lambda: int(os.getenv("RERANK_TOP_N", "8")))
    chunk_size: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "512")))
    chunk_overlap: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "64")))

    # Knowledge graph (non-LLM extraction → Kuzu).
    kg_enabled: bool = Field(default_factory=lambda: os.getenv("KG_ENABLED", "true").lower() != "false")
    kg_relations: bool = Field(default_factory=lambda: os.getenv("KG_RELATIONS", "true").lower() != "false")
    gliner_model: str = Field(default_factory=lambda: os.getenv("GLINER_MODEL", "urchade/gliner_small-v2.1"))
    rebel_model: str = Field(default_factory=lambda: os.getenv("REBEL_MODEL", "Babelscape/rebel-large"))
    kg_entity_threshold: float = Field(default_factory=lambda: float(os.getenv("KG_ENTITY_THRESHOLD", "0.5")))

    # Trigger: compact once the un-summarized history exceeds this many tokens.
    memory_token_budget: int = Field(default_factory=lambda: int(os.getenv("MEMORY_TOKEN_BUDGET", "4000")))
    # Coherence floor: how many of the most recent *whole turns* to keep verbatim.
    # Kept small on purpose — everything older folds into the summary.
    memory_keep_recent_turns: int = Field(default_factory=lambda: int(os.getenv("MEMORY_KEEP_RECENT_TURNS", "2")))
    memory_summary_max_chars: int = Field(default_factory=lambda: int(os.getenv("MEMORY_SUMMARY_MAX_CHARS", "2400")))

    qdrant_collection: str = "personal_db"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.kuzu_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)


config = Config()
