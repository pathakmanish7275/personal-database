"""Configuration loaded from .env."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# override=True: .env is the project's source of truth. Without it, stray
# exported vars in the desktop session (e.g. a stale LLM_MODEL) silently
# shadow the project config.
load_dotenv(PROJECT_ROOT / ".env", override=True)


def _opt_bool(raw: str | None) -> Optional[bool]:
    """Parse a tri-state flag: true / false / unset (omit the parameter)."""
    if raw is None:
        return None
    v = raw.strip().lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    return None


def _path(env_key: str, default: str) -> Path:
    raw = os.getenv(env_key, default)
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


class Config(BaseModel):
    # Everything runs locally: chat on `llm_host` (FLM/NPU or Ollama),
    # embeddings always on `ollama_host`. There is no cloud provider.
    ollama_host: str = Field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    # Host for the chat LLM only (Ollama-compatible API, e.g. FLM on its own
    # port). Defaults to ollama_host; embeddings always use ollama_host.
    llm_host: str = Field(default_factory=lambda: os.getenv("LLM_HOST", os.getenv("OLLAMA_HOST", "http://localhost:11434")))
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen3.5:4b"))
    # Where the Ollama fallback lives when the primary is a remote/other local
    # server (e.g. FLM). Model must exist on that host.
    fallback_llm_host: str = Field(default_factory=lambda: os.getenv("FALLBACK_LLM_HOST", os.getenv("OLLAMA_HOST", "http://localhost:11434")))
    fallback_llm_model: str = Field(default_factory=lambda: os.getenv("FALLBACK_LLM_MODEL", "qwen3.5:4b"))
    embed_model: str = Field(default_factory=lambda: os.getenv("EMBED_MODEL", "nomic-embed-text"))
    # Controls thinking for Ollama models that support it (e.g. qwen3.5).
    # "auto" → False for ≥4b (disable thinking), None for :2b (omit the flag,
    # because 2b stops using tools when thinking is fully off).
    # Explicitly set LLM_THINKING=true|false to override.
    llm_thinking: Optional[bool] = Field(default=None)

    # ── agent loop ─────────────────────────────────────────────────────────
    # How many times the model may call search_kb in one turn before we make it
    # answer with what it has. 2 allows one retry with better terms; more than
    # that mostly buys repeated queries.
    max_tool_turns: int = Field(
        default_factory=lambda: int(os.getenv("MAX_TOOL_TURNS", "2"))
    )
    # Reasoning is requested per role, because the two turns want opposite
    # things: choosing a good search query benefits from thinking, while
    # synthesising an answer that is already grounded in retrieved passages
    # mostly spends decode time on it. On FLM's /v1 endpoint `think` puts
    # reasoning in its own `reasoning_content` field, so it never pollutes the
    # answer either way.
    think_tool_turns: Optional[bool] = Field(
        default_factory=lambda: _opt_bool(os.getenv("THINK_TOOL_TURNS", "true"))
    )
    think_answer: Optional[bool] = Field(
        default_factory=lambda: _opt_bool(os.getenv("THINK_ANSWER", "false"))
    )
    # Hard ceiling on a tool turn's output (reasoning + the tool call together).
    # A thinking turn can otherwise spiral to the context cap: measured at 4096
    # tokens / 245s on one vague question, entirely inside the reasoning
    # channel. Ample for reasoning plus a short tool call.
    tool_turn_max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("TOOL_TURN_MAX_TOKENS", "768"))
    )
    # 0 = uncapped; the answer itself needs room to be complete.
    answer_max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("ANSWER_MAX_TOKENS", "0"))
    )

    # ── voice ──────────────────────────────────────────────────────────────
    voice_enabled: bool = Field(
        default_factory=lambda: os.getenv("VOICE_ENABLED", "true").lower()
        in ("true", "1", "yes")
    )
    # STT and TTS run as separate local HTTP servers speaking the OpenAI audio
    # API, so either can be swapped (including for a cloud provider) by URL.
    stt_base_url: str = Field(
        default_factory=lambda: os.getenv("STT_BASE_URL", "http://localhost:8123/v1")
    )
    tts_base_url: str = Field(
        default_factory=lambda: os.getenv("TTS_BASE_URL", "http://localhost:8880/v1")
    )
    stt_model: str = Field(default_factory=lambda: os.getenv("STT_MODEL", "whisper-1"))
    tts_voice: str = Field(default_factory=lambda: os.getenv("TTS_VOICE", "af_heart"))
    # Reasoning off by default for speech: it is the largest latency cost and
    # helps least when the answer must be two or three spoken sentences.
    voice_think: Optional[bool] = Field(
        default_factory=lambda: _opt_bool(os.getenv("VOICE_THINK", "false"))
    )
    # Spoken answers must stay short; this is a backstop for when the prompt
    # is ignored, not the primary control.
    voice_answer_max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("VOICE_ANSWER_MAX_TOKENS", "300"))
    )
    voice_greeting: str = Field(
        default_factory=lambda: os.getenv(
            "VOICE_GREETING", "Hey. What would you like to know?"
        )
    )
    # Used when the call is placed on a conversation that already has turns —
    # the agent carries that history, so the greeting should say so.
    voice_resume_greeting: str = Field(
        default_factory=lambda: os.getenv(
            "VOICE_RESUME_GREETING",
            "Picking up where we left off. What would you like to know?",
        )
    )
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

config = Config()
