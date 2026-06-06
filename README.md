# Personal Database

A fully-local, privacy-first AI assistant that answers questions from **your own corpus** — notes, journals, PDFs, books, papers. Combines semantic vector search with a knowledge graph built from your documents, and streams answers using any LLM provider you choose.

Everything runs on your machine. Your data never leaves.

---

## What it does

- **Ingest** any mix of Markdown, PDF, and text files into a local vector index (Qdrant embedded) and a knowledge graph (Kuzu embedded).
- **Ask** natural-language questions and get answers grounded in your documents, with source citations.
- **Hybrid retrieval**: every query fires a semantic vector search and a graph traversal simultaneously. Results are fused via Reciprocal Rank Fusion (RRF) before the LLM ever sees them.
- **Agentic planner**: before each retrieval, an LLM planner decides whether the question is specific enough to search on, or whether a clarifying question would produce a better answer.
- **Knowledge graph**: entities (people, projects, tools, concepts, decisions…) and relations are extracted automatically from every ingested document using GLiNER (NER) and REBEL (relation extraction) — no LLM needed for extraction.
- **Provider flexibility**: use a local Ollama model by default. Switch to Gemini, OpenAI, or Anthropic with a single env var, with automatic fallback to local Ollama on API outage.
- **Memory compaction**: long conversations are summarized automatically so context stays coherent without ballooning token counts.

---

## Architecture

```
Browser
  │
  ▼
FastAPI + Jinja2 web UI  (localhost:8000)
  │
  ├── POST /ingest/start  ──► background asyncio task (SSE progress)
  │                              │
  │                              ├── parse (pymupdf4llm / plain text)
  │                              ├── chunk (SentenceSplitter, 512t / 64 overlap)
  │                              ├── embed → Qdrant (nomic-embed-text via Ollama)
  │                              └── extract → Kuzu
  │                                    ├── GLiNER  → entities (person/project/tool/concept/…)
  │                                    └── REBEL   → relations (head─predicate→tail)
  │
  └── POST /chat/stream   ──► background asyncio task (SSE tokens)
                               │
                               ├── 1. Memory compaction (summarize old turns if budget exceeded)
                               ├── 2. Planner LLM call → clarify | search with refined query
                               ├── 3. Hybrid retrieval (parallel)
                               │     ├── vector_retrieve  → Qdrant top-k
                               │     └── graph_retrieve   → Kuzu entity match → chunk lookup
                               │                          → RRF fusion
                               ├── 4. LLM stream (primary provider → Ollama fallback)
                               └── 5. Persist message + citations to SQLite


Stores (all embedded, zero ops):
  Qdrant   ./data/qdrant/          vector index
  Kuzu     ./data/kuzu/personal.db knowledge graph
  SQLite   ./data/sessions.db      chat history
```

---

## Requirements

- **Python 3.11** (3.12 works; 3.13+ not yet supported by all ML deps)
- **[Ollama](https://ollama.com)** running locally — used for embeddings regardless of which LLM provider you choose
- **[uv](https://docs.astral.sh/uv/)** for dependency management
- 8 GB RAM minimum; 16 GB recommended for the default 20B model

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/<your-handle>/personal-database.git
cd personal-database

# 2. Install deps
uv sync

# 3. Pull the embedding model (required) and your preferred chat model
ollama pull nomic-embed-text
ollama pull gpt-oss:20b          # or any model — see LLM_MODEL in .env

# 4. Configure
cp .env.example .env
# Edit .env — at minimum set LLM_PROVIDER and any required API keys

# 5. Run
./run.sh
# → open http://localhost:8000
```

---

## Configuration

Copy `.env.example` to `.env` and edit. All values have sensible defaults.

### LLM Provider

Set `LLM_PROVIDER` to one of: `ollama`, `gemini`, `openai`, `anthropic`.

If the chosen provider fails at startup (missing key, network issue), the system falls back to local Ollama automatically.

**Embeddings always stay local** via Ollama regardless of which LLM provider is active. This keeps your corpus private and avoids per-token embedding costs.

---

#### Ollama (default — fully local)

```env
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=gpt-oss:20b
EMBED_MODEL=nomic-embed-text
```

`LLM_MODEL` accepts any model name you have pulled via `ollama pull`. Examples:

| Hardware | Suggested model |
|---|---|
| 8 GB RAM, low power | `llama3.2:3b`, `phi3:mini`, `qwen2.5:3b` |
| 16 GB RAM, mid-range | `llama3.1:8b`, `mistral:7b`, `gemma2:9b` |
| 32 GB+ RAM, high-end | `gpt-oss:20b`, `llama3.3:70b`, `qwq:32b` |

---

#### Gemini (Google — free tier available)

The free tier covers `gemini-2.5-flash` and `gemini-2.5-flash-lite`.

```env
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash-lite
GEMINI_API_KEY=your_key_here
```

Get a key at [aistudio.google.com](https://aistudio.google.com).

---

#### OpenAI

```env
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_key_here
```

Any model on your OpenAI account works — `gpt-4o`, `gpt-4o-mini`, `o3-mini`, etc.

---

#### Anthropic

```env
LLM_PROVIDER=anthropic
ANTHROPIC_MODEL=claude-3-5-haiku-20241022
ANTHROPIC_API_KEY=your_key_here
```

Any model on your Anthropic account works — `claude-opus-4-8`, `claude-sonnet-4-6`, `claude-haiku-4-5`, etc.

---

### Adding your own provider

The provider switch lives in `src/personal_db/stores.py → configure_llama_index()`. The pattern is:

```python
elif provider == "myprovider" and config.myprovider_api_key:
    try:
        from llama_index.llms.myprovider import MyProvider
        llm = MyProvider(model=config.myprovider_model, api_key=config.myprovider_api_key)
    except Exception as e:
        log.warning("MyProvider init failed (%s); falling back to Ollama", e)
        llm = None
```

Add the corresponding config fields to `config.py` and the LlamaIndex integration package to `pyproject.toml`. Any LlamaIndex-compatible LLM works.

---

### Full configuration reference

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | Active LLM provider |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API base URL |
| `LLM_MODEL` | `gpt-oss:20b` | Ollama model for chat |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | Gemini model name |
| `GEMINI_API_KEY` | _(empty)_ | Gemini API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `OPENAI_API_KEY` | _(empty)_ | OpenAI API key |
| `ANTHROPIC_MODEL` | `claude-3-5-haiku-20241022` | Anthropic model name |
| `ANTHROPIC_API_KEY` | _(empty)_ | Anthropic API key |
| `DATA_DIR` | `./data` | Root for all local stores |
| `QDRANT_PATH` | `./data/qdrant` | Qdrant embedded store path |
| `KUZU_PATH` | `./data/kuzu/personal.db` | Kuzu graph DB path |
| `RAW_DIR` | `./data/raw` | Uploaded document storage |
| `TOP_K_VECTOR` | `6` | Vector results per query |
| `TOP_K_GRAPH` | `10` | Graph entity hop limit |
| `RERANK_TOP_N` | `8` | Fused results sent to LLM |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `MIN_SIMILARITY_SCORE` | `0.45` | Vector score threshold |
| `MEMORY_TOKEN_BUDGET` | `4000` | Tokens before compaction triggers |
| `MEMORY_KEEP_RECENT_TURNS` | `2` | Verbatim turns kept after compaction |
| `MEMORY_SUMMARY_MAX_CHARS` | `2400` | Max characters in compressed summary |
| `KG_ENABLED` | `true` | Enable knowledge graph |
| `KG_RELATIONS` | `true` | Extract relations (REBEL) |
| `GLINER_MODEL` | `urchade/gliner_small-v2.1` | GLiNER model for NER |
| `REBEL_MODEL` | `Babelscape/rebel-large` | REBEL model for relations |
| `KG_ENTITY_THRESHOLD` | `0.5` | GLiNER confidence cutoff |

---

## How hybrid retrieval works

```
User question
     │
     ▼
  Planner LLM
  ┌──────────┐
  │ clarify? │──yes──► ask user, wait for reply
  │ search?  │
  └────┬─────┘
       │ refined query
       ▼
 ┌─────────────────────────────────────────────┐
 │              Parallel retrieval             │
 │                                             │
 │  Vector path            Graph path          │
 │  ──────────             ──────────          │
 │  embed query            GLiNER on query     │
 │  → Qdrant top-k         → entity lookup     │
 │    by cosine sim          in Kuzu           │
 │                         → fetch linked      │
 │                           chunks + rels     │
 └───────────────┬─────────────────────────────┘
                 │
                 ▼
         RRF fusion (k=60)
         (score = Σ 1/(k + rank))
                 │
                 ▼
         top-N chunks + relations block
                 │
                 ▼
           LLM synthesis
           (cites [N] inline)
```

Chunks that appear in both the vector and graph results are marked `origin: "both"` in the UI (shown with a `kg+` badge on the citation).

---

## Knowledge graph

On every ingest, two non-LLM models run over each chunk:

- **GLiNER** (`urchade/gliner_small-v2.1`, ~150 MB) — zero-shot NER with a personal ontology: `person`, `project`, `tool`, `concept`, `decision`, `place`, `event`, `technology`, `organization`
- **REBEL** (`Babelscape/rebel-large`, ~1.5 GB) — seq2seq relation extraction producing `(head, predicate, tail)` triples

Extracted data lands in Kuzu with the schema:

```
(Document) -[PART_OF]→ (Chunk)
(Chunk)    -[MENTIONS]→ (Entity)
(Entity)   -[RELATES_TO {predicate}]→ (Entity)
```

You can explore the graph at `/graph` in the web UI — search entities, view neighborhoods, and see counts.

REBEL is large (~1.5 GB). To skip relation extraction and only run entity extraction, set `KG_RELATIONS=false`. To disable the graph entirely, set `KG_ENABLED=false`.

---

## Agentic planner

Every chat turn runs a fast LLM call before retrieval to decide:

- **search** — the question is specific enough; rewrites it into a short keyword-rich query stripped of pronouns and filler, then fires retrieval.
- **clarify** — the question is genuinely ambiguous (a pronoun without antecedent, a topic with two equally-valid interpretations). Returns one follow-up question; no retrieval runs until the user replies.

A hard anti-loop guard prevents the planner from asking the same clarification twice: if the prior assistant turn was a clarification (detected via a `citations=NULL` marker in the DB), the planner is forced to treat the user's reply as the answer and proceed to search.

---

## Memory compaction

Conversation history is stored verbatim in SQLite. Once the unsummarized portion exceeds `MEMORY_TOKEN_BUDGET` tokens, the system:

1. Groups messages into complete user+assistant turn pairs (never splits mid-turn).
2. Keeps the most recent `MEMORY_KEEP_RECENT_TURNS` turns verbatim (coherence floor).
3. Summarizes everything older into a single compressed block using the same LLM.
4. Replaces the raw history with `[summary block] + [recent verbatim turns]`.

This keeps the effective context window small without losing continuity.

---

## Running tests

```bash
uv run pytest
```

88 tests, covering: sessions, memory compaction, ingest, KG extraction, hybrid retrieval, planner (including anti-loop cases), LLM runtime fallback, and the web API.

---

## Project layout

```
src/personal_db/
├── config.py          # Pydantic config loaded from .env
├── stores.py          # Qdrant + Kuzu init, LlamaIndex Settings wiring
├── sessions.py        # SQLite chat session store
├── memory.py          # Turn-aware memory compaction
├── ingest.py          # Parse → chunk → embed → Qdrant
├── ingest_jobs.py     # Background ingest with SSE progress
├── kg.py              # Kuzu graph layer (schema, populate, query)
├── extract/
│   ├── entities.py    # GLiNER wrapper
│   ├── relations.py   # REBEL wrapper
│   └── _text.py       # Shared text utilities
├── retrieve.py        # Hybrid retrieval + RRF fusion
├── planner.py         # Agentic clarify-or-search planner
├── llm_runtime.py     # safe_chat / safe_stream_chat with fallback
├── chat.py            # Full chat pipeline (orchestrates everything)
└── web/
    ├── app.py         # FastAPI routes
    └── templates/     # Jinja2 templates (chat, graph, ingest, library)
```

---

## Roadmap

- [ ] Re-ranking pass (bge-reranker-v2-m3) between fusion and LLM
- [ ] BM25 sparse vectors alongside dense (Qdrant sparse support)
- [ ] Folder watcher for auto-ingest on file drop
- [ ] LLM-enhanced KG extraction for flagged documents (journals, decision logs)
- [ ] Graph community detection (HDBSCAN over chunk embeddings)
- [ ] Export/import corpus snapshots

---

## License

MIT — see [LICENSE](LICENSE).
