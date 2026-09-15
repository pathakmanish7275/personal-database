# Personal Database

A fully-local, privacy-first AI assistant that answers questions from **your own corpus** — notes, journals, PDFs, books, papers. Combines semantic vector search with a knowledge graph built from your documents, and answers by voice or text — all on local models.

Everything runs on your machine. Your data never leaves.

---

## What it does

- **Ingest** any mix of Markdown, PDF, and text files into a local vector index (Qdrant embedded) and a knowledge graph (Kuzu embedded).
- **Ask** natural-language questions and get answers grounded in your documents, with source citations.
- **Hybrid retrieval**: every query fires a semantic vector search and a graph traversal simultaneously. Results are fused via Reciprocal Rank Fusion (RRF) before the LLM ever sees them.
- **Agentic retrieval**: the knowledge base is exposed to the model as a *tool*, not a fixed pipeline stage. The model decides whether to search at all, writes its own query, may search again with better terms, or asks a clarifying question — so greetings and general-knowledge questions skip retrieval entirely instead of paying for it.
- **Knowledge graph**: entities (people, projects, tools, concepts, decisions…) and relations are extracted automatically from every ingested document using GLiNER (NER) and REBEL (relation extraction) — no LLM needed for extraction.
- **Runs entirely on your machine**: chat on an AMD NPU via [FastFlowLM](https://github.com/ROCm/FastFlowLM) when available, otherwise Ollama; embeddings, speech-to-text and text-to-speech all local. No API keys, no cloud provider, nothing leaves the device.
- **Voice and text, one assistant**: press Call on the chat page and talk. A spoken turn runs the *same* agent, searches the *same* knowledge base, and lands in the *same* conversation thread — so you can ask by voice and follow up by typing. Speech-to-text and text-to-speech are local; nothing leaves the machine.
- **Memory compaction**: long conversations are summarized automatically so context stays coherent without ballooning token counts.

---

## Architecture

```
Browser
  │
  ▼
FastAPI + Jinja2 web UI  (localhost:8765)
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
  └── POST /chat/{sid}/ask ──► background asyncio task; GET /chat/{sid}/stream (SSE)
                               │
                               ├── 1. Memory compaction (summarize old turns if budget exceeded)
                               └── 2. Agent loop — the model drives retrieval as a tool
                                     │
                                     ├── turn 0: model sees `search_kb` and decides
                                     │     ├── answers directly  → greetings, general
                                     │     │                       knowledge, follow-ups
                                     │     │                       (no retrieval at all)
                                     │     ├── asks a clarifying question (no tool call)
                                     │     └── emits tool_call(search_kb, query)
                                     │           │
                                     │           └── hybrid retrieval (parallel)
                                     │                 ├── vector_retrieve → Qdrant top-k
                                     │                 └── graph_retrieve  → Kuzu entities
                                     │                                     → RRF fusion
                                     ├── turn 1..N: passages fed back; model answers
                                     │              or searches once more (capped)
                                     └── persist message + accumulated citations → SQLite

Stores (all embedded, zero ops):
  Qdrant   ./data/qdrant/          vector index
  Kuzu     ./data/kuzu/personal.db knowledge graph
  SQLite   ./data/sessions.db      chat history
```

---

## Requirements

- **Python 3.11** (3.12 works; 3.13+ not yet supported by all ML deps)
- **[Ollama](https://ollama.com)** running locally — always serves embeddings, and serves chat unless `LLM_HOST` points elsewhere
- *(optional)* **[FastFlowLM](https://github.com/ROCm/FastFlowLM)** to run chat on an AMD Ryzen AI NPU. Without it everything still works on Ollama
- **[uv](https://docs.astral.sh/uv/)** for dependency management
- 8 GB RAM minimum; 16 GB recommended. The default chat model is a 4B; larger models need proportionally more

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/pathakmanish7275/personal-database.git
cd personal-database

# 2. Install deps
uv sync

# 3. Pull the embedding model (required) and your preferred chat model
ollama pull nomic-embed-text
ollama pull qwen3.5:4b           # chat model — must support tool calling

# 4. Configure
cp .env.example .env
# Edit .env — defaults work; set LLM_HOST if you run FLM on the NPU

# 5. Run
./run.sh
# → open http://localhost:8765
```

---

## Configuration

Copy `.env.example` to `.env` and edit. All values have sensible defaults.

### Full configuration reference

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API base URL |
| `LLM_MODEL` | `qwen3.5:4b` | Chat model — must support tool calling |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
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
| `LLM_HOST` | _(= `OLLAMA_HOST`)_ | Chat-model server; point at FLM to run on the NPU |
| `FALLBACK_LLM_HOST` | _(= `OLLAMA_HOST`)_ | Where the fallback model lives |
| `FALLBACK_LLM_MODEL` | `qwen3.5:4b` | Fallback chat model |
| `MAX_TOOL_TURNS` | `2` | Max `search_kb` calls per turn |
| `VOICE_ENABLED` | `true` | Enable the Call button |
| `STT_BASE_URL` | `http://localhost:8123/v1` | Speech-to-text server |
| `TTS_BASE_URL` | `http://localhost:8880/v1` | Text-to-speech server |
| `VOICE_THINK` | `false` | Request reasoning on voice turns |
| `VOICE_ANSWER_MAX_TOKENS` | `300` | Keeps spoken answers short |
| `THINK_TOOL_TURNS` | `true` | Request reasoning on the opening turn |
| `THINK_ANSWER` | `false` | Request reasoning while synthesising |
| `TOOL_TURN_MAX_TOKENS` | `768` | Output cap on a tool turn (stops reasoning spirals) |
| `ANSWER_MAX_TOKENS` | `0` | Output cap on the answer; 0 = uncapped |
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

## Agentic retrieval

Retrieval is a **tool the model calls**, not a fixed pipeline stage. Each turn the
model is handed one tool, `search_kb`, and decides for itself what to do:

- **answer directly** — greetings, general knowledge, or anything answerable from
  the conversation so far. No retrieval runs, so these turns are fast.
- **call `search_kb`** — it writes its own query, reads the returned passages, and
  either answers or searches once more with better terms.
- **ask a clarifying question** — needs no special path; it is simply the model
  answering without calling the tool.

This replaced an earlier design that ran a separate planner LLM call before every
retrieval. That call cost ~31% of each turn's latency and was paid even when the
answer was obviously "just search" — or when no search was needed at all.

**Requirements.** This needs a model with tool-calling support, reached over the
**OpenAI-format** endpoint (`/v1/chat/completions`). FLM's Ollama-format
`/api/chat` silently *drops* the `tools` array — the model never learns the tools
exist. `qwen3.5` supports tool calling; `gpt-oss` does not.

**Guards.** Each drawn from a failure seen against the live model:

| guard | why |
|---|---|
| `MAX_TOOL_TURNS` (2) | bounds how many searches one turn may run |
| repeated-query detection | re-running an exhausted query cannot yield new context |
| placeholder rejection | a model once echoed an unfilled `<…>` template into the query |
| `TOOL_TURN_MAX_TOKENS` (768) | a thinking turn otherwise spiralled to the context cap — 4096 tokens / 245s, entirely inside the reasoning channel |
| raw-question fallback | if that cap is hit before a tool call appears, search the user's own words rather than return nothing |
| unknown tool / retrieval error | reported back to the model as text; never kills the turn |

**Reasoning** is requested per role: on for the opening turn, where choosing a good
query benefits from it, and off once passages are in hand and the model is only
synthesising. On the OpenAI endpoint `think: true` puts reasoning in its own
`reasoning_content` field, so it never appears in the answer.

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

118 tests, covering: sessions, memory compaction, ingest, KG extraction, hybrid retrieval, the agent loop (tool dispatch, loop guards, token caps, fallback), the streaming wire client, LLM runtime fallback, and the web API.

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
├── agent.py           # Agent loop: model drives search_kb as a tool
├── tools.py           # search_kb schema + citation ledger
├── llm_stream.py      # Streaming OpenAI-format client (FLM / Ollama)
├── llm_runtime.py     # safe_chat with fallback (memory compaction)
├── planner.py         # (legacy) pre-agent clarify-or-search planner
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

---

## Acknowledgements

NPU inference is powered by **[FastFlowLM](https://github.com/ROCm/FastFlowLM)** (MIT).
Speech-to-text uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper);
text-to-speech uses [Kokoro](https://github.com/hexgrad/kokoro); the voice pipeline
is built on [Pipecat](https://github.com/pipecat-ai/pipecat). Entity and relation
extraction use [GLiNER](https://github.com/urchade/GLiNER) and
[REBEL](https://github.com/Babelscape/rebel).

## License

MIT — see [LICENSE](LICENSE).
