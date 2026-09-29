# Knowledge-graph extraction with a hosted model

Separate from the app on purpose. `personal_db` builds its graph with the
original GLiNER + REBEL extractor; nothing in `src/` imports anything here.
This directory is a one-off pipeline for re-indexing the **existing** corpus
with a hosted model, which is markedly more accurate but needs a network call
per chunk.

## Why hosted

Measured on 10 identical corpus chunks, production code path both sides:

| extractor | s/chunk | entities | relations |
|---|---|---|---|
| REBEL/GLiNER (the app's) | 2.53 | 57 | 19 |
| qwen3.5:4b on FLM/NPU | 8.77 | 47 | 7 |
| gemma-4-26b-a4b | 9.70 | 176 | 116 |
| deepseek-v4-flash-0731 | 12.29 | 223 | 172 |

Both local options were dead ends:

* **gpt-oss:20b cannot tool-call on FLM at all.** Even with `tool_choice`
  forcing a function, it returns no `tool_calls` and answers in prose. Giving
  it a tool also does not shorten its reasoning (77.6s / 5465 chars without
  tools vs 77.7s / 5484 with). Its reasoning cannot be reduced either —
  neither `reasoning_effort`, `options.reasoning_effort` nor a `Reasoning: low`
  system line beats run-to-run noise, and FLM 1.0.6 adds no server-mode
  control.
* **qwen3.5:4b** is strictly worse than REBEL here: slower, fewer entities,
  under half the relations, six of ten chunks yielding none, and it invents
  relations. Tools make it worse still — it stringifies nested arrays and the
  string is malformed JSON.

## Why filtering lives in `phase2_build.py`, not the prompt

A tightened prompt cut relations by only 7.5% and kept emitting the exact
patterns it was told to omit. Extraction is also nondeterministic even at
`temperature: 0` (OpenRouter routes across provider replicas): only ~214 of
~750 relations recurred between identical runs. So prompt-level filtering can
be neither relied on nor measured at this sample size. The rules in phase 2 are
deterministic and remove ~20%.

## Running it

Needs `OPEN_ROUTER_KEY` in `.env`. Run from the repo root.

```bash
# phase 1 — extract to JSONL. Resumable: rerunning skips finished chunks.
set -a; . ./.env; set +a
uv run python tools/kg-deepseek/phase1_extract.py

# phase 2 — build a graph from that JSONL. No API calls, so filters are free
# to retune: edit REASONS and rerun.
DRY=1 uv run python tools/kg-deepseek/phase2_build.py   # report only
DRY=0 uv run python tools/kg-deepseek/phase2_build.py   # write
```

Phase 2 writes to `data/kg-rebuild/kuzu/personal.db` by default, never to the
live graph. Swapping it in is a manual `mv`, with the app stopped — Kuzu is
embedded and single-writer.

The split matters: phase 1 is the slow, rate-limited, paid part, and its raw
output is banked in `data/kg-rebuild/extractions.jsonl`. Changing what reaches
the graph never costs another request.

## Cost

deepseek-v4-flash is $0.04/M in, $0.08/M out — about **$0.28** for a
3283-chunk corpus, or free within OpenRouter's 1000 requests/day. Its `:batch`
variant is *more* expensive ($0.11/$0.33), so plain concurrent requests are
both cheaper and faster. 661 chunks took 17.1 minutes at 8 concurrent with no
rate limiting and zero failures.

## `dedup_qdrant.py`

A maintenance script, unrelated to extraction. It removes duplicate chunk
vectors, keyed on `(document content-hash, chunk text hash)`. It was needed
once because `list_documents` fell back to LlamaIndex's per-ingest `ref_doc_id`
when resolving document identity, so the dedup gate compared a content hash
against a set of fresh UUIDs and never matched — every ingest re-embedded the
whole corpus. That bug is fixed in `ingest.py::_stable_doc_id`; this script
cleans up a store that already has the damage.
