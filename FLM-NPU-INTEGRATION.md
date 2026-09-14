# FLM / NPU Chat-LLM Integration — Working Notes

What we changed to run the assistant's chat model on the local AMD NPU via
**FastFlowLM (FLM)**, how it's wired, the bugs we hit, and how to operate it.

Status: **working, verified end-to-end 2026-09-13.** App serves chat on the NPU
(`qwen3.5:4b` via FLM) as an **agent**: the knowledge base is a tool the model
calls, retrieval is no longer a fixed pipeline stage, and tokens stream live to
the UI. Transparent fallback to local Ollama; shutdown never kills the NPU
mid-generation.

**Model choice is forced by tool calling.** `gpt-oss:20b` is the stronger
synthesiser and was briefly the default, but its model card says *"Tool Calling
Support: No"* and testing confirms it. `qwen3.5` supports tool calling, so the
agent design requires it. See §8 for the quality/latency trade.

---

## 1. Architecture — how the model is wired

LlamaIndex's global `Settings.llm` is the single chat entry point. Selection
happens in `src/personal_db/stores.py::configure_llama_index()`.

| Role | Server | Model | Where |
|------|--------|-------|-------|
| Primary chat LLM | **FLM** `http://localhost:52625` `/v1` | `qwen3.5:4b` (NPU) | `LLM_HOST` + `LLM_MODEL` in `.env` |
| Chat fallback | Ollama `http://localhost:11434` | `qwen3.5:4b` | `FALLBACK_LLM_HOST` + `FALLBACK_LLM_MODEL` |
| Embeddings | Ollama `http://localhost:11434` | `nomic-embed-text` | `OLLAMA_HOST` + `EMBED_MODEL` |

Key idea: **chat moved to the NPU, embeddings + fallback stayed on Ollama.**
That's why there is a *separate* `LLM_HOST` (new) from `OLLAMA_HOST`. FLM speaks
an Ollama-compatible API on its own port, so the existing `Ollama` LlamaIndex
client just points at a different `base_url`.

Config knobs added (`src/personal_db/config.py`):
- `llm_host` (`LLM_HOST`) — chat LLM server, defaults to `OLLAMA_HOST`.
- `fallback_llm_host` (`FALLBACK_LLM_HOST`) — where the fallback lives.
- `fallback_llm_model` (`FALLBACK_LLM_MODEL`) — fallback model (must exist on that host).

Provider/fallback logic (`src/personal_db/llm_runtime.py`):
- `_fallback_enabled()` — true for a remote primary (`gemini`) **or** when the
  chat LLM sits on a *different* host than the fallback (the FLM case).
- `_primary_stream_is_unsafe()` — **removed.** Streaming is no longer emulated
  (§3a). The agent path streams through `llm_stream.stream_chat`; `llm_runtime`
  now serves only the non-tool `/api/chat` callers (memory compaction).

---

## 2. Why sudo went away

Initially FLM had to run as root. Root cause chain:

1. The **very first** run was `sudo flm serve`, which downloaded the ~7.8 GB
   model into **`/root/.config/flm/`** — so only root could read it. The model
   files (not FLM itself) were the reason for sudo.
2. FLM needs a large **`RLIMIT_MEMLOCK`** (it `MAP_LOCKED`s ~9 GB of NPU
   buffers). The desktop session's hard limit was **8 MB**.

Fixes applied:
- Copied the models to `~/.config/flm/models/` so the normal user can read them.
- `/dev/accel/accel0` already had an ACL granting the user `rw`.
- Raised memlock via a systemd drop-in — PAM's `limits.d` did **not** take
  effect because modern GNOME spawns everything (terminals included) from the
  **user systemd manager**, which is started by PID 1 *before* PAM, capped at
  systemd's built-in 8 MB. The file that worked:
  `/etc/systemd/system/user@.service.d/memlock.conf` → `[Service]\nLimitMEMLOCK=infinity`,
  then a **full logout/login** (the user manager must restart). Note: the user
  manager persists across logout when kept alive by a lingering user service, so
  a plain logout was not enough — a reboot cleanly reset it.

Result: `flm serve` runs as `you`, **no sudo**, and `run.sh`/`stop.sh`
never prompt.

---

## 3. FLM stability bugs discovered (v1.0.5) and the mitigations

### 3a. Streaming deadlock — NO LONGER REPRODUCIBLE (retested 2026-09-13)
The original finding: any request with `"stream": true` made FLM wedge at
`NPU Locked → Prefill → Creating checkpoint at context length N`, leaving the
process alive but the API dead. The mitigation was *emulated streaming* — one
sync `chat()` yielded as a single SSE chunk.

**This no longer reproduces.** On FLM v1.0.5 with `qwen3.5:4b` over
`/v1/chat/completions`, streaming is stable:

- short prompt: 11 clean SSE chunks, `[DONE]` terminator, API alive
- ~2000-token prompt: **11 checkpoint events** traversed, no wedge, API alive
- streaming + tools, streaming + `think:true`, and a full tool round-trip: all
  stable across dozens of requests

What changed is not known precisely — v1.0.5, the memlock fix, `-q 1`, and a
different model all moved at once. Emulated streaming has been **removed**; the
UI now streams tokens live. If a wedge ever returns, the fallback is to stop
passing `stream: true` in `llm_stream.stream_chat`.

### 3b. Tools are silently dropped on the Ollama-format endpoint
The single most important wire fact, and it fails **silently**:

| endpoint | `tools` array |
|---|---|
| `/api/chat` (Ollama format) | **discarded** — model never learns tools exist |
| `/v1/chat/completions` (OpenAI format) | works; returns structured `tool_calls` |

On `/api/chat` the model does not error — it apologises that it cannot reach
your notes and then answers from general knowledge. That is why the agent talks
to `/v1/chat/completions` directly (`llm_stream.py`) rather than through the
LlamaIndex `Ollama` client, which targets `/api/chat`.

### 3c. `think: false` is ignored → `...` leaks into answers
FLM/Qwen emits the reasoning as think-tag blocks **inside** the message
content. Left in, this pollutes chat history, the JSON planner, and memory
summaries. (Also: the model sometimes narrates literal `...` *inside* its
block, which no regex can fully repair — see §4.)

**Mitigation:** `_strip_think()` (non-greedy, multi-occurrence) +
`_clean_response()` applied to every `safe_chat` and emulated-stream result.
Two subtle traps fixed along the way:
- LlamaIndex `ChatMessage.content` is a **read-only property over `blocks`**, so
  `model_copy(update={"content": ...})` silently no-ops. Must rebuild the
  message through its constructor.
- A stray think-tag inside a code comment broke the module's syntax once; tags
  must be spelled out (`<` `think` …) or avoided in comments.

**RESOLVED 2026-09-13 — the field exists, but only on the OpenAI endpoint.**
An earlier draft proposed `LLM_THINKING=true` on the theory that FLM would
return thinking on a separate field. On `/api/chat` it does **not**: with `think`
true, false, or omitted the message is always `role`/`content`/`images` and
reasoning is in-band, so the regex strip is the only option there.

On **`/v1/chat/completions`** it does. With `think: true`, reasoning streams in
its own delta field — `reasoning_content` on FLM, `reasoning` on Ollama — and
never touches `content`. This is what makes live token streaming safe: there is
no `<think>` tag to filter out of the stream, so no incremental state machine is
needed. The agent path relies on this; `_strip_think` remains for the
`/api/chat` callers (memory compaction).

⚠️ **Do not send `reasoning_effort`.** On FLM it produces `finish_reason:
length` with **zero content**. `max_tokens` (and `max_completion_tokens`) *are*
honoured and are the correct way to bound output; `num_predict` is ignored.

Also measured: `LLM_THINKING=auto` (→ `false` for ≥4b) is the **right** default,
not a bug. On realistic prompts `think:false` is fast and clean; `think:true`
spent 158 s / 2778 tokens on a one-sentence summary.

### 3d. Truncated reasoning defeats the strip entirely
When a model spirals it hits the output cap (`done_reason="length"`,
`eval_count=4096`) **mid-reasoning**, so no closing tag is ever emitted. The
paired regex then matches nothing and the entire reasoning dump passes through
untouched — measured **14,639 chars of raw reasoning, and no answer at all**,
straight into the UI, SQLite history, the JSON planner and memory summaries.

**Mitigation:** `_OPEN_THINK_RE` also strips an *unterminated* block, and
`_primary_chat_with_recovery()` treats "produced bytes but nothing survived
extraction" as a failed turn — retrying once on the Ollama fallback rather than
returning an empty or reasoning-filled answer.

### 3e. Reasoning spirals are invisible from the client
With `think: true` a vague question can send qwen into a reasoning spiral that
runs to the context cap **entirely inside `reasoning_content`** — so the client
sees a modest number of deltas and no content, while the NPU is pinned.
Measured: one turn held the NPU for **245 seconds** on an 819-token prompt
(~4400 tokens of reasoning), against 7 seconds for the same shape of turn on
the next question. It is intermittent, not deterministic.

**Mitigation:** `TOOL_TURN_MAX_TOKENS` (default 768) caps tool turns, and if the
cap is hit before a tool call appears the agent searches the user's raw question
rather than returning an empty turn. That took one measured case from **274s to
24s**. Only the answer turn is allowed to run uncapped.

Diagnose these from FLM's own journal, which is the only place the stall is
visible:

```bash
journalctl --user -u flm-npu --since "-10 min" -o short-precise \
  | grep -E "Prefill chunk|NPU Locked|Lock Released"
```

### 3f. gpt-oss speaks harmony, and FLM does not parse it
`gpt-oss:20b` replies with a raw channel transcript in `content`:

```
<|start|>assistant<|channel|>analysis<|message|>…reasoning…<|end|>
<|start|>assistant<|channel|>final<|message|>the answer<|end|>
```

The answer is the `final` channel. `_strip_think()` extracts it — a *positive*
match, which degrades safely: no `final` channel means the model never reached
an answer, which feeds the same recovery path as §3d.

**Trap (cost us a silent regression):** a constrained reply puts an extra token
in the channel header — `<|channel|>final <|constrain|>answer<|message|>Sam`.
A pattern requiring `final` and `<|message|>` to be adjacent drops the answer,
returns empty, and wrongly falls back to Ollama on **every constrained answer**.
Caught only by an end-to-end run: the app served an Ollama result while
reporting success. The header now skips any non-`<|message|>` tokens.

### 3g. `qwen3.5:9b` was unstable
9B wedged frequently and non-deterministically (worse after repeated `SIGKILL`s
that never let the runtime release NPU contexts). After a reboot + switching to
**4B**, the sync path became reliable.

**Current status (2026-09-13):** `qwen3.5:4b` is the primary again — the agent
design needs tool calling, which `gpt-oss` does not support. Only `:9b` remains
unused; it was removed from FLM's local store along with `:0.8b`. The 4B has
been stable across the full agent test suite. Ollama keeps its own `qwen3.5:4b`
as the fallback — a separate install.

---

## 4. Request serialization — queue in *our* system, never on the NPU

FLM's own NPU queue is exactly where the crashes cluster ("NPU busy, request
queued …"). So we make sure a second request never reaches it:

1. **Job-level queue** (`src/personal_db/jobs.py`) — a process-wide asyncio
   run-lock serializes generations; queued chats emit a `queued` UI event. This
   already existed.
2. **Hard call gate** (`llm_runtime._call_gate`, a `threading.Lock`) — held for
   the *entire* duration of every LLM call, including full stream consumption.
   Covers callers *outside* the job lock (memory summarizer, planner). Makes it
   structurally impossible for two requests to be in flight to the model server.
   (An earlier note claimed "verified: 3–4 concurrent callers → 0 overlapping
   model calls". That measurement has **not** been reproduced since; treat the
   guarantee as argued-from-construction, not measured.)

   The gate is held for the life of the stream generator, so whoever opens a
   stream **must** close it: `chat.py` wraps consumption in `try/finally` with
   `response_stream.close()`. Leaving it to GC would gate every future LLM call
   in the process on refcount timing.

Result: waiting happens in our process; FLM's queue stays empty.

---

## 5. Safe shutdown — never kill the NPU mid-call

Dropping an open HTTP stream to FLM mid-generation is what crashed it. So:

- **Cooperative cancel** (`jobs.py` + `chat.py`): once `job.streaming` is set, a
  cancel is *not* a task kill. The runner stops emitting/persisting but **drains
  the stream to completion**, so the model finishes cleanly. (Before the stream
  opens — retrieval/compaction — cancel is still an instant hard cancel; a
  cancelled `asyncio.to_thread` still lets its in-flight sync HTTP call finish,
  so the connection is never torn down mid-request.)
- **In-flight tracking** (`llm_runtime.active_calls()`): counter spans the whole
  call/stream so shutdown knows when the NPU is idle.
- **Lifespan drain** (`web/app.py`): on shutdown, waits for jobs + ingest +
  `active_calls()==0` before exit, bounded by `SHUTDOWN_DRAIN_SECONDS` (default
  300s).
- **uvicorn** runs with `--timeout-graceful-shutdown` so SSE connections (which
  hold a running job) are awaited rather than severed.

---

## 6. Operations

### Start
```bash
./run.sh
```
Ensures Ollama is up, verifies the fallback + embedding models, starts **FLM** as
the current user if it isn't already serving on `LLM_HOST`, waits for health,
then launches the FastAPI app on `PORT` (default 8765) and opens the browser. If
FLM fails to come up it prints why (usually memlock or missing model in
`~/.config/flm`) and continues — chat just uses the Ollama fallback.

### Stop
```bash
./stop.sh              # graceful: drain in-flight generations, then stop FLM
KEEP_FLM=1 ./stop.sh   # stop the app but leave the NPU server warm
```
Order is deliberate: app first (drains), then FLM **only once it is idle**.

### Run FLM under supervision (survives shell/session kills)
```bash
systemd-run --user --unit=flm-npu --property=LimitMEMLOCK=infinity \
    flm serve gpt-oss:20b --port 52625 -q 1 -s 4
```
`-q 1` caps FLM's NPU queue to 1 (matches our serialization), `-s 4` limits
sockets. `systemctl --user {status,stop} flm-npu` to inspect/stop.

**`FLM_PORT` as an env var is a no-op** — `flm` reads the port from `--port`
only (`strings /usr/bin/flm` contains no `FLM_PORT`). An earlier version of this
command and of `run.sh` passed `env FLM_PORT=…`; it appeared to work purely
because 52625 is FLM's built-in default (`flm port`). Set to anything else and
FLM binds 52625 while the health check polls the port you asked for.

### Change the model
Edit `.env` and set `LLM_MODEL`. `run.sh` now sources `.env`, so `FLM_MODEL`
defaults to `LLM_MODEL` and the two cannot drift; export `FLM_MODEL=` only to
serve something different from what the app asks for. Embeddings always stay on
`OLLAMA_HOST`.

Note `run.sh`'s FLM health check hits `/api/tags`, which returns FLM's **whole
upstream catalogue**, not what is loaded — it proves the server is up, not that
the right model is being served. The model itself loads lazily on first request
(RSS stays ~0 until then; `gpt-oss:20b` settles at ~16.3 GB).

---

## 7. Gotchas & open items

- **`vps-url-opener`** (`~/.local/bin/vps-url-opener`) grabs port **8765** at
  login and repeatedly blocked the app from binding. Worked around by
  `fuser -k 8765/tcp` before launching, but this will recur every login. Real
  fix is to move the app to another `PORT=` or reconfigure/remove that helper —
  **your call**.
- **Stale env shadowing `.env`**: a leftover `LLM_MODEL` in the shell/desktop
  session silently overrode the config. Fixed by loading `.env` with
  `override=True` so project config wins.
- **Never `SIGKILL` FLM** during/between generations if you can help it —
  repeated hard kills degrade the NPU/driver state until a reboot. Use
  `stop.sh` / `systemctl --user stop flm-npu`.
- **Current model is `gpt-oss:20b`.** See §9 for why it beat the qwen line.
- **Think-tag stripping is best-effort.** When Qwen narrates literal tag names
  inside its reasoning, a regex can't fully clean it. This matters much less
  now: gpt-oss uses harmony channels (§3f), which are extracted positively
  rather than stripped. `LLM_THINKING=true` is **not** an escape hatch — FLM has
  no structured thinking field (§3c).
- **`/etc/security/limits.d/flm.conf` does not exist** and never worked — the
  memlock limit comes from the systemd drop-in (§2). A stale comment in `run.sh`
  pointed at the PAM path; corrected.
- **Emulated streaming means no per-token drip** on the NPU path — the answer
  appears at once. Cloud providers (`LLM_PROVIDER=gemini`) still stream
  normally; only FLM/`ollama-host` primaries are emulated.
- **`LLM_THINKING=auto`** disables `think` for ≥4b Qwen models and omits the
  flag for `:2b` (the 2B loses tool-use when thinking is off) — FLM ignores the
  flag anyway, so the strip in §3c is what actually keeps output clean.

---

## 8. NPU performance — measured, not assumed

Measured 2026-09-13 on AMD Ryzen AI 7 350 (XDNA2), all via FLM `/api/chat`
telemetry (`eval_count` / `eval_duration`, so throughput excludes HTTP overhead).

### Decode cost model

    ms/token ≈ 10 + 11.3 × (active params in billions)

Fitted on 0.8B (19.0 ms) and 4B (55.2 ms); it predicts `gpt-oss:20b` MoE
(3.6B active) at **50.6 ms** against **50.8 ms measured**.

| model | active params | decode |
|---|---|---|
| `qwen3.5:0.8b` | 0.8B | 52.6 tok/s |
| `gpt-oss:20b` (MoE) | ~3.6B | 19.7 tok/s |
| `qwen3.5:4b` (dense) | 4B | 18.1 tok/s |

Decode is **weight-streaming bound** (~49 GB/s effective on shared LPDDR5x),
so cost tracks *active* parameters, not total. MoE is close to free: a 20B MoE
runs at the speed of a 4B dense model.

**Operational consequence: run the most capable model that fits RAM.** Speed
barely moves with model size, so `gpt-oss:20b` costs ~nothing over a 4B.

### Prefill is batched and cheap — but don't benchmark it on toy prompts
On a realistic ~2000-token prompt: **1730 tok/s** (0.8b) and **326 tok/s**
(20b) — a healthy 17–35× over decode. Short prompts (<100 tokens) measure
25–45 tok/s, which is *fixed per-request cost*, not prefill throughput. An
earlier reading of those short-prompt numbers wrongly suggested FLM wasn't
batching prefill at all.

### gpt-oss:20b vs qwen3.5:4b — the trade we accepted
gpt-oss is the better synthesiser. At near-identical tok/s, qwen reasons in-band
and emits 10–30× more tokens for the same task, so wall-clock is worse and it
can fail outright:

| prompt | `qwen3.5:4b` | `gpt-oss:20b` |
|---|---|---|
| "all but 9 sheep" riddle | 4096 tok / 233 s, hit cap, **no answer** | "Nine." — 359 tok / 18 s |
| one-sentence summary | 3 s (`think:false`) / 155 s (`think:true`) | 7 s |
| entity extraction | 36 s | 20 s, correct |
| strict-JSON output | 2 s | valid JSON, 10 s |

### Where a real query's time went — and what the agent changed
The old fixed pipeline always ran a planner call before retrieval. Measured on
"Summarise what my database contains", 53.8 s wall clock, from FLM's journal:

| stage | prompt | prefill | decode | total |
|---|---|---|---|---|
| planner (now removed) | 453 tok | 3.97 s | 12.6 s | **16.6 s (31%)** |
| retrieval (vector + graph + RRF) | — | — | — | **0.46 s (1%)** |
| synthesis | 1038 tok | 4.67 s | 32.0 s | **36.7 s (68%)** |

**Decode is ~83% of total; prefill ~16%; retrieval and embedding are noise.**
Context-size dials (`RERANK_TOP_N`, `TOP_K_*`) therefore buy far less than they
appear to: trimming 500 tokens of context saves ~2 s of prefill, while one
avoidable LLM round-trip costs 16 s.

That is what the agent design attacks — not by making calls faster, but by not
making them. Measured after the switch (`qwen3.5:4b`):

| query | before | after |
|---|---|---|
| "Hi there" | 53 s | **7 s** — no retrieval |
| "What is the capital of France?" | 53 s | **10 s** — no retrieval |
| knowledge-base question | 53.8 s | **19–32 s** |

The win is structural: greetings, general knowledge and conversational
follow-ups now cost one short call instead of planner + retrieval + synthesis.

**Earlier idea, now moot.** These notes previously recommended moving the
planner to `qwen3.5:2b` on Ollama/CPU (measured 1.9–5.5 s at 27 tok/s — faster
*per token* than the NPU, because the model is 10× smaller). Deleting the
planner outright was better. The underlying principle still holds and is worth
remembering for future utility calls: **NPU for the big model, CPU for small
utility models** — the NPU's advantage is capacity, not speed.

### Dead ends (tested, don't retry without new FLM versions)
- **`reasoning_effort` is not plumbed through FLM.** gpt-oss generates an
  `analysis` channel at full decode cost that `_strip_think()` then discards.
  Tried `reasoning_effort` top-level, inside `options`, and a `Reasoning: low`
  system message: no reduction (the system message made it *worse*, 209 vs 76
  tokens, since the model treats it as content).
- **FLM embeddings are off by default, not absent.** `/api/embeddings` returns
  HTTP 200 with a body of literally `null` until FLM is started with
  `--embed 1` (`-e`); `/api/embed` is 404 regardless. Not worth enabling here:
  `flm serve` takes exactly one model tag, so embeddings would contend with the
  chat model for the NPU, and a query embedding is **20 ms** — 0.04% of a query.
  Idle Ollama holds only **0.03 GB** RSS, so keeping it costs nothing.

---

## 9. Files touched

- `src/personal_db/config.py` — `llm_host`, `fallback_llm_host/model`; `.env`
  loaded with `override=True`.
- `src/personal_db/stores.py` — chat LLM uses `llm_host`; embeddings stay on
  `ollama_host`.
- `src/personal_db/llm_stream.py` — **new.** Streaming OpenAI-format client for
  FLM and Ollama: tool-call delta accumulation, `reasoning_content`/`reasoning`
  handling, `max_tokens`, and the shared call gate.
- `src/personal_db/agent.py` — **new.** The tool loop: turn classification,
  content buffering, loop guards, token caps, per-role reasoning, fallback.
- `src/personal_db/tools.py` — **new.** `search_kb` schema and the citation
  ledger that keeps `[N]` stable across multiple searches.
- `src/personal_db/planner.py` — **legacy.** No longer in the pipeline; the
  model decides search-vs-clarify itself. Kept for reference.
- `src/personal_db/llm_runtime.py` — call gate (now shared via `call_gate()`),
  in-flight counter, fallback rule, think-tag stripping (paired **and**
  unterminated), harmony `final`-channel extraction, and
  `_primary_chat_with_recovery()`. Now serves only the `/api/chat` callers
  (memory compaction); `safe_stream_chat` is no longer used by the pipeline.
- `src/personal_db/jobs.py` — cooperative cancel + `drain_all()`.
- `src/personal_db/chat.py` — `streaming` flag, drain-on-cancel loop, and
  `try/finally` closing the stream so the call gate is never leaked.
- `src/personal_db/web/app.py` — lifespan shutdown drain.
- `src/personal_db/ingest_jobs.py` — `drain_all()`.
- `run.sh` / `stop.sh` — start/stop FLM as the user, graceful teardown, model
  checks. `run.sh` now sources `.env` (launcher-only vars still overridable on
  the command line), passes `--port`, and keeps the `-q 1 -s 4` NPU guardrails.
- `.env` / `.env.example` — new vars; model set to `gpt-oss:20b`.
- `tests/test_llm_runtime.py` — host-based fallback rule, plus extraction tests
  for paired/unterminated think blocks, harmony channels (including the
  constrained form) and answerless-reply recovery.
