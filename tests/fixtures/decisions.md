# Decisions Log

## 2026-05-28 — Lock LLM choice
After evaluating Gemma 4 E4B and 26B-A4B, I decided to stay with gpt-oss:20b for the
personal-database assistant. Reason: reasoning headroom on hybrid retrieval beats raw
speed for this use case. Gemma 4 may be added later as a utility model.

## 2026-05-28 — UI direction
Pivoted from Open WebUI to a custom plain FastAPI + Jinja2 web UI. The Open WebUI
look is too polished/flashy for the spartan tool I want. Three pages: Chat, Library,
Ingest. Single launch via run.sh.

## 2026-05-27 — Non-LLM KG extraction by default
GLiNER + REBEL + HDBSCAN for the default ingest path. LLM extraction with gpt-oss
reserved for manually-flagged docs. Why: keep gpt-oss free for query-time synthesis;
ingest must be fast enough that I actually use it.
