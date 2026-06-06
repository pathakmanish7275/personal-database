# Sample Decisions Log

Fictional entries, kept as sample data for tests — not a real decisions log.

## 2024-02-10 — Embedding model
Chose a small local embedding model to keep ingestion fast on modest hardware.
Larger models can be swapped in later without changing the pipeline.

## 2024-02-11 — Storage layout
Keep the vector index and the graph store in separate on-disk directories so
each can be rebuilt independently without touching the other.

## 2024-02-12 — Extraction path
Use the non-LLM entity/relation extractor by default so ingestion stays fast
and offline; reserve heavier extraction for documents flagged by hand.
