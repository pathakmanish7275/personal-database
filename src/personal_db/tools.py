"""Tools the chat model can call.

Deliberately a *single* tool. The model picking between several retrievers is
not a capability we need — `hybrid_retrieve` already runs vector and graph
search in parallel and fuses them with RRF, which is strictly better than
asking a 4B model to choose. A small tool surface also keeps tool-call accuracy
high; small models degrade quickly as the number of tools grows.
"""

from __future__ import annotations

import logging

from .retrieve import RetrievalContext, RetrievedChunk

log = logging.getLogger(__name__)

SEARCH_KB = "search_kb"

# Passed verbatim as the OpenAI `tools` array.
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": SEARCH_KB,
            "description": (
                "Search the user's personal knowledge base — their notes, journals, "
                "PDFs and books — and return matching passages. Call this whenever the "
                "question refers to the user's own documents, notes, projects or "
                "history, AND whenever it asks about the user themselves: who they "
                "are, what they work on, what they have built or decided, their "
                "background. You do not know the user; everything you can say about "
                "them has to come from this tool. Do not answer such questions from "
                "general knowledge or from what the system prompt implies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A short natural-language search query, normally under ten "
                            "words — a phrase as it would actually appear in the "
                            "user's writing. Retrieval is semantic, so a focused "
                            "phrase works and a long list of loosely related keywords "
                            "does not. Search for one thing at a time."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    }
]


class CitationLedger:
    """Accumulates citations across however many searches the model runs.

    The model may call `search_kb` more than once, and the same chunk can come
    back from several searches. Citation numbers must stay stable and unique
    across the whole turn — the UI's `[N]` chips refer to them — so this hands
    out each chunk exactly one number, keyed by (path, text)."""

    def __init__(self) -> None:
        self._seen: dict[tuple[str, str], int] = {}
        self.cites: list[dict] = []

    def number_for(self, chunk: RetrievedChunk) -> int:
        key = (chunk.path, chunk.text)
        existing = self._seen.get(key)
        if existing is not None:
            return existing
        n = len(self.cites) + 1
        self._seen[key] = n
        self.cites.append(
            {
                "n": n,
                "name": chunk.name,
                "path": chunk.path,
                "doc_id": chunk.doc_id,
                "score": chunk.vector_score,
                "origin": chunk.origin,
                "graph_hits": chunk.graph_hits or None,
            }
        )
        return n


def format_hits(ctx: RetrievalContext, ledger: CitationLedger) -> str:
    """Render retrieved passages as the tool's return value.

    Passages are labelled with their citation number so the model can attribute
    claims. An empty result is stated explicitly rather than returned blank: a
    blank tool result reads as a malfunction and invites the model to invent an
    answer instead of saying it found nothing."""
    if not ctx.chunks:
        return "No matching passages found in the knowledge base."

    parts = []
    for chunk in ctx.chunks:
        n = ledger.number_for(chunk)
        parts.append(f"[{n}] {chunk.name}\n{chunk.text.strip()}")

    if ctx.relations:
        rels = "\n".join(f"- {h} ─{p}→ {t}" for h, p, t in ctx.relations)
        parts.append(f"Related entities from the knowledge graph:\n{rels}")

    return "\n\n".join(parts)
