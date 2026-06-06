"""Non-LLM knowledge-graph extraction (GLiNER entities + REBEL relations)."""

from __future__ import annotations

from ..config import config
from .entities import extract_entities
from .relations import extract_relations


def extract_for_chunk(text: str) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    """Run the enabled extractors on one chunk → (entities, relations)."""
    entities = extract_entities(text)
    relations = extract_relations(text) if config.kg_relations else []
    return entities, relations


__all__ = ["extract_entities", "extract_relations", "extract_for_chunk"]
