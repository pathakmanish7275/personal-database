"""Zero-shot entity extraction via GLiNER (CPU).

Lazy-loaded singleton: the model is fetched on first use and cached. If GLiNER
or its weights can't load, extraction degrades to returning nothing rather than
breaking ingestion."""

from __future__ import annotations

import logging

from ..config import config
from ._text import windows

log = logging.getLogger(__name__)

# Personal-ontology labels — what we want to recognize in Alex's corpus.
LABELS = [
    "person",
    "organization",
    "project",
    "product",
    "technology",
    "tool",
    "concept",
    "place",
    "event",
    "decision",
    "role",
    "field",
]

_model = None
_failed = False


def _get_model():
    global _model, _failed
    if _model is not None or _failed:
        return _model
    try:
        from gliner import GLiNER

        # Try cached-only first so offline runs don't make network calls.
        try:
            _model = GLiNER.from_pretrained(config.gliner_model, local_files_only=True)
        except Exception:  # noqa: BLE001
            _model = GLiNER.from_pretrained(config.gliner_model)
        log.info("loaded GLiNER %s", config.gliner_model)
    except Exception as e:  # noqa: BLE001
        log.warning("GLiNER unavailable; entity extraction disabled (%s)", e)
        _failed = True
    return _model


def available() -> bool:
    return _get_model() is not None


def extract_entities(text: str, threshold: float | None = None) -> list[tuple[str, str]]:
    """Return a de-duplicated list of (name, type) for one chunk of text."""
    model = _get_model()
    if model is None or not text or not text.strip():
        return []
    thr = config.kg_entity_threshold if threshold is None else threshold
    seen: dict[str, tuple[str, str]] = {}
    for piece in windows(text, max_words=300):
        try:
            preds = model.predict_entities(piece, LABELS, threshold=thr)
        except Exception:  # noqa: BLE001
            log.exception("GLiNER predict failed on a window")
            continue
        for p in preds:
            name = (p.get("text") or "").strip()
            if len(name) < 2:
                continue
            key = name.lower()
            if key not in seen:
                seen[key] = (name, p.get("label") or "concept")
    return list(seen.values())
