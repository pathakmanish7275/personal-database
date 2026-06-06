"""Relation (triple) extraction via REBEL (CPU, seq2seq).

REBEL emits a linearized string with <triplet>/<subj>/<obj> markers that we
parse back into (head, predicate, tail). Heavier than GLiNER, so it can be
disabled with KG_RELATIONS=false. Lazy-loaded and degrades to [] on failure."""

from __future__ import annotations

import logging

from ..config import config
from ._text import windows

log = logging.getLogger(__name__)

_tok = None
_model = None
_failed = False


def _get():
    global _tok, _model, _failed
    if _model is not None or _failed:
        return _tok, _model
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        _tok = AutoTokenizer.from_pretrained(config.rebel_model)
        _model = AutoModelForSeq2SeqLM.from_pretrained(config.rebel_model)
        _model.eval()
        log.info("loaded REBEL %s", config.rebel_model)
    except Exception:  # noqa: BLE001
        log.exception("REBEL unavailable; relation extraction disabled")
        _failed = True
    return _tok, _model


def available() -> bool:
    _, model = _get()
    return model is not None


def _parse_triplets(text: str) -> list[tuple[str, str, str]]:
    """Canonical REBEL decoder: turn the linearized output into (head, rel, tail)."""
    triplets: list[tuple[str, str, str]] = []
    relation = subject = object_ = ""
    current = "x"
    cleaned = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in cleaned.split():
        if token == "<triplet>":
            current = "t"
            if relation:
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
                relation = ""
            subject = ""
        elif token == "<subj>":
            current = "s"
            if relation:
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
            object_ = ""
        elif token == "<obj>":
            current = "o"
            relation = ""
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token
    if subject and relation and object_:
        triplets.append((subject.strip(), relation.strip(), object_.strip()))
    return triplets


def extract_relations(text: str) -> list[tuple[str, str, str]]:
    """Return de-duplicated (head, predicate, tail) triples for one chunk."""
    tok, model = _get()
    if model is None or not text or not text.strip():
        return []
    import torch

    out: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for piece in windows(text, max_words=180):
        try:
            inputs = tok(piece, max_length=256, truncation=True, return_tensors="pt")
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_beams=3,
                    length_penalty=1.0,
                    early_stopping=True,
                )
            decoded = tok.batch_decode(gen, skip_special_tokens=False)[0]
        except Exception:  # noqa: BLE001
            log.exception("REBEL generate failed on a window")
            continue
        for head, rel, tail in _parse_triplets(decoded):
            if not (head and rel and tail):
                continue
            sig = (head.lower(), rel.lower(), tail.lower())
            if sig in seen:
                continue
            seen.add(sig)
            out.append((head, rel, tail))
    return out
