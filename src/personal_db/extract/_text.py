"""Shared text helpers for extractors."""

from __future__ import annotations


def windows(text: str, max_words: int = 300) -> list[str]:
    """Split long text into word windows so models stay within their input limits."""
    words = (text or "").split()
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]
