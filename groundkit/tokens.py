"""Shared tokenising, so the text and negation layers agree on what a word is.

Kept in its own module because :mod:`groundkit.text` and
:mod:`groundkit.negation` both need it and importing either from the other would
be circular.
"""

from __future__ import annotations

import re

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of",
    "in", "on", "at", "to", "for", "and", "or", "but", "with", "as", "by",
    "it", "its", "this", "that", "these", "those", "from", "has", "have", "had",
    "which", "who", "whom", "their", "there", "then", "than", "into", "about",
}


def words(text: str) -> list[str]:
    """Every token, in order, stopwords included. ``\\w`` is Unicode-aware."""
    return re.findall(r"\w+", text.lower())


def content_tokens(text: str) -> list[str]:
    """Substantive tokens: lowercased, stopwords and single characters dropped."""
    return [t for t in words(text) if len(t) > 1 and t not in STOPWORDS]
