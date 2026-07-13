"""Grounding free-text answers against retrieved evidence (RAG faithfulness).

Splits an answer into sentences and checks each one against the provided source
passages: a sentence is supported when enough of its content words appear in some
source. Unsupported sentences are flagged, never trusted.

This check is deliberately lexical and dependency-free: it flags sentences whose
content words are largely absent from the evidence (the common RAG failure). It is
not entailment. A sentence that is mostly grounded but swaps a single token (a
wrong number, a negation) can still score above threshold and pass; use an NLI
model where that matters. (The structured layer in :mod:`groundkit.verify` catches
wrong values exactly; this text layer is the softer, content-overlap check.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .provenance import Provenance

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of",
    "in", "on", "at", "to", "for", "and", "or", "but", "with", "as", "by",
    "it", "its", "this", "that", "these", "those", "from", "has", "have", "had",
    "which", "who", "whom", "their", "there", "then", "than", "into", "about",
}


def _content_tokens(text: str) -> list[str]:
    # \w is Unicode-aware, so non-Latin scripts (CJK, Cyrillic, accented Latin)
    # produce real tokens instead of vanishing and being vacuously "supported".
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """Naive sentence split on terminal punctuation."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def support_score(sentence: str, source: str) -> float:
    """Fraction of a sentence's content words that appear in ``source`` (0-1)."""
    sent = set(_content_tokens(sentence))
    if not sent:
        return 1.0  # nothing substantive to support
    src = set(_content_tokens(source))
    return len(sent & src) / len(sent)


@dataclass(frozen=True)
class SentenceVerdict:
    sentence: str
    supported: bool
    score: float
    source_index: int | None  # best-matching source when supported

    @property
    def provenance(self) -> Provenance:
        """VERIFIED when supported by the evidence, else UNSUPPORTED."""
        return Provenance.VERIFIED if self.supported else Provenance.UNSUPPORTED


@dataclass(frozen=True)
class GroundingReport:
    """Per-sentence grounding of an answer against its sources."""

    sentences: list[SentenceVerdict]

    @property
    def supported_ratio(self) -> float:
        if not self.sentences:
            return 1.0
        return sum(v.supported for v in self.sentences) / len(self.sentences)

    @property
    def unsupported(self) -> list[SentenceVerdict]:
        return [v for v in self.sentences if not v.supported]

    @property
    def faithful(self) -> bool:
        """True iff every sentence is backed by the evidence."""
        return not self.unsupported

    def flagged(self, mark: str = " [UNSUPPORTED]") -> str:
        """The answer with unsupported sentences marked inline."""
        return " ".join(
            v.sentence + ("" if v.supported else mark) for v in self.sentences
        )

    def redacted(self) -> str:
        """The answer with unsupported sentences removed."""
        return " ".join(v.sentence for v in self.sentences if v.supported)


def ground_text(
    answer: str, sources: list[str], *, threshold: float = 0.6
) -> GroundingReport:
    """Check each sentence of ``answer`` against ``sources``.

    Args:
        answer: The model's free-text answer.
        sources: The retrieved passages it was supposed to rely on.
        threshold: Minimum content-word overlap for a sentence to count as
            supported.
    """
    verdicts: list[SentenceVerdict] = []
    for sentence in split_sentences(answer):
        best_score, best_index = 0.0, None
        for i, source in enumerate(sources):
            score = support_score(sentence, source)
            if score > best_score:
                best_score, best_index = score, i
        supported = best_score >= threshold
        verdicts.append(
            SentenceVerdict(
                sentence=sentence,
                supported=supported,
                score=round(best_score, 3),
                source_index=best_index if supported else None,
            )
        )
    return GroundingReport(verdicts)
