"""Grounding free-text answers against retrieved evidence (RAG faithfulness).

Splits an answer into sentences and checks each one against the provided source
passages: a sentence is supported when enough of its content words appear in some
source. Unsupported sentences are flagged, never trusted.

Two deterministic checks, no model and no dependency. Content-word overlap flags
sentences whose substance is largely absent from the evidence, the common RAG
failure. Quantity parity (:mod:`groundkit.quantities`) then requires every figure
the sentence asserts to appear in the evidence, which closes the hole overlap
alone leaves: "within 30 days" against a source saying "within 14 days" shares
every word but the digits and scores 0.875.

A third check, polarity parity (:mod:`groundkit.negation`), catches a sentence
that flips what the evidence says. That is the highest-scoring failure of all:
an answer that drops a "not" matches every word the source contains and scores
1.000.

None of the three is entailment. An antonym carrying no negation cue
("increased" against "decreased") still passes, as does anything needing
paraphrase-level inference, so use an NLI model where that matters. The
structured layer in :mod:`groundkit.verify` remains the exact check for values you
can extract into fields.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .negation import polarity_conflicts
from .provenance import Provenance
from .quantities import unsupported_quantities
from .tokens import content_tokens as _content_tokens


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
    reason: str | None = None  # why it failed, when it did

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
    answer: str,
    sources: list[str],
    *,
    threshold: float = 0.6,
    check_quantities: bool = True,
    check_polarity: bool = True,
) -> GroundingReport:
    """Check each sentence of ``answer`` against ``sources``.

    Three checks, all deterministic. Content-word overlap catches a sentence that
    introduces material the evidence does not contain. Quantity parity catches a
    sentence that reuses the evidence's wording but changes a figure. Polarity
    parity catches a sentence that flips what the evidence says, which overlap
    scores highest of all: an answer dropping a "not" matches every word the
    source has.

    Args:
        answer: The model's free-text answer.
        sources: The retrieved passages it was supposed to rely on.
        threshold: Minimum content-word overlap for a sentence to count as
            supported.
        check_quantities: Require every number, duration, percentage and
            currency amount the sentence asserts to appear somewhere in
            ``sources``. See :mod:`groundkit.quantities` for what is and is not
            canonicalised.
        check_polarity: Flag a sentence that negates what the evidence affirms,
            or affirms what it negates. See :mod:`groundkit.negation` for the
            cue-and-scope rules and their limits.
    """
    verdicts: list[SentenceVerdict] = []
    for sentence in split_sentences(answer):
        best_score, best_index = 0.0, None
        for i, source in enumerate(sources):
            score = support_score(sentence, source)
            if score > best_score:
                best_score, best_index = score, i
        supported = best_score >= threshold
        reason = None if supported else "content words absent from the evidence"
        if supported and check_quantities:
            missing = unsupported_quantities(sentence, sources)
            if missing:
                supported = False
                reason = "quantity absent from the evidence: " + ", ".join(
                    sorted(f"{v} {u}".strip() for v, u in missing)
                )
        if supported and check_polarity:
            flipped = polarity_conflicts(sentence, sources)
            if flipped:
                supported = False
                named = sorted(flipped)
                shown = ", ".join(named[:3])
                if len(named) > 3:
                    shown += f", +{len(named) - 3} more"
                reason = f"polarity differs from the evidence: {shown}"
        verdicts.append(
            SentenceVerdict(
                sentence=sentence,
                supported=supported,
                score=round(best_score, 3),
                source_index=best_index if supported else None,
                reason=reason,
            )
        )
    return GroundingReport(verdicts)
