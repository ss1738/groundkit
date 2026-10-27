"""Checking free-text answers against retrieved evidence.

**These checks reject, they do not verify.** The distinction is the most
important thing on this page. When a check fires, it is reliable: the figure
really is absent, the polarity really is reversed. When nothing fires, that is
the absence of a detected problem and not a finding of support. Treat a clean
report as "none of my cheap checks caught anything", never as "this answer is
grounded", and never wire it to auto-accept an answer.

Three deterministic checks run per sentence, no model and no dependency.
Content-word overlap flags sentences whose substance is largely absent from the
evidence, the common RAG failure. Quantity parity
(:mod:`groundkit.quantities`) requires every figure the sentence asserts to
appear in the evidence, closing the hole overlap leaves: "within 30 days"
against a source saying "within 14 days" shares every word but the digits and
scores 0.875. Polarity parity (:mod:`groundkit.negation`) catches a sentence
that flips what the evidence says, which is the highest-scoring failure of all,
since an answer that drops a "not" matches every word the source contains and
scores 1.000.

What gets through, measured against this library:

* **relational and argument-order inversion.** "Acme sued Beta Corp" against a
  source reading "Beta Corp sued Acme" scores **1.000** and passes. Identical
  tokens, identical numbers, no negation cue. So do "the claimant owes the
  insurer" reversed, and "Policy A supersedes Policy B" reversed;
* **antonyms carrying no negation cue**, "increased" against "decreased";
* anything needing paraphrase-level inference.

None of the three is entailment, and no list of rules will make them so. Use an
NLI model where a positive finding of support is required. The structured layer
in :mod:`groundkit.verify` is the exact check for values you can extract into
fields, and is the only part of groundkit that confirms rather than fails to
object.
"""

from __future__ import annotations

from dataclasses import dataclass

from .negation import polarity_conflicts
from .provenance import Provenance
from .quantities import unsupported_quantities
from .sentences import split_sentences
from .tokens import content_tokens as _content_tokens


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
        """UNSUPPORTED when a check fired, else VERIFIED.

        VERIFIED here means "no check objected", which is weaker than the
        :mod:`groundkit.verify` sense of the same tag, where a value was
        confirmed against an authoritative record.
        """
        return Provenance.VERIFIED if self.supported else Provenance.UNSUPPORTED


@dataclass(frozen=True)
class GroundingReport:
    """Per-sentence grounding of an answer against its sources."""

    sentences: list[SentenceVerdict]

    @property
    def supported_ratio(self) -> float:
        """Fraction of sentences nothing flagged. An upper bound on faithfulness."""
        if not self.sentences:
            return 1.0
        return sum(v.supported for v in self.sentences) / len(self.sentences)

    @property
    def unsupported(self) -> list[SentenceVerdict]:
        return [v for v in self.sentences if not v.supported]

    @property
    def faithful(self) -> bool:
        """True when no check flagged any sentence.

        This is the absence of a detected problem, not a positive finding that
        the answer is supported. A reversed claim with identical wording passes
        it. Use it to route answers for review, not to accept them.
        """
        return not self.unsupported

    def flagged(self, mark: str = " [UNSUPPORTED]") -> str:
        """The answer with unsupported sentences marked inline."""
        return " ".join(
            v.sentence + ("" if v.supported else mark) for v in self.sentences
        )

    def redacted(self) -> str:
        """The answer with flagged sentences removed.

        What remains is the text nothing objected to, which is **not** the same
        as the text that is safe to show. Sentences these checks cannot see
        through, a reversed relation or a swapped antonym, survive redaction
        intact. Do not present the result as verified content.
        """
        return " ".join(v.sentence for v in self.sentences if v.supported)


def _nearest(sentence: str, evidence_sentences: list[str], k: int = 3) -> list[str]:
    """The ``k`` evidence sentences closest to ``sentence``, for polarity scoping.

    Polarity is a property of a clause, not of a document. Comparing against a
    whole passage lets one negated clause anywhere in it poison every answer
    sentence that reuses a word from it: a 2 kB passage containing "it is not
    clear where she was arrested" made every later sentence mentioning "she"
    look like a polarity conflict. Measured on RAGTruth, whole-document
    comparison fired on 30.4% of sentences at precision 0.070 against a base
    rate of 0.087, which is worse than firing at random. Scoping to the nearest
    few evidence sentences drops that to 9.6% at precision 0.102.
    """
    if not evidence_sentences:
        return []
    return sorted(
        evidence_sentences, key=lambda s: -support_score(sentence, s)
    )[:k]


def ground_text(
    answer: str,
    sources: list[str],
    *,
    threshold: float = 0.3,
    check_quantities: bool = True,
    check_polarity: bool = False,
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
            supported. Default 0.3, chosen by sweeping RAGTruth: lift over the
            base rate falls monotonically as the threshold rises, because a
            higher bar flags more correct sentences than wrong ones. At 0.6 it
            flagged 69.4% of all sentences for a lift of +0.025; at 0.3 it
            flags 33.1% for +0.064. Raise it for short, quoted evidence; lower
            it for long passages.
        check_quantities: Require every number, duration, percentage and
            currency amount the sentence asserts to appear somewhere in
            ``sources``. See :mod:`groundkit.quantities` for what is and is not
            canonicalised.
        check_polarity: Flag a sentence that negates what the evidence affirms,
            or affirms what it negates. **Off by default**, on measurement:
            scoped to the nearest evidence sentences it reaches precision 0.102
            against a RAGTruth base rate of 0.087, a lift of only +0.015, and
            adding it to any other combination of checks lowered that
            combination's lift. It earns its place on short, single-claim
            evidence, which is the regime it was built and tested in. Turn it
            on there, and measure before trusting it on long passages.
    """
    verdicts: list[SentenceVerdict] = []
    evidence_sentences = [t for src in sources for t in split_sentences(src)]
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
            flipped = polarity_conflicts(sentence, _nearest(sentence, evidence_sentences))
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
