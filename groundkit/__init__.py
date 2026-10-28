"""groundkit -- deterministic-first grounding so a model can never assert a fact.

Two layers. Layer 1 checks structured claims against an authoritative source
(:func:`ground_record`) and free-text answers against retrieved evidence
(:func:`ground_text`, which screens content-word overlap, figures asserted, and
polarity; it rejects reliably, but a clean result means only that nothing fired,
not that the answer is supported); it is exact and the model never
overrules it. Layer 2 --
whatever an LLM contributes -- is, by construction, only ever tagged ADVISORY.
:mod:`groundkit.weaviate` adapts a Weaviate v4 query result into both, without
importing the client.

Every value carries its :class:`Provenance`, so nothing unverified is ever
silently presented as truth. :mod:`groundkit.source` answers the separate
question of whether the evidence is still current, and reports it as a
separate result: attribution never carries both claims.
"""

from .facts import FactStore, normalise
from .negation import negated_tokens, polarity_conflicts
from .provenance import Grounded, Provenance
from .quantities import extract_quantities, unsupported_quantities
from .sentences import ABBREVIATIONS, split_sentences
from .source import (
    EvidenceReport,
    Freshness,
    FreshnessReport,
    FreshnessVerdict,
    SourceDescriptor,
    check_source,
    check_sources,
    digest,
)
from .text import GroundingReport, SentenceVerdict, ground_text, support_score
from .verify import contradictions, ground_record, trusted, verify_claim

__version__ = "0.10.0"
__all__ = [
    "ABBREVIATIONS",
    "EvidenceReport",
    "FactStore",
    "Freshness",
    "FreshnessReport",
    "FreshnessVerdict",
    "Grounded",
    "GroundingReport",
    "Provenance",
    "SentenceVerdict",
    "SourceDescriptor",
    "__version__",
    "check_source",
    "check_sources",
    "contradictions",
    "digest",
    "extract_quantities",
    "ground_record",
    "ground_text",
    "negated_tokens",
    "normalise",
    "polarity_conflicts",
    "split_sentences",
    "support_score",
    "trusted",
    "unsupported_quantities",
    "verify_claim",
]
