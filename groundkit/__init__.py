"""groundkit -- deterministic-first grounding so a model can never assert a fact.

Two layers. Layer 1 checks structured claims against an authoritative source
(:func:`ground_record`) and free-text answers against retrieved evidence
(:func:`ground_text`, which checks content-word overlap, that every figure the
answer asserts appears in the evidence, and that it does not flip the evidence's
polarity); it is exact and the model never
overrules it. Layer 2 --
whatever an LLM contributes -- is, by construction, only ever tagged ADVISORY.
Every value carries its :class:`Provenance`, so nothing unverified is ever
silently presented as truth.
"""

from .facts import FactStore, normalise
from .negation import negated_tokens, polarity_conflicts
from .provenance import Grounded, Provenance
from .quantities import extract_quantities, unsupported_quantities
from .text import GroundingReport, SentenceVerdict, ground_text, support_score
from .verify import contradictions, ground_record, trusted, verify_claim

__version__ = "0.3.0"
__all__ = [
    "FactStore",
    "Grounded",
    "GroundingReport",
    "Provenance",
    "SentenceVerdict",
    "__version__",
    "contradictions",
    "extract_quantities",
    "ground_record",
    "ground_text",
    "negated_tokens",
    "normalise",
    "polarity_conflicts",
    "support_score",
    "trusted",
    "unsupported_quantities",
    "verify_claim",
]
