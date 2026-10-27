"""groundkit -- deterministic-first grounding so a model can never assert a fact.

Two layers. Layer 1 checks structured claims against an authoritative source
(:func:`ground_record`) and free-text answers against retrieved evidence
(:func:`ground_text`); it is exact and the model never overrules it. Layer 2 --
whatever an LLM contributes -- is, by construction, only ever tagged ADVISORY.
Every value carries its :class:`Provenance`, so nothing unverified is ever
silently presented as truth.
"""

from .facts import FactStore, normalise
from .provenance import Grounded, Provenance
from .text import GroundingReport, SentenceVerdict, ground_text, support_score
from .verify import contradictions, ground_record, trusted, verify_claim

__version__ = "0.1.0"
__all__ = [
    "FactStore",
    "Grounded",
    "GroundingReport",
    "Provenance",
    "SentenceVerdict",
    "__version__",
    "contradictions",
    "ground_record",
    "ground_text",
    "normalise",
    "support_score",
    "trusted",
    "verify_claim",
]
