"""Grounding structured model output against the authoritative source.

Layer 1 (this module) decides every field deterministically; the model never gets
to overrule the source. A field the model extracted is:

* VERIFIED     -- it matches the source;
* CONTRADICTED -- the source disagrees (the *source* value is kept, not the model's);
* ADVISORY     -- the source has no record, so it's an unverified model claim.

Nothing the model says is silently promoted to fact. Whatever an LLM adds as
Layer 2 is, by construction, only ever ADVISORY.
"""

from __future__ import annotations

from typing import Any

from .facts import FactStore, normalise
from .provenance import Grounded, Provenance
from .source import SourceDescriptor


def verify_claim(
    store: FactStore,
    key: str,
    value: Any,
    *,
    source: str | SourceDescriptor | None = None,
) -> Grounded:
    """Ground a single ``key = value`` claim against the store.

    ``source`` labels where the authority came from. Left unset it falls back to
    the store's own :class:`~groundkit.source.SourceDescriptor`, or to the string
    ``"authoritative"`` when the store has none. Note this records *which*
    snapshot answered, and says nothing about whether that snapshot is still
    current: check it with :func:`~groundkit.source.check_sources`.
    """
    if source is None:
        source = store.source if store.source is not None else "authoritative"
    if not store.has(key):
        return Grounded(
            value,
            Provenance.ADVISORY,
            source=None,
            note="no authoritative record; unverified model claim",
        )
    truth = store.get(key)
    if normalise(truth) == normalise(value):
        return Grounded(truth, Provenance.VERIFIED, source=source)
    return Grounded(
        truth,  # the authoritative value wins -- never the model's
        Provenance.CONTRADICTED,
        source=source,
        note=f"model claimed {value!r}; source says {truth!r}",
    )


def ground_record(
    store: FactStore,
    record: dict[str, Any],
    *,
    source: str | SourceDescriptor | None = None,
) -> dict[str, Grounded]:
    """Ground every field of a model-extracted record."""
    return {k: verify_claim(store, k, v, source=source) for k, v in record.items()}


def trusted(grounded: dict[str, Grounded]) -> dict[str, Any]:
    """The subset safe to present as fact -- VERIFIED values only."""
    return {k: g.value for k, g in grounded.items() if g.is_trustworthy}


def contradictions(grounded: dict[str, Grounded]) -> dict[str, Grounded]:
    """Fields where the model disagreed with the source."""
    return {
        k: g for k, g in grounded.items() if g.provenance is Provenance.CONTRADICTED
    }
