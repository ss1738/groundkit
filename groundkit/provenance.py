"""Provenance tags -- every grounded value says where it came from.

The whole point of groundkit is that a value is never just a value; it carries
*how much you should trust it*. An authoritative lookup is VERIFIED; a model's
unbacked claim is ADVISORY (kept, but clearly not fact); a claim the source
contradicts is CONTRADICTED (and the authoritative value wins); an answer
sentence with no supporting evidence is UNSUPPORTED.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .source import SourceDescriptor


class Provenance(str, Enum):
    """Where a value's authority comes from."""

    VERIFIED = "verified"          # confirmed against an authoritative source
    CONTRADICTED = "contradicted"  # source disagrees; the source value is kept
    ADVISORY = "advisory"          # model claim with no source to check it
    UNSUPPORTED = "unsupported"    # answer text not backed by provided evidence


@dataclass(frozen=True)
class Grounded:
    """A value plus its provenance.

    For CONTRADICTED, ``value`` is always the *authoritative* value -- the model's
    rejected claim is recorded in ``note``, never surfaced as fact.
    """

    value: Any
    provenance: Provenance
    source: str | SourceDescriptor | None = None
    note: str | None = None

    @property
    def is_trustworthy(self) -> bool:
        """True only for VERIFIED values -- the ones safe to present as fact."""
        return self.provenance is Provenance.VERIFIED
