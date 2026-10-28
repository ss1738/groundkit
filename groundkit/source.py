"""Freshness: is the evidence still current? A separate question from grounding.

groundkit's other checks all ask one thing: is this answer supported by the
evidence it was given? This module asks the other thing: is that evidence still
what the authoritative source says? They are different claims and conflating
them produces a confident wrong answer.

The failure is specific. Index a policy saying "refunds within 30 days", answer
from it, then change the policy to 14 days while the old chunk stays in the
index. The answer still matches its retrieved source perfectly, so every
grounding check passes, and the answer is wrong about current policy. No amount
of attribution accuracy detects that, because attribution is not the thing that
broke.

So the two results stay apart. :func:`check_sources` returns a
:class:`FreshnessReport`; :func:`~groundkit.text.ground_text` returns a
:class:`~groundkit.text.GroundingReport`; and
:class:`~groundkit.source.EvidenceReport` holds both without merging them, so a
caller has to look at each. "Supported by this snapshot" and "snapshot still
current" are reported as two answers because they are two answers.

Freshness is established one of two ways, and a descriptor may carry both:

* **revision**, compared against the live source through a resolver callback you
  supply. This is authoritative and costs a lookup.
* **lease** (``fetched_at`` + ``ttl_seconds``), checked locally with no lookup.
  A lease is a deliberate statement that the snapshot is trusted for a period,
  not evidence that it is current.

Anything unverifiable is ``UNKNOWN``, never ``CURRENT``, and
:attr:`FreshnessVerdict.is_current` is true only for ``CURRENT``. Not knowing
fails closed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable


class Freshness(str, Enum):
    """Whether a snapshot still reflects its source."""

    CURRENT = "current"    # revision matched, or a valid lease
    STALE = "stale"        # the live source has moved on
    EXPIRED = "expired"    # the lease elapsed
    UNKNOWN = "unknown"    # not checkable; never treated as current


def digest(content: str) -> str:
    """A stable revision hash for a chunk of text: ``sha256:<hex>``."""
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


def _parse_time(value: str) -> datetime:
    """Parse an ISO 8601 timestamp, assuming UTC when no offset is given."""
    text = value.strip()
    if text.endswith(("Z", "z")):
        # fromisoformat does not accept the military 'Z' before Python 3.11.
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"fetched_at is not an ISO 8601 timestamp: {value!r}") from exc
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


@dataclass(frozen=True)
class SourceDescriptor:
    """What a retrieved chunk came from, enough to tell whether it still holds.

    Args:
        uri: Identifier of the authoritative source, passed to the resolver.
        revision: Revision of the content at fetch time, such as
            :func:`digest` of the chunk, an ETag or a version id.
        fetched_at: ISO 8601 timestamp of the fetch.
        ttl_seconds: How long the snapshot is trusted without rechecking.
    """

    uri: str
    revision: str | None = None
    fetched_at: str | None = None
    ttl_seconds: int | None = None

    def __post_init__(self) -> None:
        if not self.uri:
            raise ValueError("uri must not be empty")
        if self.ttl_seconds is not None:
            if self.ttl_seconds < 0:
                raise ValueError("ttl_seconds must not be negative")
            if self.fetched_at is None:
                raise ValueError("ttl_seconds requires fetched_at")
        if self.fetched_at is not None:
            _parse_time(self.fetched_at)  # fail at construction, not at check time

    def age_seconds(self, now: datetime | None = None) -> float | None:
        """Seconds since ``fetched_at``, or None when it was not recorded."""
        if self.fetched_at is None:
            return None
        moment = now or datetime.now(timezone.utc)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return (moment - _parse_time(self.fetched_at)).total_seconds()


@dataclass(frozen=True)
class FreshnessVerdict:
    """One source's freshness, with the reason and the authority attached."""

    source: SourceDescriptor
    freshness: Freshness
    note: str | None = None
    live_revision: str | None = None
    authority: str | None = None
    """What the resolver actually spoke for.

    CURRENT is only ever as strong as this. A resolver pointed at your search
    index answers "unchanged in the index", which is a different claim from
    "unchanged at the source of truth": an index holding a stale copy reports
    CURRENT quite correctly, because nothing in the index has changed. Name the
    authority so the distinction survives into the output.
    """

    @property
    def is_current(self) -> bool:
        """True only for CURRENT. UNKNOWN is not current."""
        return self.freshness is Freshness.CURRENT


@dataclass(frozen=True)
class FreshnessReport:
    """Freshness across every source behind an answer."""

    verdicts: list[FreshnessVerdict]

    @property
    def all_current(self) -> bool:
        """True only when every source is positively known to be current."""
        return bool(self.verdicts) and all(v.is_current for v in self.verdicts)

    def _of(self, kind: Freshness) -> list[FreshnessVerdict]:
        return [v for v in self.verdicts if v.freshness is kind]

    @property
    def stale(self) -> list[FreshnessVerdict]:
        return self._of(Freshness.STALE)

    @property
    def expired(self) -> list[FreshnessVerdict]:
        return self._of(Freshness.EXPIRED)

    @property
    def unknown(self) -> list[FreshnessVerdict]:
        return self._of(Freshness.UNKNOWN)

    @property
    def problems(self) -> list[FreshnessVerdict]:
        """Everything not positively current."""
        return [v for v in self.verdicts if not v.is_current]


def check_source(
    source: SourceDescriptor,
    resolver: Callable[[str], str | None] | None = None,
    *,
    now: datetime | None = None,
    authority: str | None = None,
) -> FreshnessVerdict:
    """Decide whether one snapshot still reflects its source.

    Args:
        source: The descriptor recorded at fetch time.
        resolver: ``uri -> current revision``, or None to skip the lookup.
            Returning None means the resolver could not say, which is UNKNOWN,
            not current. An exception is caught and reported the same way, so a
            failing lookup degrades rather than taking the request down.
        now: Override the clock, for testing.
        authority: What ``resolver`` speaks for, recorded on the verdict. A
            resolver pointed at a search index can only ever report "unchanged
            in the index"; say so here rather than letting CURRENT be read as
            "unchanged at the source of truth".
    """
    moment = now or datetime.now(timezone.utc)

    def verdict(freshness, note=None, live=None):
        return FreshnessVerdict(source, freshness, note, live, authority)

    if source.ttl_seconds is not None:
        age = source.age_seconds(moment)
        if age is not None and age > source.ttl_seconds:
            return verdict(Freshness.EXPIRED, note=f"lease of {source.ttl_seconds}s elapsed {age:.0f}s after fetch")

    if resolver is not None and source.revision is not None:
        try:
            live = resolver(source.uri)
        except Exception as exc:  # noqa: BLE001 - a caller's resolver may
            # raise anything (network, auth, driver); a failed lookup must
            # degrade to UNKNOWN rather than take the request down.
            return verdict(Freshness.UNKNOWN, note=f"resolver raised {type(exc).__name__}: {exc}")
        if live is None:
            return verdict(Freshness.UNKNOWN, note="resolver returned no revision for this uri")
        if live != source.revision:
            return verdict(
                Freshness.STALE,
                note=f"indexed revision {source.revision}, source now {live}",
                live=live,
            )
        return verdict(Freshness.CURRENT, note="revision matches the source", live=live)

    if source.ttl_seconds is not None:
        return verdict(Freshness.CURRENT, note="within its lease; revision was not checked")

    if source.revision is not None:
        return verdict(Freshness.UNKNOWN, note="revision recorded but no resolver was supplied to compare it")
    return verdict(Freshness.UNKNOWN, note="no revision and no lease recorded; nothing to check")


def check_sources(
    sources: Iterable[SourceDescriptor],
    resolver: Callable[[str], str | None] | None = None,
    *,
    now: datetime | None = None,
    authority: str | None = None,
) -> FreshnessReport:
    """Check every source behind an answer. See :func:`check_source`."""
    return FreshnessReport(
        [check_source(s, resolver, now=now, authority=authority) for s in sources]
    )


@dataclass(frozen=True)
class EvidenceReport:
    """Grounding and freshness side by side, deliberately not merged.

    ``trustworthy`` requires both: an answer faithful to current evidence. Read
    ``grounding`` and ``freshness`` separately to tell the two failures apart,
    because they call for different fixes. A grounding failure means the model
    misused its evidence. A freshness failure means the pipeline served evidence
    that no longer holds, and the model may have done nothing wrong.
    """

    grounding: object  # GroundingReport; typed loosely to avoid a cycle
    freshness: FreshnessReport

    @property
    def trustworthy(self) -> bool:
        return bool(self.grounding.faithful) and self.freshness.all_current
