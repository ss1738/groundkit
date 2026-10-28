"""Weaviate adapter: retrieved objects in, grounding and freshness out.

Duck-typed on purpose. Nothing here imports ``weaviate``, so groundkit keeps its
zero dependencies and this module works against a real client, a stub, or a
recorded fixture. It reads the shape the v4 Python client returns:

* ``QueryReturn.objects``, or a plain list of objects;
* ``Object.uuid``, ``Object.collection``, ``Object.properties`` (a mapping);
* ``Object.metadata.last_update_time`` (a datetime, or None when the query did
  not ask for it).

The point is that Weaviate already knows the two things a freshness check needs.
``uuid`` identifies the chunk and ``last_update_time`` changes whenever it is
updated, so a revision is available without you tracking one yourself. Build
passages at retrieval time, and later ask Weaviate whether those objects have
moved on.

    passages = to_passages(response, text_property="content")
    report = ground_weaviate(answer, passages,
                             resolver=revision_resolver(collection))

    report.grounding.faithful     # is the answer supported by those chunks?
    report.freshness.all_current  # are those chunks still what Weaviate holds?
    report.trustworthy            # both

Those stay two results, as everywhere else in groundkit. An answer can be
perfectly faithful to a chunk the collection has since changed, and that is a
pipeline failure, not a model failure.

**What "current" means here, and what it does not.** :func:`revision_resolver`
asks Weaviate, so it answers *"unchanged in the index"*. That is not the same
claim as *"unchanged at the source of truth"*, and the gap between them is
exactly where a stale index lives. If the authoritative policy changes from 30
days to 14 while the indexed chunk still says 30, this adapter reports CURRENT
and ``trustworthy`` comes back True: nothing in Weaviate has changed, and the
adapter is answering the question it was asked. The answer is still wrong about
current policy.

Pointing the resolver at the authoritative store instead reports STALE for the
same answer and chunk. So there are two separate claims worth keeping visible,
and :attr:`~groundkit.source.FreshnessVerdict.authority` records which one you
got. A TTL lease is a third thing again: a decision to trust the snapshot for a
period, not an observation that nothing changed. Credit to RateLedger on the
Weaviate forum for drawing this boundary.

A note on metadata: ``last_update_time`` is only populated when the query asks
for it. Rather than silently falling back to a content hash, which would make
every later freshness check disagree with itself, :func:`to_passages` raises and
tells you what to add to the query. Use ``revision_from="content"`` if you would
rather hash the text and not request metadata at all.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from .source import EvidenceReport, SourceDescriptor, check_sources, digest
from .text import ground_text

_UPDATED = "updated:"
_SCHEMES = ("last_update_time", "content")


@dataclass(frozen=True)
class Passage:
    """One retrieved chunk: its text, and where it came from."""

    text: str
    source: SourceDescriptor


def _iter_objects(result: Any) -> list[Any]:
    """Accept a QueryReturn, a list of objects, or a list of Passages."""
    objects = getattr(result, "objects", result)
    if objects is None:
        return []
    return list(objects)


def _text_of(obj: Any, text_property: str) -> str:
    properties = getattr(obj, "properties", None)
    if properties is None:
        raise ValueError("object has no .properties; is this a Weaviate object?")
    try:
        value = properties[text_property]
    except (KeyError, TypeError):
        available = ", ".join(sorted(map(str, properties))) or "none"
        raise ValueError(
            f"property {text_property!r} not found on the object; "
            f"available properties: {available}. Pass text_property= to choose."
        ) from None
    if value is None:
        raise ValueError(f"property {text_property!r} is None on this object")
    return str(value)


def _iso(moment: datetime) -> str:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat()


def _uri(obj: Any) -> str:
    uuid = getattr(obj, "uuid", None)
    if uuid is None:
        raise ValueError("object has no .uuid; cannot identify it for freshness")
    collection = getattr(obj, "collection", None) or "object"
    return f"weaviate://{collection}/{uuid}"


def _revision(obj: Any, text: str, revision_from: str) -> str:
    if revision_from == "content":
        return digest(text)
    metadata = getattr(obj, "metadata", None)
    stamp = getattr(metadata, "last_update_time", None)
    if stamp is None:
        raise ValueError(
            "last_update_time is missing: the query did not request it. Add "
            "return_metadata=wvc.query.MetadataQuery(last_update_time=True) to "
            'the search, or pass revision_from="content" to hash the text '
            "instead. Falling back silently would make later freshness checks "
            "compare two different kinds of revision and always report STALE."
        )
    return _UPDATED + _iso(stamp)


def to_passages(
    result: Any,
    *,
    text_property: str = "text",
    revision_from: str = "last_update_time",
    fetched_at: datetime | None = None,
    ttl_seconds: int | None = None,
) -> list[Passage]:
    """Turn a Weaviate query result into passages carrying their provenance.

    Args:
        result: A ``QueryReturn``, or any iterable of Weaviate objects.
        text_property: Which property holds the chunk text.
        revision_from: ``"last_update_time"`` to use Weaviate's own update
            timestamp, or ``"content"`` to hash the text. Whichever you pick
            here must match the resolver used later.
        fetched_at: Retrieval time, defaulting to now.
        ttl_seconds: Optional lease, after which the snapshot is EXPIRED
            without consulting the source at all.
    """
    if revision_from not in _SCHEMES:
        raise ValueError(f"revision_from must be one of {_SCHEMES}")
    moment = fetched_at or datetime.now(timezone.utc)
    passages = []
    for obj in _iter_objects(result):
        text = _text_of(obj, text_property)
        passages.append(Passage(
            text=text,
            source=SourceDescriptor(
                uri=_uri(obj),
                revision=_revision(obj, text, revision_from),
                fetched_at=_iso(moment),
                ttl_seconds=ttl_seconds,
            ),
        ))
    return passages


def texts(passages: Iterable[Passage]) -> list[str]:
    """Just the chunk texts, for :func:`~groundkit.text.ground_text`."""
    return [p.text for p in passages]


def sources(passages: Iterable[Passage]) -> list[SourceDescriptor]:
    """Just the descriptors, for :func:`~groundkit.source.check_sources`."""
    return [p.source for p in passages]


def revision_resolver(
    collection: Any,
    *,
    revision_from: str = "last_update_time",
    text_property: str = "text",
) -> Callable[[str], str | None]:
    """Build a ``uri -> current revision`` resolver backed by a live collection.

    This resolves against **Weaviate**, so it answers "has the indexed object
    changed?" and not "has the source of truth changed?". An index holding a
    stale copy reports CURRENT, correctly and unhelpfully. Point a resolver at
    your authoritative store if you need the second claim.

    ``collection`` is anything exposing ``.query.fetch_object_by_id(uuid)``, so a
    real Weaviate collection or a stub. ``revision_from`` must match what
    :func:`to_passages` used, or every check reports STALE.

    A deleted object resolves to ``"deleted"``, which differs from any stored
    revision and so reports STALE, since a source that no longer exists is not
    current. A lookup that raises is caught upstream by
    :func:`~groundkit.source.check_source` and reported as UNKNOWN.
    """
    if revision_from not in _SCHEMES:
        raise ValueError(f"revision_from must be one of {_SCHEMES}")

    def resolve(uri: str) -> str | None:
        obj = collection.query.fetch_object_by_id(uri.rsplit("/", 1)[-1])
        if obj is None:
            return "deleted"
        if revision_from == "content":
            return digest(_text_of(obj, text_property))
        stamp = getattr(getattr(obj, "metadata", None), "last_update_time", None)
        if stamp is None:
            return None  # cannot say, so UNKNOWN rather than a false STALE
        return _UPDATED + _iso(stamp)

    return resolve


def ground_weaviate(
    answer: str,
    passages: Sequence[Passage],
    *,
    resolver: Callable[[str], str | None] | None = None,
    now: datetime | None = None,
    authority: str | None = "weaviate index",
    **ground_kwargs: Any,
) -> EvidenceReport:
    """Ground an answer against retrieved passages and check they are current.

    Returns both results side by side. ``trustworthy`` needs both; read
    ``grounding`` and ``freshness`` separately to tell the two failures apart.

    ``authority`` records what the resolver speaks for, and defaults to the
    index because :func:`revision_resolver` asks Weaviate. Pass the name of your
    source of truth when the resolver consults one, so CURRENT is not read as a
    stronger claim than it is. See the module docstring.
    """
    return EvidenceReport(
        grounding=ground_text(answer, texts(passages), **ground_kwargs),
        freshness=check_sources(
            sources(passages), resolver, now=now, authority=authority
        ),
    )
