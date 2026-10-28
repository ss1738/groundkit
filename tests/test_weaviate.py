"""Tests for the Weaviate adapter, against stand-ins for the v4 client shapes.

The adapter never imports weaviate, so these dataclasses are a faithful enough
stub: they carry the same attribute names the real Object, MetadataReturn and
QueryReturn expose.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pytest

from groundkit import Freshness
from groundkit.weaviate import (
    Passage,
    ground_weaviate,
    revision_resolver,
    sources,
    texts,
    to_passages,
)

OLD = "Customers may request a refund within 30 days of purchase."
NEW = "Customers may request a refund within 14 days of purchase."
T0 = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)
T1 = dt.datetime(2026, 10, 20, tzinfo=dt.timezone.utc)


@dataclass
class Meta:
    last_update_time: dt.datetime | None = None


@dataclass
class Obj:
    uuid: str
    collection: str
    properties: dict
    metadata: Meta


@dataclass
class QueryReturn:
    objects: list


class _Query:
    def __init__(self, store):
        self.store = store

    def fetch_object_by_id(self, uuid):
        return self.store.get(uuid)


class Collection:
    def __init__(self, store):
        self.query = _Query(store)


def _result(text=OLD, stamp=T0, uuid="abc-123"):
    return QueryReturn([Obj(uuid, "Policies", {"text": text}, Meta(stamp))])


def test_passages_carry_uuid_and_update_time_as_provenance():
    p = to_passages(_result(), fetched_at=T0)[0]
    assert p.text == OLD
    assert p.source.uri == "weaviate://Policies/abc-123"
    assert p.source.revision == "updated:2026-09-01T00:00:00+00:00"
    assert p.source.fetched_at.startswith("2026-09-01")


def test_a_changed_chunk_is_stale_while_grounding_still_passes():
    """The scenario the adapter exists for, end to end."""
    passages = to_passages(_result(), fetched_at=T0)
    live = Collection({"abc-123": Obj("abc-123", "Policies", {"text": NEW}, Meta(T1))})

    report = ground_weaviate(OLD, passages, resolver=revision_resolver(live))

    assert report.grounding.faithful         # the answer does match its chunk
    assert not report.freshness.all_current  # but Weaviate has moved on
    assert not report.trustworthy            # so the answer is not trustworthy
    assert report.freshness.stale != []


def test_unchanged_chunk_is_trustworthy():
    passages = to_passages(_result(), fetched_at=T0)
    live = Collection({"abc-123": Obj("abc-123", "Policies", {"text": OLD}, Meta(T0))})
    report = ground_weaviate(OLD, passages, resolver=revision_resolver(live))
    assert report.trustworthy


def test_deleted_object_reads_as_stale():
    passages = to_passages(_result(), fetched_at=T0)
    report = ground_weaviate(OLD, passages, resolver=revision_resolver(Collection({})))
    v = report.freshness.verdicts[0]
    assert v.freshness is Freshness.STALE
    assert v.live_revision == "deleted"


def test_resolver_failure_is_unknown_not_a_false_stale():
    class Broken:
        class query:
            @staticmethod
            def fetch_object_by_id(uuid):
                raise TimeoutError("unreachable")

    passages = to_passages(_result(), fetched_at=T0)
    report = ground_weaviate(OLD, passages, resolver=revision_resolver(Broken))
    assert report.freshness.verdicts[0].freshness is Freshness.UNKNOWN
    assert not report.trustworthy


def test_missing_metadata_raises_rather_than_falling_back():
    """A silent fallback would make every later check report a false STALE."""
    with pytest.raises(ValueError, match="last_update_time is missing"):
        to_passages(_result(stamp=None))
    # The documented escape hatch works and needs no metadata.
    p = to_passages(_result(stamp=None), revision_from="content")[0]
    assert p.source.revision.startswith("sha256:")


def test_content_revision_round_trips_through_its_own_resolver():
    passages = to_passages(_result(stamp=None), revision_from="content", fetched_at=T0)
    same = Collection({"abc-123": Obj("abc-123", "P", {"text": OLD}, Meta(None))})
    changed = Collection({"abc-123": Obj("abc-123", "P", {"text": NEW}, Meta(None))})
    resolve_same = revision_resolver(same, revision_from="content")
    resolve_changed = revision_resolver(changed, revision_from="content")
    assert ground_weaviate(OLD, passages, resolver=resolve_same).freshness.all_current
    assert not ground_weaviate(OLD, passages, resolver=resolve_changed).freshness.all_current


def test_accepts_a_bare_list_as_well_as_a_query_return():
    objs = [Obj("u1", "P", {"text": OLD}, Meta(T0))]
    assert len(to_passages(objs, fetched_at=T0)) == 1
    assert to_passages([], fetched_at=T0) == []


def test_helpful_errors_for_the_wrong_property_or_shape():
    with pytest.raises(ValueError, match="available properties: text"):
        to_passages(_result(), text_property="content")
    with pytest.raises(ValueError, match="no .properties"):
        to_passages([object()])
    with pytest.raises(ValueError, match="revision_from must be"):
        to_passages(_result(), revision_from="nonsense")
    with pytest.raises(ValueError, match="revision_from must be"):
        revision_resolver(Collection({}), revision_from="nonsense")


def test_ttl_lease_expires_without_consulting_weaviate():
    passages = to_passages(_result(), fetched_at=T0, ttl_seconds=3600)
    late = T0 + dt.timedelta(days=1)
    report = ground_weaviate(OLD, passages, now=late)   # no resolver at all
    assert report.freshness.verdicts[0].freshness is Freshness.EXPIRED


def test_texts_and_sources_split_the_passages():
    passages = to_passages(_result(), fetched_at=T0)
    assert texts(passages) == [OLD]
    assert sources(passages) == [passages[0].source]
    assert isinstance(passages[0], Passage)


def test_grounding_kwargs_reach_ground_text():
    """A wrong figure is still caught through the adapter."""
    passages = to_passages(_result(text=NEW), fetched_at=T0)
    assert not ground_weaviate(OLD, passages).grounding.faithful       # 30 vs 14
    assert ground_weaviate(
        OLD, passages, check_quantities=False
    ).grounding.faithful


def test_adapter_does_not_import_weaviate():
    import sys

    assert "weaviate" not in sys.modules or sys.modules["weaviate"].__name__ != "weaviate"
    import groundkit.weaviate as adapter

    assert adapter.__name__ == "groundkit.weaviate"



def test_index_freshness_is_not_authority_freshness():
    """A stale index reports CURRENT, because nothing in the index changed.

    Raised by RateLedger on the Weaviate forum. "Unchanged in Weaviate" and
    "unchanged at the upstream authority" are different claims, and the adapter
    only ever answers the first.
    """
    from groundkit import digest

    indexed = Obj("abc-123", "Policies", {"text": OLD}, Meta(T0))
    passages = to_passages([indexed], fetched_at=T0)
    weaviate = Collection({"abc-123": indexed})      # index never changed

    via_index = ground_weaviate(OLD, passages, resolver=revision_resolver(weaviate))
    assert via_index.freshness.verdicts[0].freshness is Freshness.CURRENT
    assert via_index.trustworthy                      # and the answer is still wrong
    assert via_index.freshness.verdicts[0].authority == "weaviate index"

    # The same answer and chunk, resolved against the source of truth.
    via_authority = ground_weaviate(
        OLD, passages, resolver=lambda uri: digest(NEW), authority="policy store"
    )
    assert via_authority.freshness.verdicts[0].freshness is Freshness.STALE
    assert not via_authority.trustworthy
    assert via_authority.freshness.verdicts[0].authority == "policy store"
