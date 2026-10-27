"""Tests for groundkit's two grounding layers."""

from groundkit import (
    FactStore,
    Provenance,
    contradictions,
    ground_record,
    ground_text,
    trusted,
    verify_claim,
)


def _store():
    return FactStore({"capital": "Paris", "ceo": "Jane Doe", "founded": "1990"})


def test_verified_claim():
    g = verify_claim(_store(), "capital", "Paris")
    assert g.provenance is Provenance.VERIFIED
    assert g.value == "Paris"
    assert g.is_trustworthy


def test_verified_is_normalised():
    g = verify_claim(_store(), "ceo", "  jane   DOE ")
    assert g.provenance is Provenance.VERIFIED


def test_contradiction_keeps_authoritative_value():
    g = verify_claim(_store(), "capital", "Lyon")
    assert g.provenance is Provenance.CONTRADICTED
    assert g.value == "Paris"          # source wins, never the model's "Lyon"
    assert "Lyon" in g.note
    assert not g.is_trustworthy


def test_unknown_key_is_advisory():
    g = verify_claim(_store(), "revenue", "1B")
    assert g.provenance is Provenance.ADVISORY
    assert g.value == "1B"             # kept, but flagged as unverified
    assert not g.is_trustworthy


def test_ground_record_and_helpers():
    record = {"capital": "Lyon", "ceo": "Jane Doe", "revenue": "1B"}
    grounded = ground_record(_store(), record)
    assert grounded["capital"].provenance is Provenance.CONTRADICTED
    assert grounded["ceo"].provenance is Provenance.VERIFIED
    assert grounded["revenue"].provenance is Provenance.ADVISORY

    # Only the verified field is safe to present as fact.
    assert trusted(grounded) == {"ceo": "Jane Doe"}
    assert set(contradictions(grounded)) == {"capital"}


def test_no_hallucination_reaches_trusted():
    """Whatever the model gets wrong must never appear in trusted()."""
    store = FactStore({"a": "1", "b": "2", "c": "3"})
    record = {"a": "1", "b": "99", "c": "3", "d": "hallucinated"}
    t = trusted(ground_record(store, record))
    assert t == {"a": "1", "c": "3"}   # wrong "b" and unverifiable "d" excluded


def test_ground_text_supported_and_unsupported():
    sources = [
        "Acme was founded in 1990 in Leeds.",
        "The chief executive of Acme is Jane Doe.",
    ]
    answer = "Acme was founded in 1990. Acme is the largest firm in Europe."
    report = ground_text(answer, sources)
    assert report.sentences[0].supported is True
    assert report.sentences[1].supported is False   # "largest firm in Europe" unbacked
    assert report.sentences[0].provenance is Provenance.VERIFIED
    assert report.sentences[1].provenance is Provenance.UNSUPPORTED
    assert report.faithful is False
    assert len(report.unsupported) == 1
    assert "[UNSUPPORTED]" in report.flagged()
    assert "largest firm" not in report.redacted()


def test_non_latin_sentence_is_not_vacuously_supported():
    # A non-Latin sentence must not pass as supported against unrelated evidence.
    report = ground_text("北京是中国的首都。", ["The capital of France is Paris."])
    assert report.sentences[0].supported is False
    assert report.faithful is False


def test_fully_faithful_answer():
    sources = ["Water boils at 100 degrees celsius at sea level."]
    report = ground_text("Water boils at 100 degrees celsius.", sources)
    assert report.faithful is True
    assert report.supported_ratio == 1.0


def test_empty_answer_is_vacuously_faithful():
    report = ground_text("", ["anything"])
    assert report.faithful is True
    assert report.supported_ratio == 1.0


def test_stale_snapshot_still_verifies():
    """Grounding is not freshness: a stale snapshot still verifies.

    Pins the documented limitation in README "Scope: grounding is not freshness".
    An answer that matches its retrieved chunk perfectly is VERIFIED even when the
    authoritative source has since changed, because nothing records a revision or
    a fetched-at time. If this ever stops being true, the README must change too.
    """
    stale_chunk = "Customers may request a refund within 30 days of purchase."
    answer = "Customers may request a refund within 30 days of purchase."

    report = ground_text(answer, [stale_chunk])
    assert report.faithful
    assert report.sentences[0].score == 1.0
    assert report.sentences[0].provenance is Provenance.VERIFIED

    # Same answer against the *current* policy: a single swapped number still
    # clears the lexical threshold, so the text layer cannot catch it either.
    current = "Customers may request a refund within 14 days of purchase."
    assert ground_text(answer, [current]).sentences[0].supported

    # A store loaded from stale data is equally unaware it is stale.
    assert verify_claim(FactStore({"refund_window_days": 30}),
                        "refund_window_days", 30).is_trustworthy

    # The structured layer does catch the wrong value within a current snapshot.
    g = verify_claim(FactStore({"refund_window_days": 14}), "refund_window_days", 30)
    assert g.provenance is Provenance.CONTRADICTED
    assert g.value == 14


def test_no_freshness_api_is_advertised():
    """No public name promises revision/staleness handling."""
    import groundkit

    names = [n for n in dir(groundkit) if not n.startswith("_")]
    assert not [
        n for n in names
        if any(k in n.lower() for k in ("revision", "stale", "fresh", "expire"))
    ]
