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
    assert report.faithful is False
    assert len(report.unsupported) == 1
    assert "[UNSUPPORTED]" in report.flagged()
    assert "largest firm" not in report.redacted()


def test_fully_faithful_answer():
    sources = ["Water boils at 100 degrees celsius at sea level."]
    report = ground_text("Water boils at 100 degrees celsius.", sources)
    assert report.faithful is True
    assert report.supported_ratio == 1.0


def test_empty_answer_is_vacuously_faithful():
    report = ground_text("", ["anything"])
    assert report.faithful is True
    assert report.supported_ratio == 1.0
