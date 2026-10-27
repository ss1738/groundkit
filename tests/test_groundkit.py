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
from groundkit.negation import negated_tokens, polarity_conflicts


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

    # Against the *current* policy the quantity check does catch it, even though
    # the lexical score (0.875) still clears the threshold. Freshness is the gap;
    # a swapped number inside a current snapshot is not.
    current = "Customers may request a refund within 14 days of purchase."
    caught = ground_text(answer, [current]).sentences[0]
    assert caught.score >= 0.6
    assert not caught.supported
    assert "quantity" in caught.reason

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


# --- Quantity parity --------------------------------------------------------


def test_quantity_swap_is_caught_despite_high_overlap():
    """The failure content-word overlap cannot see: same words, wrong figure."""
    src = ["Customers may request a refund within 14 days of purchase."]
    report = ground_text("Customers may request a refund within 30 days of purchase.", src)
    v = report.sentences[0]
    assert v.score >= 0.6           # overlap alone would have passed it
    assert not v.supported
    assert v.provenance is Provenance.UNSUPPORTED
    assert "30 day" in v.reason


def test_exact_duration_conversions_are_equivalent():
    """Converting week -> days stops the check false-flagging equivalent wording.

    The quantity check only ever adds strictness; it cannot rescue a sentence
    whose content words already miss the evidence, so these use phrasings with
    enough overlap to reach the quantity stage.
    """
    base = "Customers may request a refund within {} of purchase."
    assert ground_text(base.format("two weeks"), [base.format("14 days")]).faithful
    assert ground_text(
        base.format("14 days"), [base.format("a two week period")]
    ).faithful
    # A month is not a fixed number of days, so it is never converted to one.
    bad = ground_text(base.format("30 days"), [base.format("one month")])
    assert not bad.faithful
    assert "30 day" in bad.sentences[0].reason


def test_quantity_check_only_adds_strictness():
    """A sentence failing on overlap is not rescued by matching quantities."""
    v = ground_text("You have two weeks.", ["You have 14 days."]).sentences[0]
    assert not v.supported
    assert v.reason == "content words absent from the evidence"


def test_quantity_check_covers_money_percent_and_dose():
    assert not ground_text("The rate is 5%.", ["The rate is 7%."]).faithful
    assert ground_text("The rate is 5%.", ["The rate is 5 percent."]).faithful
    assert not ground_text("A fee of £1,250.", ["A fee of £1,500."]).faithful
    assert not ground_text("Take 500 mg daily.", ["Take 250 mg daily."]).faithful


def test_quantity_check_does_not_false_flag():
    sources = [
        "The Eiffel Tower was completed in 1889.",
        "It stands 330 m tall in Paris.",
    ]
    # Figures present across the evidence set, not just the best-matching passage.
    assert ground_text("The Eiffel Tower was completed in 1889.", sources).faithful
    assert ground_text("It stands 330 m tall.", sources).faithful
    # A bare number matches the same value carrying a unit in the source.
    assert ground_text("The window is 14.", ["The window is 14 days."]).faithful
    # Sentences with no quantities are unaffected.
    assert ground_text("It stands tall in Paris.", sources).faithful


def test_quantity_check_can_be_disabled():
    src = ["Customers may request a refund within 14 days of purchase."]
    answer = "Customers may request a refund within 30 days of purchase."
    assert not ground_text(answer, src).faithful
    assert ground_text(answer, src, check_quantities=False).faithful


def test_verdict_reason_is_set_for_both_failure_modes():
    r = ground_text(
        "The fee is 5%. Giraffes outnumber lampposts in Oslo.",
        ["The fee is 7%."],
    )
    assert "quantity" in r.sentences[0].reason
    assert "content words" in r.sentences[1].reason
    assert ground_text("The fee is 7%.", ["The fee is 7%."]).sentences[0].reason is None


# --- Polarity parity --------------------------------------------------------


def test_polarity_flip_is_caught_at_a_perfect_overlap_score():
    """The worst overlap failure: dropping a "not" scores 1.0."""
    src = ["Refunds are not available after 14 days."]
    v = ground_text("Refunds are available after 14 days.", src).sentences[0]
    assert v.score == 1.0            # every answer word is in the evidence
    assert not v.supported
    assert v.provenance is Provenance.UNSUPPORTED
    assert "polarity" in v.reason


def test_polarity_catches_both_directions():
    affirmed = ["The drug is approved for children."]
    negated = ["The drug is not approved for children."]
    assert not ground_text("The drug is not approved for children.", affirmed).faithful
    assert not ground_text("The drug is approved for children.", negated).faithful
    # Agreeing polarity is left alone, in either direction.
    assert ground_text("The drug is approved for children.", affirmed).faithful
    assert ground_text("The drug is not approved for children.", negated).faithful


def test_polarity_handles_contractions_without_tripping_on_can():
    assert not ground_text(
        "The policy doesn't cover flood damage.", ["The policy covers flood damage."]
    ).faithful
    # A bare "can" must not read as "can't".
    assert ground_text(
        "You can request a refund within 30 days.",
        ["You can request a refund within 30 days."],
    ).faithful


def test_polarity_post_cue_scopes_backwards():
    assert not ground_text("The claim was denied.", ["The claim was approved."]).faithful
    assert ground_text("The claim was denied.", ["The claim was denied."]).faithful


def test_polarity_skips_pseudo_cues_and_respects_terminators():
    assert negated_tokens("Not only does it cover flood, it covers fire.") == set()
    assert "fire" not in negated_tokens("It does not cover flood but it covers fire.")
    assert ground_text(
        "The policy covers flood but not fire.",
        ["The policy covers flood but not fire."],
    ).faithful


def test_polarity_one_agreeing_source_clears_the_conflict():
    sources = [
        "Refunds are not available before 14 days.",
        "Refunds are available after 14 days.",
    ]
    assert ground_text("Refunds are available after 14 days.", sources).faithful


def test_polarity_check_can_be_disabled():
    src = ["Refunds are not available after 14 days."]
    answer = "Refunds are available after 14 days."
    assert not ground_text(answer, src).faithful
    assert ground_text(answer, src, check_polarity=False).faithful


def test_polarity_benchmark_does_not_regress():
    """The labelled set in benchmarks/bench_polarity.py must stay clean."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
    from bench_polarity import FLIPPED, SUPPORTED

    assert all(polarity_conflicts(a, s) for a, s in FLIPPED)
    assert not any(polarity_conflicts(a, s) for a, s in SUPPORTED)
