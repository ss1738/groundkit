"""Tests for groundkit's two grounding layers."""

import pytest

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
from groundkit.sentences import split_sentences
from groundkit.source import (
    EvidenceReport,
    Freshness,
    FreshnessReport,
    SourceDescriptor,
    check_source,
    check_sources,
    digest,
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


def test_freshness_is_a_separate_result_from_grounding():
    """The hook exists, and it does not change what grounding reports."""
    import groundkit

    assert hasattr(groundkit, "check_sources")
    assert hasattr(groundkit, "SourceDescriptor")
    # ground_text still answers only "supported by this evidence".
    stale_chunk = "Customers may request a refund within 30 days of purchase."
    report = ground_text(stale_chunk, [stale_chunk])
    assert report.faithful
    assert not hasattr(report, "freshness")


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
    v = ground_text(
        "You have two weeks.", ["You have 14 days."], threshold=0.6
    ).sentences[0]
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
    v = ground_text(
        "Refunds are available after 14 days.", src, check_polarity=True
    ).sentences[0]
    assert v.score == 1.0            # every answer word is in the evidence
    assert not v.supported
    assert v.provenance is Provenance.UNSUPPORTED
    assert "polarity" in v.reason


def test_polarity_catches_both_directions():
    affirmed = ["The drug is approved for children."]
    negated = ["The drug is not approved for children."]
    pol = {"check_polarity": True}
    assert not ground_text("The drug is not approved for children.", affirmed, **pol).faithful
    assert not ground_text("The drug is approved for children.", negated, **pol).faithful
    # Agreeing polarity is left alone, in either direction.
    assert ground_text("The drug is approved for children.", affirmed, **pol).faithful
    assert ground_text("The drug is not approved for children.", negated, **pol).faithful


def test_polarity_handles_contractions_without_tripping_on_can():
    assert not ground_text(
        "The policy doesn't cover flood damage.", ["The policy covers flood damage."],
        check_polarity=True,
    ).faithful
    # A bare "can" must not read as "can't".
    assert ground_text(
        "You can request a refund within 30 days.",
        ["You can request a refund within 30 days."],
    ).faithful


def test_polarity_post_cue_scopes_backwards():
    assert not ground_text(
        "The claim was denied.", ["The claim was approved."], check_polarity=True
    ).faithful
    assert ground_text(
        "The claim was denied.", ["The claim was denied."], check_polarity=True
    ).faithful


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
    assert not ground_text(answer, src, check_polarity=True).faithful
    assert ground_text(answer, src, check_polarity=False).faithful


def test_polarity_benchmark_does_not_regress():
    """The labelled set in benchmarks/bench_polarity.py must stay clean."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
    from bench_polarity import FLIPPED, SUPPORTED

    assert all(polarity_conflicts(a, s) for a, s in FLIPPED)
    assert not any(polarity_conflicts(a, s) for a, s in SUPPORTED)


# --- Freshness --------------------------------------------------------------


_CHUNK = "Customers may request a refund within 30 days of purchase."
_UPDATED = "Customers may request a refund within 14 days of purchase."


def _descriptor(**kw):
    base = {"uri": "policy/refunds", "revision": digest(_CHUNK),
            "fetched_at": "2026-09-01T00:00:00Z"}
    base.update(kw)
    return SourceDescriptor(**base)


def test_stale_snapshot_is_caught_by_the_resolver():
    """The case the grounding checks cannot see, now reported separately."""
    live = {"policy/refunds": digest(_UPDATED)}
    report = check_sources([_descriptor()], resolver=lambda uri: live.get(uri))
    v = report.verdicts[0]
    assert v.freshness is Freshness.STALE
    assert not v.is_current
    assert not report.all_current
    assert report.stale == [v]
    assert v.live_revision == digest(_UPDATED)

    # Meanwhile grounding against that same stale chunk still passes, correctly.
    assert ground_text(_CHUNK, [_CHUNK]).faithful


def test_unchanged_source_is_current():
    live = {"policy/refunds": digest(_CHUNK)}
    report = check_sources([_descriptor()], resolver=lambda uri: live.get(uri))
    assert report.all_current
    assert report.problems == []


def test_unverifiable_is_unknown_never_current():
    # No resolver supplied.
    assert check_source(_descriptor()).freshness is Freshness.UNKNOWN
    # Resolver cannot say.
    assert check_source(_descriptor(), lambda uri: None).freshness is Freshness.UNKNOWN
    # Nothing recorded to check at all.
    bare = SourceDescriptor(uri="policy/refunds")
    assert check_source(bare, lambda uri: "v1").freshness is Freshness.UNKNOWN
    # UNKNOWN must never read as current.
    assert not check_source(bare).is_current
    assert not FreshnessReport([check_source(bare)]).all_current


def test_resolver_failure_degrades_rather_than_raising():
    def broken(uri):
        raise TimeoutError("source unreachable")

    v = check_source(_descriptor(), broken)
    assert v.freshness is Freshness.UNKNOWN
    assert "TimeoutError" in v.note


def test_lease_expiry_is_checked_without_a_resolver():
    from datetime import datetime, timezone

    d = _descriptor(ttl_seconds=3600)
    fresh = datetime(2026, 9, 1, 0, 30, tzinfo=timezone.utc)
    stale = datetime(2026, 9, 2, 0, 0, tzinfo=timezone.utc)
    assert check_source(d, now=fresh).freshness is Freshness.CURRENT
    expired = check_source(d, now=stale)
    assert expired.freshness is Freshness.EXPIRED
    assert "lease" in expired.note
    # An elapsed lease wins over a matching revision: do not serve it unchecked.
    live = {"policy/refunds": digest(_CHUNK)}
    assert check_source(d, lambda u: live.get(u), now=stale).freshness is Freshness.EXPIRED


def test_descriptor_validates_at_construction():
    with pytest.raises(ValueError):
        SourceDescriptor(uri="")
    with pytest.raises(ValueError):
        SourceDescriptor(uri="u", fetched_at="not-a-timestamp")
    with pytest.raises(ValueError):
        SourceDescriptor(uri="u", ttl_seconds=60)            # ttl needs fetched_at
    with pytest.raises(ValueError):
        SourceDescriptor(uri="u", fetched_at="2026-09-01T00:00:00Z", ttl_seconds=-1)
    # Both ISO forms parse, including the 'Z' suffix Python 3.9 rejects natively.
    assert SourceDescriptor(uri="u", fetched_at="2026-09-01T00:00:00Z").age_seconds()
    assert SourceDescriptor(uri="u", fetched_at="2026-09-01T00:00:00+00:00")


def test_evidence_report_keeps_the_two_results_apart():
    live = {"policy/refunds": digest(_UPDATED)}
    grounding = ground_text(_CHUNK, [_CHUNK])
    freshness = check_sources([_descriptor()], resolver=lambda uri: live.get(uri))
    report = EvidenceReport(grounding=grounding, freshness=freshness)

    assert grounding.faithful            # faithful to the snapshot it was given
    assert not freshness.all_current     # and the snapshot has moved on
    assert not report.trustworthy        # so the answer is not trustworthy
    # The two failures stay distinguishable, because they need different fixes.
    assert report.grounding.faithful is True
    assert report.freshness.stale != []


def test_factstore_carries_its_source_without_changing_verdicts():
    d = _descriptor()
    store = FactStore({"refund_window_days": 14}, source=d)
    g = verify_claim(store, "refund_window_days", 14)
    assert g.provenance is Provenance.VERIFIED
    assert g.source is d                      # which snapshot answered
    # An explicit source still wins, and a store without one is unchanged.
    assert verify_claim(store, "refund_window_days", 14, source="manual").source == "manual"
    assert verify_claim(FactStore({"a": 1}), "a", 1).source == "authoritative"


def test_digest_is_stable_and_content_addressed():
    assert digest(_CHUNK) == digest(_CHUNK)
    assert digest(_CHUNK) != digest(_UPDATED)
    assert digest(_CHUNK).startswith("sha256:")


# --- Sentence boundaries ----------------------------------------------------


SPLIT_CASES = [
    ("Dr. Smith approved the claim. The fee is 1500 pounds.", 2),
    ("The policy covers e.g. flood and fire. It excludes war.", 2),
    ("Refunds take approx. 14 days. No exceptions apply.", 2),
    ("See Art. 13 of the Act. It applies to all members.", 2),
    ("Acme Inc. filed the claim. It was denied.", 2),
    ("The rate rose to 3.5% last year. It has since fallen.", 2),
    ("Contact us at help@acme.co.uk. We reply within 24 hours.", 2),
    ("The U.S. policy differs. It covers flood.", 2),
    ("Payment of 1,500.00 is due. No extensions.", 2),
    ("Claims must be filed by Jan. 31. Late claims are rejected.", 2),
    ("J. Smith signed it. The claim closed.", 2),
    ("One. Two. Three.", 3),
    ("Is it covered? Yes! It is.", 3),
    ('He said "it is covered." We agreed.', 2),
]


def test_abbreviations_do_not_end_a_sentence():
    for text, expected in SPLIT_CASES:
        assert len(split_sentences(text)) == expected, text


def test_dotted_acronyms_split_but_domains_do_not():
    # Every dotted piece is short, so it is an acronym, not a boundary.
    assert len(split_sentences("The U.S. policy differs. It covers flood.")) == 2
    # A domain has a long piece, so the final period really does end it.
    assert len(split_sentences("Email help@acme.co.uk. We reply.")) == 2
    assert split_sentences("Email help@acme.co.uk. We reply.")[0].endswith("co.uk.")


def test_bad_split_no_longer_corrupts_the_verdict():
    """Regression: splitting on "Art." stranded the claim from its negation.

    The fragment "13, cover flood damage." carried no negation cue, so polarity
    parity passed it, and redacted() returned it as text safe to show. That
    inverted the answer's meaning.
    """
    answer = "The policy does not, per Art. 13, cover flood damage."
    sources = ["The policy covers flood damage, per Art. 13."]
    report = ground_text(answer, sources, check_polarity=True)

    assert len(report.sentences) == 1            # one sentence, not two
    assert not report.faithful
    assert "polarity" in report.sentences[0].reason
    assert report.redacted() == ""               # nothing was safe to show


def test_empty_and_unterminated_text():
    assert split_sentences("") == []
    assert split_sentences("   ") == []
    assert split_sentences("No terminal punctuation") == ["No terminal punctuation"]
    assert ground_text("", ["anything"]).faithful     # nothing to contradict


def test_unknown_abbreviation_still_splits():
    """Unknown abbreviations fall back to the old behaviour, never worse."""
    assert len(split_sentences("Filed by Xyz. The claim closed.")) == 2


# --- What these checks cannot see -------------------------------------------


RELATIONAL_INVERSIONS = [
    ("Acme sued Beta Corp.", ["Beta Corp sued Acme."]),
    ("The claimant owes the insurer 500 pounds.",
     ["The insurer owes the claimant 500 pounds."]),
    ("Policy A supersedes Policy B.", ["Policy B supersedes Policy A."]),
    ("The fee rose from 14 to 30 pounds.", ["The fee fell from 30 to 14 pounds."]),
    ("Coverage increased after the review.", ["Coverage decreased after the review."]),
    ("The deadline is before the hearing.", ["The deadline is after the hearing."]),
]


def test_relational_inversions_pass_and_that_is_documented():
    """Pins the blind spot the README and module docstring describe.

    These answers reverse what the evidence says and every check passes them,
    three at a perfect overlap score. Identical tokens, identical numbers, no
    negation cue. This is why a clean report means "nothing fired", not
    "supported", and why the library must not be wired to auto-accept.

    If this test ever starts failing, a check has improved and the
    documentation claiming these get through must be corrected in the same
    commit.
    """
    passed = [a for a, s in RELATIONAL_INVERSIONS if ground_text(a, s).faithful]
    assert len(passed) == len(RELATIONAL_INVERSIONS)

    perfect = [
        a for a, s in RELATIONAL_INVERSIONS
        if ground_text(a, s).sentences[0].score == 1.0
    ]
    assert len(perfect) >= 3


def test_redaction_keeps_an_inverted_claim_intact():
    """redacted() removes what was flagged, which is not the same as safe."""
    answer = "Acme sued Beta Corp."
    report = ground_text(answer, ["Beta Corp sued Acme."])
    assert report.redacted() == answer     # survives redaction, still reversed


def test_rejections_are_the_reliable_direction():
    """The asymmetry stated in the docs: flags are trustworthy, passes are not."""
    # Fires, and is right to.
    assert not ground_text(
        "Refunds within 30 days.", ["Refunds within 14 days."]
    ).faithful
    assert not ground_text(
        "Refunds are available.", ["Refunds are not available."], check_polarity=True
    ).faithful
    # Does not fire, and is wrong not to.
    assert ground_text("Acme sued Beta Corp.", ["Beta Corp sued Acme."]).faithful
