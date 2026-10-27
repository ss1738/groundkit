"""Precision and recall of the polarity check on a labelled set.

A polarity check that fires on correct answers is worse than none, so the number
that matters here is precision. The set is small and hand-built: it is a
regression floor and a statement of what the heuristic is aimed at, not a public
benchmark, and it was written by the same hand as the code. Treat the figures as
"does not regress" rather than "generalises".

Run: python benchmarks/bench_polarity.py
"""

from __future__ import annotations

from groundkit.negation import polarity_conflicts

# Answers that flip what the evidence says. Every one must be flagged.
FLIPPED = [
    ("The policy does not cover water damage.", ["The policy covers water damage."]),
    ("Refunds are not available after 14 days.", ["Refunds are available after 14 days."]),
    ("The patient has no history of diabetes.", ["The patient has a history of diabetes."]),
    ("The drug is not approved for children.", ["The drug is approved for children."]),
    ("This clause does not apply to contractors.", ["This clause applies to contractors."]),
    ("Refunds are available after 14 days.", ["Refunds are not available after 14 days."]),
    ("The account was never suspended.", ["The account was suspended."]),
    ("The policy doesn't cover flood damage.", ["The policy covers flood damage."]),
    ("The claim was denied.", ["The claim was approved."]),
    ("Coverage is excluded for flood damage.", ["Coverage applies for flood damage."]),
    ("The request was rejected.", ["The request was granted."]),
]

# Answers the evidence supports. None may be flagged.
SUPPORTED = [
    ("The policy does not cover water damage.", ["The policy does not cover water damage."]),
    ("Refunds are available after 14 days.", ["Refunds are available after 14 days."]),
    ("The patient has no history of diabetes.", ["The patient has no history of diabetes."]),
    ("Refunds are available after 14 days.",
     ["Refunds are available after 14 days. Claims must be filed online."]),
    ("The drug is approved for adults.", ["The drug is approved for adults and adolescents."]),
    ("Not only does the policy cover flood, it covers fire.",
     ["The policy covers flood and fire."]),
    ("Refunds are available after 14 days.",
     ["Refunds are not available before 14 days.", "Refunds are available after 14 days."]),
    ("The account was suspended.", ["The account was suspended for non-payment."]),
    ("The policy doesn't cover flood.", ["The policy doesn't cover flood."]),
    ("You can request a refund within 30 days.", ["You can request a refund within 30 days."]),
    ("No refund is issued after 14 days.", ["No refund is issued after 14 days."]),
    ("The fee cannot be waived.", ["The fee cannot be waived."]),
    ("Claims must be filed online.", ["Claims must be filed online within 30 days."]),
    ("The policy covers flood but not fire.", ["The policy covers flood but not fire."]),
    ("The claim was denied.", ["The claim was denied."]),
    ("The request was rejected by the panel.", ["The request was rejected by the panel."]),
]


def main() -> None:
    caught = [(a, s) for a, s in FLIPPED if polarity_conflicts(a, s)]
    flagged = [(a, s) for a, s in SUPPORTED if polarity_conflicts(a, s)]

    tp, fn, fp = len(caught), len(FLIPPED) - len(caught), len(flagged)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(FLIPPED)

    print(f"polarity flips:      {tp}/{len(FLIPPED)} caught")
    for a, _ in [(a, s) for a, s in FLIPPED if (a, s) not in caught]:
        print(f"    missed: {a}")
    print(f"supported answers:   {len(SUPPORTED) - fp}/{len(SUPPORTED)} left alone")
    for a, s in flagged:
        print(f"    false flag: {a} -> {sorted(polarity_conflicts(a, s))}")
    print(f"\nprecision {precision:.3f}   recall {recall:.3f}   "
          f"(false negatives {fn})")
    print("\nSmall hand-built set. A floor against regression, not evidence of")
    print("generalisation: no public negation-in-RAG benchmark is used here.")
    print("Note what these numbers do and do not say. Precision is the quality of")
    print("the rejections, which is the direction this check is good at. Recall is")
    print("measured over failures the author thought to write down, so it is not")
    print("the fraction of real-world polarity errors caught.")


if __name__ == "__main__":
    main()
