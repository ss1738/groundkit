"""How many wrong facts reach the user: trust-the-model vs groundkit.

Simulates an extraction model over records with a known authoritative source. The
model hallucinates each verifiable field with some probability (a wrong value) and
also emits fields the source can't check. We compare two policies:

* trust-the-model -- present everything the model returned as fact;
* groundkit       -- present only what the authoritative source VERIFIED.

The metric is *wrong facts surfaced as truth*. groundkit's guarantee is that this
is zero: contradictions are caught deterministically (source value kept), and
unverifiable claims are labelled ADVISORY rather than asserted.

Run: python benchmarks/bench_grounding.py
"""

from __future__ import annotations

import random

from groundkit import FactStore, ground_record, trusted

N_RECORDS = 5000
HALLUCINATION_RATE = 0.2
SEED = 3
VERIFIABLE = ["capital", "ceo", "founded"]  # in the source
UNVERIFIABLE = "revenue"                     # never in the source


def main() -> None:
    rng = random.Random(SEED)

    trust_wrong = 0       # wrong facts surfaced by trust-the-model
    groundkit_wrong = 0   # wrong facts surfaced by groundkit
    caught = 0            # hallucinations groundkit flagged
    total_hallucinations = 0

    for i in range(N_RECORDS):
        truth = {
            "capital": f"City{i % 50}",
            "ceo": f"Person{i % 97}",
            "founded": str(1900 + i % 120),
        }
        store = FactStore(truth)

        record: dict[str, str] = {}
        for field in VERIFIABLE:
            if rng.random() < HALLUCINATION_RATE:
                record[field] = "WRONG-" + truth[field]   # a hallucinated value
                total_hallucinations += 1
            else:
                record[field] = truth[field]
        record[UNVERIFIABLE] = "unverifiable-claim"        # not in the source

        # trust-the-model: everything asserted; count values that differ from truth
        for field in VERIFIABLE:
            if record[field] != truth[field]:
                trust_wrong += 1

        # groundkit: only VERIFIED values are presented
        grounded = ground_record(store, record)
        surfaced = trusted(grounded)
        for field, value in surfaced.items():
            if field in truth and value != truth[field]:
                groundkit_wrong += 1
        for field in VERIFIABLE:
            if record[field] != truth[field] and field not in surfaced:
                caught += 1

    print(f"records:                       {N_RECORDS}")
    print(f"hallucination rate / field:    {HALLUCINATION_RATE:.0%}")
    print(f"total hallucinated fields:     {total_hallucinations}")
    print()
    print(f"wrong facts surfaced -- trust model:  {trust_wrong}")
    print(f"wrong facts surfaced -- groundkit:    {groundkit_wrong}")
    print(f"hallucinations caught by groundkit:   {caught}/{total_hallucinations}"
          f"  ({caught / total_hallucinations:.1%})")


if __name__ == "__main__":
    main()
