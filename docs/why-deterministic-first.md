# Authority and generation should never be the same source

An LLM is a brilliant *proposer* and an unreliable *authority*. The mistake most
pipelines make is letting it be both, the model generates a value and, by virtue
of being the thing that spoke last, that value is treated as fact. A hallucinated
company number, a wrong CEO, an invented statistic: all delivered to the user with
the same confidence as the correct fields around them.

groundkit's rule is simple: **the model may propose a value; only an authoritative
source may make it a fact.** Generation and authority are kept as separate sources,
and authority always wins.

## Two layers

**Layer 1 is deterministic and the model never touches it.** You give groundkit a
source you actually trust, a register, a database, retrieved passages. Each claim
the model produced is checked against it:

- `VERIFIED`, the claim matches the source.
- `CONTRADICTED`, the source disagrees. The **source's** value is kept; the
  model's rejected claim is recorded in a note, never surfaced.
- `ADVISORY`, the source has no record, so the claim can't be checked. It's kept
  but explicitly labelled as an unverified model claim.

**Layer 2 is whatever the LLM adds**, and by construction it can only ever be
`ADVISORY`. There is no code path by which a model's unverified output becomes a
fact.

Every value returns as `Grounded(value, provenance, source, note)`, so the caller
always knows what's safe. `trusted()` gives you only the `VERIFIED` subset.

## Why fail-safe matters

The important property is what happens when the source *doesn't* know. Trust-the-
model gives you a confident guess dressed as fact. groundkit gives you an explicit
`ADVISORY` tag, the absence of verification is made visible, not hidden. You get
to decide (show it with a caveat, ask a human, drop it) instead of the model
deciding for you by omission.

## The numbers

5,000 records, a model hallucinating 20% of the verifiable fields, plus
unverifiable extras:

| Policy | Wrong facts surfaced as truth |
|---|---|
| Trust the model | 2,927 |
| groundkit | 0 |

Every contradiction is caught deterministically (source value kept); every
unverifiable claim is labelled rather than asserted. The catch rate on the
structured layer is 100% because it's exact, not statistical.

## RAG answers, too

The same principle applies to free text. `ground_text` splits an answer into
sentences and checks each against the passages it was meant to use:

```python
from groundkit import ground_text

sources = ["The Eiffel Tower was completed in 1889.", "It stands 330 m tall in Paris."]
answer  = "The Eiffel Tower was completed in 1889. It is the most visited monument on Earth."

report = ground_text(answer, sources)
report.faithful       # False
report.redacted()     # "The Eiffel Tower was completed in 1889."
```

The check is lexical: it flags sentences whose content words are largely absent
from the evidence, which is the common RAG failure. A second deterministic pass
requires every figure the sentence asserts to appear in the evidence, so a swapped
number is caught even though the wording still scores 0.875. A third pass compares
polarity, so an answer that drops a "not" is caught even at a perfect 1.000
overlap score. It is still not entailment, and the asymmetry matters: these
checks reject reliably but never confirm. An antonym carrying no cue passes, and
so does a reversed relation, "Acme sued Beta Corp" against "Beta Corp sued Acme",
at a perfect overlap score. A clean result means nothing fired. Swap in an NLI
model where you need a positive finding of support. For catching a wrong value
exactly, the structured layer above is the right tool; for "did the model bolt on a
whole claim that isn't in the sources", the lexical overlap is cheap, dependency-free,
and surprisingly effective.

Both layers check an answer against the snapshot they were handed, never whether that
snapshot is still current. Replace an authoritative policy and leave the old chunk in
the index and the old answer still verifies against it, correctly and uselessly.
"Supported by this evidence" and "this evidence is still current" are two claims, and
deterministic grounding only settles the first. `groundkit.source` settles the
second, through a revision compared at read time or a TTL lease, and deliberately
returns it as its own result. Collapsing them into one verdict would be the same
mistake as letting the model vouch for itself: one check, two claims, no way to
tell which failed.

## Try it

```python
from groundkit import FactStore, ground_record, trusted

store  = FactStore({"ceo": "Jane Doe"})
record = {"ceo": "John Smith"}          # model got it wrong
trusted(ground_record(store, record))   # {}  -- the wrong value never reaches the user
```

Code, tests, benchmark: https://github.com/ss1738/groundkit
