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
from the evidence, which is the common RAG failure. It won't catch a single swapped
token (a wrong number, a negation) in an otherwise-grounded sentence, and it isn't
entailment, so swap in an NLI model where you need that. For catching a wrong value
exactly, the structured layer above is the right tool; for "did the model bolt on a
whole claim that isn't in the sources", the lexical overlap is cheap, dependency-free,
and surprisingly effective.

## Try it

```python
from groundkit import FactStore, ground_record, trusted

store  = FactStore({"ceo": "Jane Doe"})
record = {"ceo": "John Smith"}          # model got it wrong
trusted(ground_record(store, record))   # {}  -- the wrong value never reaches the user
```

Code, tests, benchmark: https://github.com/ss1738/groundkit
