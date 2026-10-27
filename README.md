# groundkit

[![CI](https://github.com/ss1738/groundkit/actions/workflows/ci.yml/badge.svg)](https://github.com/ss1738/groundkit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Deterministic-first grounding for LLM output.** A model can *propose*, but it
can never *assert a fact*. groundkit checks structured claims against an
authoritative source and free-text answers against retrieved evidence, tags every
value with its **provenance**, and guarantees nothing unverified is silently
presented as truth. Pure Python, zero dependencies.

```python
from groundkit import FactStore, ground_record, trusted

store  = FactStore({"capital_of_france": "Paris", "ceo_of_acme": "Jane Doe"})
record = {
    "capital_of_france": "Lyon",   # model got it wrong (source says Paris)
    "ceo_of_acme": "Jane Doe",     # right
    "acme_founded": "1990",        # unverifiable, not in the source
}

grounded = ground_record(store, record)
grounded["capital_of_france"].provenance   # Provenance.CONTRADICTED  -> value "Paris" (source wins)
grounded["acme_founded"].provenance        # Provenance.ADVISORY      -> kept, but labelled
trusted(grounded)                          # {"ceo_of_acme": "Jane Doe"}  -- safe to present as fact
```

## The pattern

Two layers, borrowed from how you'd build a system you actually trust:

- **Layer 1: deterministic.** Facts are checked against an authoritative source
  (a register, a database, retrieved passages). Exact, no model involved. A claim
  is `VERIFIED`, `CONTRADICTED` (and the *source* value is kept, never the
  model's), or `ADVISORY` (no record to check it against).
- **Layer 2: advisory only.** Whatever an LLM contributes is, by construction,
  tagged `ADVISORY`. It's never promoted to fact.

Every value comes back as a `Grounded(value, provenance, source, note)`, so the
caller always knows what's trustworthy. `trusted()` hands you only the `VERIFIED`
subset.

## Benchmark: wrong facts reaching the user

`benchmarks/bench_grounding.py`: 5,000 records, a model hallucinating 20% of
verifiable fields, plus unverifiable extras:

| Policy | Wrong facts surfaced as truth |
|---|---|
| Trust the model | 2,927 |
| **groundkit** | **0** |

![groundkit: wrong facts reaching the user, lower is better](docs/benchmark.png)

groundkit catches **100%** of hallucinated fields deterministically (the
authoritative value is kept) and labels every unverifiable claim `ADVISORY`
instead of asserting it.

## RAG answer faithfulness

`ground_text` checks each sentence of an answer against the passages it was
supposed to use:

```python
from groundkit import ground_text

sources = ["The Eiffel Tower was completed in 1889.", "It stands 330 m tall in Paris."]
answer  = "The Eiffel Tower was completed in 1889. It is the most visited monument on Earth."

report = ground_text(answer, sources)
report.faithful            # False -- the 2nd sentence isn't supported
report.unsupported         # [SentenceVerdict("It is the most visited monument on Earth.", ...)]
report.redacted()          # "The Eiffel Tower was completed in 1889."
report.flagged()           # "... 1889. It is the most visited monument on Earth. [UNSUPPORTED]"
```

Sentences are split with an abbreviation-aware splitter, because every verdict
below is computed per sentence. Cutting "per Art. 13, cover flood damage" in half
strands the claim from its negation, and `redacted()` then returns a fragment that
inverts the answer. `ABBREVIATIONS` is a short English list you can extend;
unknown abbreviations split as before, so adding to it only ever helps.

Two deterministic checks run on each sentence, both dependency-free. Content-word overlap flags
sentences whose substance is absent from the evidence (the common RAG failure).
Quantity parity then requires every figure the sentence asserts to appear in the
evidence:

```python
ground_text("Refunds are issued within 30 days.",
            ["Refunds are issued within 14 days."])
# supported: False, score: 0.875, reason: "quantity absent from the evidence: 30 day"
```

Overlap alone scores that 0.875 and passes it. The quantity check catches wrong
numbers, durations, percentages, currency amounts and doses, converting only
where the conversion is exact (`two weeks` and `14 days` are the same quantity;
a month is never 30 days). Pass `check_quantities=False` to disable it.

A third check, polarity parity, catches the highest-scoring failure of all:

```python
ground_text("Refunds are available after 14 days.",
            ["Refunds are not available after 14 days."])
# supported: False, score: 1.000, reason: "polarity differs from the evidence: ..."
```

That scores a **perfect 1.000** on overlap, because every word of the answer is in
the evidence. The evidence just has one more word, and it reverses the claim. The
check is NegEx-style cue and scope detection: it recognises pseudo-cues ("not
only", "no doubt"), stops at clause-turning conjunctions, reads contractions so
`doesn't` is a cue while a plain `can` is not, and clears a token if any one
passage agrees with the answer's polarity. `benchmarks/bench_polarity.py` scores
**precision 1.000, recall 1.000** on a 27-case labelled set. That set is small and
hand-built, so read it as a regression floor rather than evidence of
generalisation. Pass `check_polarity=False` to disable it.

None of the three checks is entailment. An **antonym carrying no negation cue**
("increased" against "decreased") still passes, as does anything needing
paraphrase-level inference, so use an NLI model where that matters. For exact
wrong-value detection on fields you can extract, use the structured layer above.

## Scope: grounding is not freshness

groundkit's grounding checks answer **"is this answer supported by the evidence it
was given?"** They never answer **"is that evidence still current?"** Those are
separate claims, and the library keeps them separate rather than folding one into
the other.

Concretely: index a policy saying "refunds within 30 days", answer from it, then
change the authoritative policy to 14 days while the old chunk stays in your index.
`ground_text` still returns `VERIFIED` with a support score of 1.0. The answer
matches its retrieved source perfectly and is wrong about current policy. The same
holds for `FactStore`: a store loaded from stale data returns `VERIFIED`, because
nothing records when it was loaded.

Note the quantity check does not help here, and cannot. It compares the answer
against the evidence it was given; when the evidence itself is the stale chunk,
the figures agree. Fresh evidence with a wrong figure is caught; stale evidence
faithfully quoted is not.

`groundkit.source` checks the second claim, and reports it **separately**:

```python
from groundkit import SourceDescriptor, check_sources, digest, ground_text, EvidenceReport

src = SourceDescriptor(
    uri="policy/refunds",
    revision=digest(chunk),              # or an ETag, or a version id
    fetched_at="2026-09-01T00:00:00Z",
    ttl_seconds=86_400,
)

grounding = ground_text(answer, [chunk])                 # faithful to the snapshot?
freshness = check_sources([src], resolver=live_revision) # snapshot still current?

EvidenceReport(grounding, freshness).trustworthy         # needs both
```

`resolver` is `uri -> current revision`, called at read time. When the policy
changes underneath you, grounding still passes (correctly, the answer does match
its chunk) and freshness returns `STALE`. Two results, because they need two
different fixes: a grounding failure means the model misused its evidence, a
freshness failure means your pipeline served evidence that no longer holds and
the model may have done nothing wrong.

Freshness comes from a revision compared against the live source, or a lease
(`fetched_at` + `ttl_seconds`) checked locally with no lookup. A lease is a
statement that you trust the snapshot for a period, not evidence that it is
current. Anything unverifiable is `UNKNOWN`, never `CURRENT`, and `is_current` is
true only for `CURRENT`, so not knowing fails closed. A resolver that raises
degrades to `UNKNOWN` with the error attached rather than taking the request down.

`FactStore(facts, source=...)` carries a descriptor the same way, so a lookup can
say which snapshot answered it. That records *which* snapshot, and still says
nothing about whether it is current; check that with `check_sources`.

What the library will not do is let attribution carry both claims. There is no
combined "verified and fresh" verdict, because the two failures are not
interchangeable.

Within a single snapshot both layers catch wrong values:
`verify_claim(FactStore({"refund_window_days": 14}), "refund_window_days", 30)`
returns `CONTRADICTED` and keeps 14, and `ground_text` flags the same swap in free
text through quantity parity.

## Weaviate

`groundkit.weaviate` turns a v4 query result into passages that carry their own
provenance. It is duck-typed and **imports nothing**, so groundkit stays
dependency-free and the adapter works against a real client, a stub or a fixture.

```python
from groundkit.weaviate import to_passages, revision_resolver, ground_weaviate

passages = to_passages(response, text_property="content")
report = ground_weaviate(answer, passages, resolver=revision_resolver(collection))

report.grounding.faithful      # is the answer supported by those chunks?
report.freshness.all_current   # are those chunks still what Weaviate holds?
report.trustworthy             # both
```

Weaviate already stores the two things a freshness check needs: `uuid` identifies
the chunk and `last_update_time` moves whenever it is updated, so you get a
revision without tracking one yourself. Update a chunk in the collection and the
answer that quoted it stays `faithful` (correctly, it does match what it was
given) while freshness turns `STALE`, naming both revisions. A deleted object also
reads `STALE`, because a source that no longer exists is not current.

If the query did not request `last_update_time`, `to_passages` **raises and tells
you what to add**, rather than falling back to a content hash. A silent fallback
would compare two different kinds of revision later and report `STALE` forever.
Pass `revision_from="content"` to hash the text deliberately instead.

## API

- **`FactStore(facts)`**: `.has` / `.get` / `.matches` / `.add`, whitespace- and
  case-normalised comparison.
- **`ground_record(store, record) -> {key: Grounded}`**, **`verify_claim(...)`**
- **`trusted(grounded)`**: VERIFIED values only. **`contradictions(grounded)`**.
- **`ground_text(answer, sources, threshold=0.6, check_quantities=True,
  check_polarity=True) -> GroundingReport`** with `.faithful`, `.supported_ratio`, `.unsupported`,
  `.flagged()`, `.redacted()`. Each `.sentences[i]` is a `SentenceVerdict`
  carrying `.supported`, `.score`, `.reason` and `.provenance`
  (`VERIFIED` / `UNSUPPORTED`).
- **`extract_quantities(text)`**, **`unsupported_quantities(sentence, sources)`**
  from `groundkit.quantities`, if you want the figures directly.
- **`negated_tokens(text)`**, **`polarity_conflicts(sentence, sources)`** from
  `groundkit.negation`, likewise for polarity.
- **`Provenance`**: `VERIFIED` · `CONTRADICTED` · `ADVISORY` · `UNSUPPORTED`.
- **`split_sentences(text)`** and the `ABBREVIATIONS` set from
  `groundkit.sentences`, if you need the boundaries or want to extend the list.
- **`SourceDescriptor(uri, revision=None, fetched_at=None, ttl_seconds=None)`**
  with `.age_seconds()`, and **`digest(text)`** for a content hash.
- **`check_sources(sources, resolver=None, now=None) -> FreshnessReport`** with
  `.all_current`, `.stale`, `.expired`, `.unknown`, `.problems`. Single-source
  form is **`check_source(...)`**, returning a `FreshnessVerdict` with
  `.freshness`, `.is_current`, `.note`, `.live_revision`.
- **`Freshness`**: `CURRENT` · `STALE` · `EXPIRED` · `UNKNOWN`.
- **`EvidenceReport(grounding, freshness)`**: both results, `.trustworthy`
  requires both.

## Install

```bash
git clone https://github.com/ss1738/groundkit
cd groundkit && pip install -e .
```

Not published on PyPI. The name `groundkit` there belongs to an unrelated project,
so `pip install groundkit` does **not** install this library. Python ≥ 3.9, no
runtime dependencies.

## Related

Part of a set of four small libraries for not trusting an unverified LLM output:

- [stigma](https://github.com/ss1738/stigma): aggregate many models with consensus that learns which to trust
- [disco](https://github.com/ss1738/disco): search with a verifier (an open FunSearch)
- **groundkit** (this repo): ground a model's output against a source
- [certain](https://github.com/ss1738/certain): calibrate confidence and abstain

## License

MIT © 2026 Satyawan Singh
