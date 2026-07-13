# groundkit

[![CI](https://github.com/ss1738/groundkit/actions/workflows/ci.yml/badge.svg)](https://github.com/ss1738/groundkit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/groundkit)](https://pypi.org/project/groundkit/)
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

The support check is lexical and dependency-free: it flags sentences whose content
words are largely absent from the evidence (the common RAG failure). It is not
entailment, so a mostly-grounded sentence that swaps a single token (a wrong number,
a negation) can still pass; use an NLI model where that matters. For exact
wrong-value detection, use the structured layer above.

## API

- **`FactStore(facts)`**: `.has` / `.get` / `.matches` / `.add`, whitespace- and
  case-normalised comparison.
- **`ground_record(store, record) -> {key: Grounded}`**, **`verify_claim(...)`**
- **`trusted(grounded)`**: VERIFIED values only. **`contradictions(grounded)`**.
- **`ground_text(answer, sources, threshold=0.6) -> GroundingReport`** with
  `.faithful`, `.supported_ratio`, `.unsupported`, `.flagged()`, `.redacted()`.
  Each `.sentences[i]` is a `SentenceVerdict` carrying `.supported` and
  `.provenance` (`VERIFIED` / `UNSUPPORTED`).
- **`Provenance`**: `VERIFIED` · `CONTRADICTED` · `ADVISORY` · `UNSUPPORTED`.

## Install

```bash
pip install groundkit      # or: pip install -e . from a clone
```

Python ≥ 3.9, no runtime dependencies.

## Related

Part of a set of four small libraries for not trusting an unverified LLM output:

- [stigma](https://github.com/ss1738/stigma): aggregate many models with consensus that learns which to trust
- [disco](https://github.com/ss1738/disco): search with a verifier (an open FunSearch)
- **groundkit** (this repo): ground a model's output against a source
- [certain](https://github.com/ss1738/certain): calibrate confidence and abstain

## License

MIT © 2026 Satyawan Singh
