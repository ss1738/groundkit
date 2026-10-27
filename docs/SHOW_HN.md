# Launch copy for groundkit

## Show HN title (pick one)
- **Show HN: groundkit, let your LLM propose, but never assert a fact**
- Show HN: groundkit, deterministic-first grounding to stop LLM hallucinations
- Show HN: Every LLM value carries provenance, nothing unverified reaches the user

## Show HN body

Most LLM pipelines let the model both generate an answer and implicitly vouch for
it. That's where hallucination lives: a confident wrong value goes straight to the
user as fact.

groundkit splits generating from vouching. It's a small (zero-dependency) Python
library built on one rule: **a model can propose a value, but only an
authoritative source can make it a fact.**

Two layers:

1. **Deterministic.** Structured claims are checked against a source you trust (a
   register, a DB, retrieved passages). A claim is `VERIFIED`, `CONTRADICTED` (and
   the *source's* value is kept, never the model's), or `ADVISORY` (nothing to
   check it against). Free-text answers are checked sentence-by-sentence against
   the evidence they were supposed to use.
2. **Advisory.** Whatever the LLM adds is, by construction, tagged `ADVISORY`, it
   is never promoted to fact.

Every value comes back tagged with its provenance, and `trusted()` hands you only
the verified subset. On a 5,000-record benchmark with a model hallucinating 20% of
fields: **trust-the-model surfaces 2,927 wrong facts; groundkit surfaces 0**,
every contradiction caught deterministically, every unverifiable claim labelled
rather than asserted.

It also does RAG faithfulness: `ground_text(answer, sources)` flags any sentence
that introduces content the sources don't support, and can redact it.

Repo, tests, benchmark: https://github.com/ss1738/groundkit

Honest scope: the text check is three deterministic passes, content-word overlap
plus quantity parity plus polarity parity. It catches sentences that add
unsupported content (the common RAG failure), sentences that reuse the evidence's
wording while changing a figure, and sentences that flip what the evidence says,
which is the failure that scores a perfect 1.000 on overlap. The checks reject rather than verify: when one
fires it is reliable, but a clean result means only that nothing fired. Measured
against the library, "Acme sued Beta Corp" against a source reading "Beta Corp
sued Acme" scores 1.000 and passes, as do antonyms carrying no cue. Relational
inversion and paraphrase-level entailment need a model; drop in NLI for that, and
do not wire the text layer to auto-accept. The structured layer
is exact and is where the 100% catch rate comes from. Grounding and freshness are kept
apart: the checks above ask whether an answer is supported by the evidence it was
given, never whether that evidence is still current, so an answer citing a stale
chunk perfectly still comes back VERIFIED. `groundkit.source` answers the second
question through a read-time revision resolver or a TTL lease, and reports it as a
separate result rather than folding it into attribution, because the two failures
need different fixes. Anything unverifiable is UNKNOWN, never current. Feedback
welcome.

## First comment (post right after)

The design principle is that authority and generation should be different sources.
An LLM is a great *proposer* and an unreliable *authority*, so groundkit never
lets it be the authority: the deterministic layer always wins, and the model's
contribution is quarantined as ADVISORY until something trustworthy confirms it.
The nice property is that this fails safe, if the source has no record, you don't
get a confident guess presented as fact, you get an explicit "unverified" tag the
caller has to decide what to do with.

## Part of a trio
groundkit is one of three small libraries on the same theme, don't trust an
unverified LLM output:
- stigma, aggregate many models (consensus that learns which to trust)
- disco, search with a verifier (open FunSearch)
- groundkit, ground a single model's output (this one)

## Where else to post
- r/MachineLearning ("[P] ..."), r/LocalLLaMA, r/Rag
- X/Twitter: the 2,927 → 0 result + "let the model propose, not assert"
- lobste.rs (ml / ai)
