"""Per-sentence evaluation on RAGTruth, a public hallucination corpus.

The hand-built set in ``bench_polarity.py`` is a regression floor written by the
same hand as the code. This is the opposite: 18,821 sentences of real LLM output
with human span labels, which nobody here chose.

The number that matters is **lift**, precision minus the base rate. A check that
fires at random scores the base rate exactly, so lift is how much the check
actually tells you. Precision alone flatters a check that fires constantly.

Get the data (MIT licensed, ~36 MB, not vendored here):

    mkdir -p ragtruth && cd ragtruth
    curl -LO https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl
    curl -LO https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl

Run: python benchmarks/bench_public.py path/to/ragtruth

Reference: Niu et al., "RAGTruth: A Hallucination Corpus for Developing
Trustworthy Retrieval-Augmented Language Models" (arXiv:2401.00396).
"""

from __future__ import annotations

import json
import sys

from groundkit.negation import polarity_conflicts
from groundkit.quantities import unsupported_quantities
from groundkit.sentences import split_sentences
from groundkit.text import _nearest, support_score


def _evidence(source: dict) -> str:
    info, task = source["source_info"], source["task_type"]
    if task == "QA":
        return info["passages"]
    if task == "Summary":
        return info if isinstance(info, str) else json.dumps(info)
    return json.dumps(info)


def load(directory: str) -> list[tuple[float, bool, bool, bool]]:
    """One row per response sentence: (overlap, quantity, polarity, is_hallucinated)."""
    with open(f"{directory}/source_info.jsonl") as fh:
        sources = {json.loads(line)["source_id"]: json.loads(line) for line in fh}
    with open(f"{directory}/response.jsonl") as fh:
        responses = [json.loads(line) for line in fh]
    responses = [r for r in responses if r.get("split") == "test"]

    rows = []
    for response in responses:
        evidence = _evidence(sources[response["source_id"]])
        evidence_sentences = split_sentences(evidence)
        text = response["response"]
        spans = [(lab["start"], lab["end"]) for lab in (response.get("labels") or [])]
        cursor = 0
        for sentence in split_sentences(text):
            start = text.find(sentence, cursor)
            start = cursor if start < 0 else start
            end = start + len(sentence)
            cursor = end
            rows.append((
                max((support_score(sentence, e) for e in evidence_sentences),
                    default=0.0),
                bool(unsupported_quantities(sentence, [evidence])),
                bool(polarity_conflicts(sentence, _nearest(sentence, evidence_sentences))),
                any(a < end and start < b for a, b in spans),
            ))
    return rows


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip().splitlines()[-4].strip())
    rows = load(sys.argv[1])
    n = len(rows)
    bad = sum(1 for *_, b in rows if b)
    base = bad / n
    print(f"RAGTruth test split: {n} sentences, {bad} hallucinated, "
          f"base rate {base:.3f}\n")
    print(f"{'check':34} {'flags':>7} {'recall':>8} {'precision':>10} {'lift':>8}")

    def row(name, pick):
        tp = sum(1 for r in rows if pick(r) and r[3])
        fp = sum(1 for r in rows if pick(r) and not r[3])
        prec = tp / (tp + fp) if tp + fp else float("nan")
        print(f"{name:34} {(tp + fp) / n:6.1%} {tp / bad:8.3f} "
              f"{prec:10.3f} {prec - base:+8.3f}")

    for t in (0.2, 0.3, 0.4, 0.6):
        row(f"overlap only, threshold {t}", lambda r, t=t: r[0] < t)
    row("quantity only", lambda r: r[1])
    row("polarity only (scoped)", lambda r: r[2])
    row("shipped default (0.3 + quantity)", lambda r: r[0] < 0.3 or r[1])
    row("  ...plus polarity", lambda r: r[0] < 0.3 or r[1] or r[2])

    print("\nQuantity parity is the check that earns its place: it fires on a few")
    print("per cent of sentences at roughly three times the base rate. Overlap")
    print("loses lift as its threshold rises, because a higher bar rejects more")
    print("correct sentences than wrong ones. Polarity adds nothing here and")
    print("lowers the lift of any combination it joins, which is why it is off")
    print("by default; it was built and measured on short single-claim evidence.")


if __name__ == "__main__":
    main()
