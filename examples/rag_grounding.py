"""Two ways to use groundkit: structured facts, and RAG answer faithfulness."""

from groundkit import FactStore, ground_record, ground_text, trusted


def structured() -> None:
    print("== structured fact grounding ==")
    store = FactStore({"capital_of_france": "Paris", "ceo_of_acme": "Jane Doe"})
    # An LLM extraction: one right, one wrong, one unverifiable.
    record = {
        "capital_of_france": "Lyon",       # wrong
        "ceo_of_acme": "Jane Doe",         # right
        "acme_founded": "1990",            # not in the source
    }
    for key, g in ground_record(store, record).items():
        print(f"  {key:20s} {g.provenance.value:12s} -> {g.value!r}"
              + (f"  ({g.note})" if g.note else ""))
    print(f"  safe to present as fact: {trusted(ground_record(store, record))}\n")


def rag() -> None:
    print("== RAG answer faithfulness ==")
    sources = [
        "The Eiffel Tower was completed in 1889 for the World's Fair.",
        "It stands 330 metres tall in Paris.",
    ]
    answer = (
        "The Eiffel Tower was completed in 1889. "
        "It is 330 metres tall. "
        "It is the most visited monument on Earth."
    )
    report = ground_text(answer, sources)
    for v in report.sentences:
        tag = "ok " if v.supported else "!! "
        print(f"  {tag}[{v.score:.2f}] {v.sentence}")
    print(f"  faithful: {report.faithful}   supported: {report.supported_ratio:.0%}")
    print(f"  safe answer: {report.redacted()}")


if __name__ == "__main__":
    structured()
    rag()
