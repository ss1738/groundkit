"""Sentence splitting that does not break on abbreviations.

Splitting on terminal punctuation alone cuts "Dr. Smith", "e.g.", "Art. 13" and
"Jan. 31" in half, and every text-layer verdict is computed per sentence, so a
bad boundary corrupts the result rather than merely looking untidy.

The damage is concrete. Splitting

    The policy does not, per Art. 13, cover flood damage.

after "Art." strands "cover flood damage" in a fragment with no negation cue, so
polarity parity sees nothing to flag and marks it supported, and ``redacted()``,
which is documented as the text safe to show, returns "13, cover flood damage."
That inverts the answer's meaning and presents it as verified.

A period is therefore treated as a boundary unless something says otherwise:

* the token before it is a known abbreviation ("Dr", "Art", "approx", "Jan");
* it is an initial, a single letter such as the "J." in "J. Smith";
* it is part of a dotted acronym, where every dot-separated piece is one or two
  characters ("U.S.", "e.g.", "Ph.D."). The two-character limit is what keeps
  "help@acme.co.uk." splitting normally;
* the next sentence would start with a lowercase letter, which in practice means
  the period was not ending anything.

The abbreviation list is English and deliberately short, weighted towards the
legal, clinical and financial text where this matters. Unknown abbreviations
split, which is the old behaviour, so adding to the list only ever improves
things.
"""

from __future__ import annotations

import re

ABBREVIATIONS = frozenset({
    # titles
    "dr", "mr", "mrs", "ms", "prof", "rev", "hon", "sr", "jr", "st", "capt",
    # organisations
    "inc", "ltd", "llc", "llp", "plc", "co", "corp", "bros", "assn", "dept",
    # references and legal
    "art", "arts", "sec", "secs", "cl", "para", "paras", "pt", "ch", "chap",
    "fig", "figs", "no", "nos", "vol", "vols", "ed", "eds", "p", "pp", "ff",
    "cf", "al", "ibid", "seq", "reg", "regs", "sched",
    # measurement and hedging
    "approx", "est", "max", "min", "avg", "incl", "excl", "etc", "vs", "viz",
    # clinical
    "mg", "ml", "kg", "mcg", "iu", "bid", "tid", "qd", "prn", "wt", "ht",
    # dates
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sept", "sep", "oct",
    "nov", "dec", "mon", "tues", "tue", "wed", "thurs", "thur", "thu", "fri",
    "sat", "sun",
})

_RE_BOUNDARY = re.compile(r"([.!?]+)([\"')\]]*)(\s+)")
_RE_TRAILING_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)$")


def _is_dotted_acronym(token: str) -> bool:
    """True for "U.S", "e.g", "Ph.D": every dotted piece is one or two chars."""
    if "." not in token:
        return False
    pieces = [p for p in token.split(".") if p]
    return bool(pieces) and all(len(p) <= 2 for p in pieces)


def _is_boundary(text: str, match: re.Match) -> bool:
    before = text[: match.start(1)]
    word = _RE_TRAILING_WORD.search(before)
    if word:
        token = word.group(1)
        bare = token.rstrip(".").lower()
        if bare in ABBREVIATIONS:
            return False
        if len(bare) == 1:          # an initial: "J. Smith"
            return False
        if _is_dotted_acronym(token):
            return False
    after = text[match.end():]
    return not (after and after[0].islower())


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, keeping abbreviations intact."""
    text = text.strip()
    if not text:
        return []
    out: list[str] = []
    start = 0
    for match in _RE_BOUNDARY.finditer(text):
        if not _is_boundary(text, match):
            continue
        piece = text[start:match.end(2)].strip()
        if piece:
            out.append(piece)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out
