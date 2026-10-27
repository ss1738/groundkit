"""Quantity parity: the numbers a sentence asserts must appear in the evidence.

Content-word overlap cannot see a swapped number. "Refunds within 30 days"
against a source saying "within 14 days" shares every word but the digits, so it
scores 0.875 and passes a 0.6 threshold while being wrong about the only part
that matters. The same hole swallows wrong prices, dates, dosages and rates.

This module closes it deterministically. It extracts the quantities a sentence
asserts, canonicalises them, and requires each one to appear somewhere in the
retrieved evidence. No model, no dependency, microseconds.

Canonicalisation is deliberately conservative, because a false "unsupported" is
expensive:

* durations are converted only where the conversion is exact (week -> 7 days,
  fortnight -> 14 days). Months and years are never converted to days, since
  they are not fixed lengths; they compare within their own unit.
* number words are read up to twenty plus the tens, so "two weeks" and "14 days"
  are the same quantity.
* a bare number in the sentence matches that value under any unit in the source,
  so "the window is 30" is still supported by "30 days". A sentence quantity that
  carries a unit must match that unit, or a bare number of the same value.
* ordinals, version-like strings and anything not parsed are ignored rather than
  guessed at.

What it does not do: arithmetic (it will not derive "50%" from "1 in 2"), unit
systems beyond time, or ranges. It answers one question, whether the figure was
in the evidence, and leaves everything else to the caller.
"""

from __future__ import annotations

import re

# Exact conversions only. A month is not 30 days and a year is not 365.
_TO_DAYS = {"day": 1, "days": 1, "week": 7, "weeks": 7, "fortnight": 14,
            "fortnights": 14}

_UNIT_ALIASES = {
    "percent": "%", "pct": "%", "%": "%",
    "hour": "hour", "hours": "hour", "hr": "hour", "hrs": "hour",
    "minute": "minute", "minutes": "minute", "min": "minute", "mins": "minute",
    "second": "second", "seconds": "second", "sec": "second", "secs": "second",
    "month": "month", "months": "month",
    "year": "year", "years": "year", "yr": "year", "yrs": "year",
    "day": "day", "days": "day",
    "mg": "mg", "kg": "kg", "g": "g", "ml": "ml", "l": "l",
    "km": "km", "cm": "cm", "mm": "mm", "m": "m",
}

_CURRENCY = {"£": "gbp", "$": "usd", "€": "eur", "gbp": "gbp", "usd": "usd",
             "eur": "eur"}

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}

_UNIT_PATTERN = "|".join(
    re.escape(u)
    for u in sorted(set(_UNIT_ALIASES) | set(_TO_DAYS), key=len, reverse=True)
)
_WORD_PATTERN = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))

# £1,000  /  $1000.50
_RE_CURRENCY = re.compile(r"([£$€])\s?(\d[\d,]*(?:\.\d+)?)")
# 1,000 GBP
_RE_CURRENCY_SUFFIX = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(gbp|usd|eur)\b", re.IGNORECASE)
# 30 days / 30-day / 5% / 5 percent
_RE_NUM_UNIT = re.compile(
    rf"(\d[\d,]*(?:\.\d+)?)\s*[-\s]?\s*({_UNIT_PATTERN})(?!\w)", re.IGNORECASE
)
# two weeks / fourteen days
_RE_WORD_UNIT = re.compile(
    rf"\b({_WORD_PATTERN})\s*[-\s]?\s*({_UNIT_PATTERN})(?!\w)", re.IGNORECASE
)
_RE_BARE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?![\w.]*[a-z])", re.IGNORECASE)


def _num(raw: str) -> float:
    return float(raw.replace(",", ""))


def _fmt(value: float) -> str:
    return f"{value:g}"


def _canonical(value: float, unit: str) -> tuple[str, str]:
    """Normalise a (value, unit) pair, converting durations where exact."""
    u = unit.lower()
    if u in _TO_DAYS:
        return _fmt(value * _TO_DAYS[u]), "day"
    return _fmt(value), _UNIT_ALIASES.get(u, u)


def extract_quantities(text: str) -> set[tuple[str, str]]:
    """Every quantity in ``text`` as canonical ``(value, unit)`` pairs.

    A unit of ``""`` means a bare number carrying no unit.
    """
    found: set[tuple[str, str]] = set()
    consumed: list[tuple[int, int]] = []

    for rx, handler in (
        (_RE_CURRENCY, lambda m: (_num(m.group(2)), _CURRENCY[m.group(1)])),
        (_RE_CURRENCY_SUFFIX,
         lambda m: (_num(m.group(1)), _CURRENCY[m.group(2).lower()])),
        (_RE_NUM_UNIT, lambda m: (_num(m.group(1)), m.group(2))),
        (_RE_WORD_UNIT,
         lambda m: (float(_NUMBER_WORDS[m.group(1).lower()]), m.group(2))),
    ):
        for m in rx.finditer(text):
            value, unit = handler(m)
            if unit in ("gbp", "usd", "eur"):
                found.add((_fmt(value), unit))
            else:
                found.add(_canonical(value, unit))
            consumed.append(m.span())

    for m in _RE_BARE.finditer(text):
        if any(s <= m.start() and m.end() <= e for s, e in consumed):
            continue  # already captured with its unit
        found.add((_fmt(_num(m.group(1))), ""))
    return found


def unsupported_quantities(
    sentence: str, sources: list[str]
) -> set[tuple[str, str]]:
    """Quantities asserted by ``sentence`` that appear in no source.

    A bare sentence number matches the same value under any unit; a sentence
    quantity carrying a unit matches the same unit, or a bare source number of
    the same value.
    """
    in_sentence = extract_quantities(sentence)
    if not in_sentence:
        return set()
    in_sources: set[tuple[str, str]] = set()
    for source in sources:
        in_sources |= extract_quantities(source)
    source_values = {v for v, _ in in_sources}

    missing = set()
    for value, unit in in_sentence:
        if unit == "":
            if value not in source_values:
                missing.add((value, unit))
        elif (value, unit) not in in_sources and (value, "") not in in_sources:
            missing.add((value, unit))
    return missing
