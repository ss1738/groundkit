"""The authoritative source -- a deterministic key/value store of ground truth.

This is Layer 1: plain, exact lookups you trust (a register, a database, an API
response). No model touches it. Comparisons are normalised for whitespace and
case so ``"Jane Doe"`` and ``" jane  doe "`` match, but nothing fuzzier -- Layer 1
must stay strict.
"""

from __future__ import annotations

import re
from typing import Any


def normalise(value: Any) -> str:
    """Casefold and collapse whitespace for strict value comparison."""
    return re.sub(r"\s+", " ", str(value).strip()).lower()


class FactStore:
    """An authoritative key -> value store.

    Args:
        facts: Initial ground-truth mapping.
    """

    def __init__(self, facts: dict[str, Any] | None = None) -> None:
        self._facts: dict[str, Any] = dict(facts or {})

    def has(self, key: str) -> bool:
        return key in self._facts

    def get(self, key: str) -> Any:
        return self._facts.get(key)

    def matches(self, key: str, value: Any) -> bool:
        """True iff ``key`` exists and its stored value equals ``value``."""
        return self.has(key) and normalise(self._facts[key]) == normalise(value)

    def add(self, key: str, value: Any) -> None:
        self._facts[key] = value

    def __len__(self) -> int:
        return len(self._facts)
