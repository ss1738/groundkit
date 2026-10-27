"""Polarity parity: the answer must not flip what the evidence says.

Content-word overlap is blind to polarity, and the failure is worse than it looks.
"Refunds are available after 14 days" scores **1.000** against a source reading
"Refunds are not available after 14 days", because every word of the answer does
appear in the evidence. The evidence simply has one more word, and that word
reverses the claim.

This is a NegEx-style check (Chapman et al., 2001): find negation cues, take the
tokens in their scope, and compare which tokens the answer negates against which
tokens the evidence negates. A token the answer negates while the evidence
affirms it, or the reverse, is a conflict.

Precision is favoured over recall throughout, because a false "unsupported" costs
more than a missed one here:

* pseudo-cues that look like negation and are not ("not only", "no doubt",
  "cannot be ruled out") are recognised and skipped.
* scope ends at a conjunction that turns the clause ("but", "however",
  "except") or after a short window, rather than running to the end.
* a token is only in conflict when *every* source containing it disagrees. One
  passage agreeing is enough to clear it.
* contractions are read before tokens are split, so "doesn't" is a cue while a
  plain "can" is not mistaken for "can't".

What it does not do: antonyms without a cue ("included" vs "excluded" is caught
only because "excluded" is a cue; "rose" vs "fell" is not), scope across
sentences, or double negation. It is a cue-and-scope heuristic, not entailment.
"""

from __future__ import annotations

import re

from .tokens import STOPWORDS

# Multi-token cues are matched longest-first.
_CUES: tuple[tuple[str, ...], ...] = (
    ("not",), ("no",), ("never",), ("none",), ("neither",), ("nor",),
    ("without",), ("cannot",), ("lacks",), ("lacking",), ("absent",),
    ("denies",), ("denied",), ("deny",), ("excludes",), ("excluded",),
    ("exclude",), ("excluding",), ("prohibits",), ("prohibited",),
    ("unable", "to"), ("fails", "to"), ("failed", "to"), ("failure", "to"),
    ("absence", "of"), ("lack", "of"), ("free", "of"), ("devoid", "of"),
    ("ruled", "out"), ("rules", "out"), ("rule", "out"),
    ("declined", "to"), ("refused", "to"), ("not", "eligible"),
    ("ineligible", "for"), ("ineligible",),
)

# Look like cues, are not. Checked before cues and skipped over.
_PSEUDO: tuple[tuple[str, ...], ...] = (
    ("not", "only"), ("not", "just"), ("not", "merely"), ("not", "necessarily"),
    ("no", "doubt"), ("no", "less"), ("no", "more", "than"),
    ("no", "fewer", "than"), ("no", "later", "than"), ("no", "earlier", "than"),
    ("cannot", "be", "excluded"), ("cannot", "be", "ruled", "out"),
    ("no", "increase"), ("no", "change"), ("no", "further"),
)

# Cues that follow what they negate ("the claim was denied"), so their scope
# runs backwards. NegEx calls these post-negation triggers.
_POST_CUES: tuple[tuple[str, ...], ...] = (
    ("denied",), ("denies",), ("rejected",), ("refused",), ("declined",),
    ("prohibited",), ("excluded",), ("unavailable",), ("ineligible",),
    ("ruled", "out"), ("was", "absent"), ("is", "absent"), ("not", "found"),
    ("not", "present"), ("was", "negative"),
)
_POST_SCOPE = 3

# Words that turn the clause, ending the scope of a cue.
_TERMINATORS = frozenset({
    "but", "however", "although", "though", "except", "unless", "whereas",
    "while", "therefore", "otherwise", "nevertheless", "nonetheless",
    "aside", "apart", "because", "since", "yet",
})

_RE_TOKEN = re.compile(r"[\w']+")


def _tokens(text: str) -> list[str]:
    """Tokens keeping apostrophes, so "doesn't" survives as one token."""
    return _RE_TOKEN.findall(text.lower())


def _bare(token: str) -> str:
    return token.replace("'", "")


def _match_at(tokens: list[str], i: int, phrases) -> int:
    """Length of the longest phrase in ``phrases`` starting at ``i``, else 0."""
    best = 0
    for phrase in phrases:
        k = len(phrase)
        if k > best and tokens[i:i + k] == list(phrase):
            best = k
    return best


def _cue_at(tokens: list[str], i: int) -> int:
    """Length of a real negation cue at ``i``; 0 for none or a pseudo-cue."""
    if _match_at(tokens, i, _PSEUDO):
        return 0
    token = tokens[i]
    # Contractions: doesn't, isn't, won't, can't ... but never a bare "can".
    if token.endswith("n't"):
        return 1
    return _match_at(tokens, i, _CUES)


def negated_tokens(text: str, scope: int = 6) -> set[str]:
    """Content tokens falling inside the scope of a negation cue."""
    tokens = _tokens(text)
    negated: set[str] = set()
    i, n = 0, len(tokens)
    while i < n:
        skip = _match_at(tokens, i, _PSEUDO)
        if skip:
            i += skip
            continue
        cue = _cue_at(tokens, i)
        if not cue:
            i += 1
            continue
        j, end = i + cue, min(n, i + cue + scope)
        while j < end:
            word = _bare(tokens[j])
            if word in _TERMINATORS or _cue_at(tokens, j):
                break
            if len(word) > 1 and word not in STOPWORDS:
                negated.add(word)
            j += 1
        i += cue

    # Post-negation cues scope backwards over the few tokens before them.
    i = 0
    while i < n:
        post = _match_at(tokens, i, _POST_CUES)
        if not post or _match_at(tokens, i, _PSEUDO):
            i += 1
            continue
        j, stop = i - 1, max(-1, i - 1 - _POST_SCOPE)
        while j > stop:
            word = _bare(tokens[j])
            if word in _TERMINATORS or _cue_at(tokens, j):
                break
            if len(word) > 1 and word not in STOPWORDS:
                negated.add(word)
            j -= 1
        i += post
    return negated


def _cue_words() -> frozenset[str]:
    out = {w for phrase in _CUES for w in phrase}
    out |= {w for phrase in _PSEUDO for w in phrase}
    out |= {w for phrase in _POST_CUES for w in phrase}
    return frozenset(out)


_CUE_WORDS = _cue_words()


def polarity_conflicts(
    sentence: str, sources: list[str], scope: int = 6
) -> set[str]:
    """Tokens whose polarity the sentence flips relative to every source.

    A token counts as a conflict only when it appears in at least one source and
    *no* source containing it agrees with the sentence's polarity.
    """
    from .tokens import content_tokens

    in_sentence = set(content_tokens(sentence))
    if not in_sentence:
        return set()
    sentence_negated = negated_tokens(sentence, scope)

    prepared = [
        (set(content_tokens(src)), negated_tokens(src, scope)) for src in sources
    ]

    conflicts = set()
    for token in in_sentence:
        if token in _CUE_WORDS:
            continue  # a cue's own polarity is not a claim
        negated_here = token in sentence_negated
        present = agreed = False
        for src_tokens, src_negated in prepared:
            if token not in src_tokens:
                continue
            present = True
            if (token in src_negated) == negated_here:
                agreed = True
                break
        if present and not agreed:
            conflicts.add(token)
    return conflicts
