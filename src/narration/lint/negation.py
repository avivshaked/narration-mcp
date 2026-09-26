"""The positive-only check of voice descriptions (design section 3.5; WP10).

A negated quality tends to be heard by the design model as that quality, so a description should name what
is wanted ("smooth") rather than what is not ("not rough"). The check is a plain word list, whole-word and
case-insensitive, with no language model. It **warns and never refuses**: it misses negations without
those words ("less theatrical", "free of rasp") and can raise false alarms ("a no-nonsense tone").

Each finding reports the phrase, its offset (code points into the description as sent) and a suggestion.
The phrase starts at the trigger word and runs to the end of the hyphenated word it is part of (``non-
breathy``, ``no-nonsense``), or else through the next word (``not theatrical``), stopping at punctuation;
a longer phrase from the suggestion table that matches there (``no movie-trailer delivery``) wins.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Final

from narration.contracts.models import LintFinding, LintResult
from narration.text.canonical import Token, tokens

POLICY: Final = "warn"
"""Owner decision (section 3.5, Q6): the design runs as asked, and ``lint`` lists the findings."""

TRIGGER_WORDS: Final[frozenset[str]] = frozenset(
    {"not", "no", "never", "without", "avoid", "don't", "dont", "nor", "neither", "none"}
)
"""Whole words, compared case-insensitively (with ’ read as ')."""

NT_SUFFIX: Final = "n't"
"""Every other ``n't`` form (isn't, won't, can't, …) is a trigger too."""

NON_PREFIX: Final = "non-"
"""The prefix ``non-``: ``non`` as a whole word followed by a hyphen and a letter (``nonchalant`` is not)."""

HYPHENS: Final[frozenset[str]] = frozenset({"-", "‐", "‑"})
APOSTROPHES: Final[frozenset[str]] = frozenset({"'", "’"})

ALLOWED_NEGATIVES: Final[tuple[str, ...]] = ("unhurried", "understated", "effortless")
"""Morphological negatives that name a positive quality; they contain no trigger word, so they pass."""

SUGGESTIONS: Final[tuple[tuple[tuple[str, ...], str], ...]] = (
    (("not theatrical", "never theatrical"), "natural, understated delivery"),
    (("not gravelly", "not rough"), "smooth, clean tone"),
    (("not whispery",), "full, clearly voiced"),
    (("not dramatic", "no movie-trailer delivery"), "even, conversational documentary delivery"),
    (("not rushed", "without hurry"), "unhurried, measured pace"),
    (("not monotone",), "gently varied intonation"),
)
"""The suggestion table of section 3.5: negated phrases and a positive rephrasing for each."""

DEFAULT_SUGGESTION: Final = "Describe the quality you want instead (e.g. 'smooth' rather than 'not rough')."
"""The table's "(no entry)" row."""

NOTE: Final = "Negated qualities tend to come out as that quality; rephrase and design again if the candidates show it."


def _fold(text: str) -> str:
    return text.casefold().replace("’", "'")


def _alnum(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "LMN"


@dataclass(frozen=True, slots=True)
class _Trigger:
    start: int
    end: int


def _triggers(text: str) -> list[_Trigger]:
    """Trigger words: maximal runs of letters, marks and digits, with an apostrophe allowed between two of
    them (so ``don't`` is one word), matched whole and case-insensitively; and the prefix ``non-``."""
    found: list[_Trigger] = []
    i, n = 0, len(text)
    while i < n:
        if not _alnum(text[i]):
            i += 1
            continue
        start = i
        while i < n and (_alnum(text[i]) or (text[i] in APOSTROPHES and i + 1 < n and _alnum(text[i + 1]))):
            i += 1
        word = _fold(text[start:i])
        if word in TRIGGER_WORDS or word.endswith(NT_SUFFIX):
            found.append(_Trigger(start, i))
        elif word == "non" and i + 1 < n and text[i] in HYPHENS and _alnum(text[i + 1]):
            found.append(_Trigger(start, i + 1))
    return found


_TABLE: Final[tuple[tuple[tuple[str, ...], str], ...]] = tuple(
    (tuple(_fold(phrase).split(" ")), suggestion) for phrases, suggestion in SUGGESTIONS for phrase in phrases
)


def _table_match(text: str, toks: tuple[Token, ...], k: int) -> tuple[int, str] | None:
    """The longest table phrase whose words are the cores of tokens k, k+1, …, with no punctuation between
    them; returns (end offset, suggestion)."""
    best: tuple[int, int, str] | None = None
    for words, suggestion in _TABLE:
        last = k + len(words) - 1
        if last >= len(toks):
            continue
        run = toks[k : last + 1]
        if any(_fold(text[t.core_start : t.core_end]) != w for t, w in zip(run, words, strict=True)):
            continue
        if any(t.core_end != t.end for t in run[:-1]) or any(t.core_start != t.start for t in run[1:]):
            continue
        if best is None or len(words) > best[0]:
            best = (len(words), run[-1].core_end, suggestion)
    return None if best is None else (best[1], best[2])


def _finding(text: str, toks: tuple[Token, ...], trigger: _Trigger) -> LintFinding:
    k = next(i for i, t in enumerate(toks) if t.start <= trigger.start < t.end)
    token = toks[k]
    end = trigger.end
    suggestion = DEFAULT_SUGGESTION
    if trigger.end < token.core_end:
        end = token.core_end  # the rest of a hyphenated word: non-breathy, no-nonsense
    elif token.core_end == token.end and k + 1 < len(toks):
        following = toks[k + 1]
        if following.has_core and following.core_start == following.start:
            end = following.core_end
    if trigger.start == token.core_start and trigger.end == token.core_end:
        match = _table_match(text, toks, k)
        if match is not None:
            end, suggestion = max(end, match[0]), match[1]
    return LintFinding(phrase=text[trigger.start : end], offset=trigger.start, suggestion=suggestion)


class NegationLinter:
    """The ``DescriptionLinter`` of section 3.5: warns, never refuses."""

    def lint(self, description: str) -> LintResult:
        """Whole-word, case-insensitive word-list findings, each with its phrase, offset and suggestion."""
        toks = tokens(description)
        findings = tuple(_finding(description, toks, t) for t in _triggers(description))
        return LintResult(policy=POLICY, findings=findings, note=NOTE)


def lint(description: str) -> LintResult:
    """``NegationLinter().lint(description)``."""
    return NegationLinter().lint(description)
