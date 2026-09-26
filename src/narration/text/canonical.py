"""Canonical form, words, the join and spoken length (design section 7.2).

These are the service's single definitions: the text pipeline, the aligner's transcript (WP15) and the QA
word ranges (WP14) all split words with ``words`` so that a word index means the same thing everywhere.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from .rules import WHITESPACE


def is_whitespace(ch: str) -> bool:
    """Whether a character is whitespace for canonical form and word splitting (``rules.WHITESPACE``)."""
    return ch in WHITESPACE


def is_punctuation(ch: str) -> bool:
    """Unicode general category P* (connector, dash, open, close, initial, final, other punctuation)."""
    return unicodedata.category(ch).startswith("P")


def canonical_form(text: str) -> str:
    """NFC, every run of whitespace made one space (U+0020), and the ends trimmed (section 7.2).

    Nothing else changes: no punctuation or capital is added or removed, so a sentence may run across two
    cues. Refused characters (``narration.text.sanitise``) are not this function's concern; it leaves them.
    """
    nfc = unicodedata.normalize("NFC", text)
    out: list[str] = []
    pending_space = False
    for ch in nfc:
        if ch in WHITESPACE:
            pending_space = bool(out)
            continue
        if pending_space:
            out.append(" ")
            pending_space = False
        out.append(ch)
    collapsed = "".join(out)
    # Replacing one whitespace character by another, or trimming, cannot un-normalise a string in practice;
    # normalising again costs nothing when it is already NFC and keeps the result NFC by construction.
    return collapsed if unicodedata.is_normalized("NFC", collapsed) else unicodedata.normalize("NFC", collapsed)


@dataclass(frozen=True, slots=True)
class Token:
    """A maximal run of non-whitespace characters, ``[start, end)`` in code points.

    ``[core_start, core_end)`` is the token with its leading and trailing punctuation (category P*) set
    aside; it is empty (``core_start == core_end``) for a token of punctuation only, such as a lone dash.
    """

    start: int
    end: int
    core_start: int
    core_end: int

    @property
    def has_core(self) -> bool:
        return self.core_end > self.core_start


@dataclass(frozen=True, slots=True)
class Word:
    """A word (section 7.2): a whitespace-separated token with its leading and trailing punctuation set aside.

    ``index`` counts words only; a token of punctuation only is not a word. ``text`` is the core, so
    ``forty-eight`` is one word and ``seven`` in ``seven.`` is a whole word.
    """

    index: int
    start: int
    end: int
    core_start: int
    core_end: int
    text: str


def tokens(text: str) -> tuple[Token, ...]:
    """Every whitespace-separated token of ``text``, with its core."""
    out: list[Token] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] in WHITESPACE:
            i += 1
            continue
        start = i
        while i < n and text[i] not in WHITESPACE:
            i += 1
        core_start, core_end = start, i
        while core_start < core_end and is_punctuation(text[core_start]):
            core_start += 1
        while core_end > core_start and is_punctuation(text[core_end - 1]):
            core_end -= 1
        out.append(Token(start, i, core_start, core_end))
    return tuple(out)


def words(text: str) -> tuple[Word, ...]:
    """The words of ``text`` in order (tokens with a non-empty core), numbered from 0.

    Word indices do not change under canonical form, because it changes only whitespace and Unicode form:
    the same words come out of a cue as sent and of its canonical text.
    """
    return tuple(
        Word(index, t.start, t.end, t.core_start, t.core_end, text[t.core_start : t.core_end])
        for index, t in enumerate(t for t in tokens(text) if t.has_core)
    )


def join(texts: Iterable[str]) -> str:
    """The join (section 7.2): texts joined by one space. Nothing is added at a join."""
    return " ".join(texts)


def spoken_length(text: str) -> int:
    """Spoken length (``spoken_chars``, section 7.2): length in Unicode code points, not UTF-16 units."""
    return len(text)
