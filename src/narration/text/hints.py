"""Pronunciation hints inside a cue (design section 9.1 step 3.1) and terms split across cues (step 5).

A term matches case-sensitively, in the cue's canonical text, where it stands as whole words: bounded by
the cue's start or end, a space, or punctuation (category P*, the apostrophe and the hyphen included). So
``Gastrella`` matches in ``Gastrella's`` and in ``Gastrella-side``, and only the term is replaced.

Terms are applied **longest first** (ties by the term's code points), each claiming the places it matches
left to right; a later, shorter term never matches inside a place already claimed. A term with no
respelling still claims its places and is recorded, but leaves the engine text as the spoken text.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.models import Hint, HintApplied

from .canonical import canonical_form, is_punctuation

Origin = tuple[int, int | None]
"""Where one engine-text code point comes from: (offset in the cue's spoken text, the index of the hint
application whose respelling it is, or None for a code point of the spoken text itself)."""


@dataclass(frozen=True, slots=True)
class PreparedHint:
    """A request's hint with its term and respelling in canonical form. ``index`` is its place in ``hints``."""

    index: int
    term: str
    respell: str | None


@dataclass(frozen=True, slots=True)
class CueHints:
    """A cue after its hints: the engine text, what was applied, and each engine code point's origin."""

    engine: str
    applied: tuple[HintApplied, ...]
    origin: tuple[Origin, ...]
    respelled_terms: tuple[str, ...]
    """For each application index in ``origin``, the term it respelled."""


@dataclass(frozen=True, slots=True)
class SplitTerm:
    """A term that matches only across a cue boundary (``TERM_SPLIT_ACROSS_CUES``).

    ``first_cue``..``last_cue`` are the cues it runs across; ``offset`` is its start in the first cue's
    spoken text.
    """

    term: str
    first_cue: int
    last_cue: int
    offset: int


def prepare_hints(hints: Sequence[Hint]) -> tuple[PreparedHint, ...]:
    """Canonicalise every hint's term and respelling; refuse what cannot be applied unambiguously.

    Raises ``NarrationError(INVALID_ARGUMENT)`` naming the field for a term that is only whitespace, a
    respelling that is empty or only whitespace (it would silence the term; leave ``respell`` out to keep
    the term as written), and a term sent twice (one request never mixes two renderings of a term).
    """
    prepared: list[PreparedHint] = []
    first_index: dict[str, int] = {}
    for i, hint in enumerate(hints):
        term = canonical_form(hint.term)
        if not term:
            raise NarrationError(
                codes.INVALID_ARGUMENT,
                f"hints[{i}].term has no characters other than whitespace",
                field=f"hints[{i}].term",
                hint="Send the term as it appears in the spoken text.",
            )
        respell = None if hint.respell is None else canonical_form(hint.respell)
        if respell == "":
            raise NarrationError(
                codes.INVALID_ARGUMENT,
                f"hints[{i}].respell for '{term}' is empty, which would silence the term",
                field=f"hints[{i}].respell",
                hint="Give the respelling the engine should read, or leave respell out to keep the term as written.",
            )
        if term in first_index:
            raise NarrationError(
                codes.INVALID_ARGUMENT,
                f"hints[{i}].term '{term}' is already hints[{first_index[term]}].term",
                field=f"hints[{i}].term",
                hint="Send each term once, with the one respelling it should have in this request.",
            )
        first_index[term] = i
        prepared.append(PreparedHint(i, term, respell))
    return tuple(prepared)


def matching_order(hints: Sequence[PreparedHint]) -> tuple[PreparedHint, ...]:
    """Longest term first; terms of equal length in code-point order, so the result never depends on the
    order the request listed them in."""
    return tuple(sorted(hints, key=lambda h: (-len(h.term), h.term)))


def _is_boundary(ch: str) -> bool:
    return ch == " " or is_punctuation(ch)


def _stands_alone(text: str, start: int, end: int) -> bool:
    return (start == 0 or _is_boundary(text[start - 1])) and (end == len(text) or _is_boundary(text[end]))


def apply_hints(spoken: str, ordered: Sequence[PreparedHint]) -> CueHints:
    """Apply hints (in ``matching_order``) inside one cue's canonical text.

    The engine text takes each respelling in place of its term; the spoken text is unchanged. Each
    application is recorded as ``HintApplied`` with its offset in the spoken text, in order of offset.
    """
    claimed: list[tuple[int, int, PreparedHint]] = []
    for hint in ordered:
        n = len(hint.term)
        i = spoken.find(hint.term)
        while i != -1:
            end = i + n
            if _stands_alone(spoken, i, end) and all(end <= s or i >= e for s, e, _ in claimed):
                claimed.append((i, end, hint))
                i = spoken.find(hint.term, end)
            else:
                i = spoken.find(hint.term, i + 1)
    claimed.sort(key=lambda c: c[0])

    pieces: list[str] = []
    origin: list[Origin] = []
    respelled: list[str] = []
    pos = 0
    for start, end, hint in claimed:
        pieces.append(spoken[pos:start])
        origin.extend((j, None) for j in range(pos, start))
        if hint.respell is None:
            pieces.append(spoken[start:end])
            origin.extend((j, None) for j in range(start, end))
        else:
            pieces.append(hint.respell)
            origin.extend((start, len(respelled)) for _ in hint.respell)
            respelled.append(hint.term)
        pos = end
    pieces.append(spoken[pos:])
    origin.extend((j, None) for j in range(pos, len(spoken)))
    applied = tuple(HintApplied(term=h.term, respell=h.respell, offset=s) for s, _, h in claimed)
    return CueHints("".join(pieces), applied, tuple(origin), tuple(respelled))


def split_terms(spoken_cues: Sequence[str], hints: Sequence[PreparedHint]) -> tuple[SplitTerm, ...]:
    """Terms that would match, as whole words, only in the join of the cues, across a cue boundary.

    Such a term is never applied (section 9.1 step 5, R3); each place it would match is reported once.
    """
    if len(spoken_cues) < 2:
        return ()
    starts: list[int] = []
    pos = 0
    for text in spoken_cues:
        starts.append(pos)
        pos += len(text) + 1
    boundaries = [start - 1 for start in starts[1:]]  # the index of each joining space
    joined = " ".join(spoken_cues)

    def cue_at(offset: int) -> int:
        cue = 0
        while cue + 1 < len(starts) and starts[cue + 1] <= offset:
            cue += 1
        return cue

    found: list[SplitTerm] = []
    for hint in hints:
        if " " not in hint.term:
            continue
        n = len(hint.term)
        i = joined.find(hint.term)
        while i != -1:
            end = i + n
            if any(i < b < end for b in boundaries) and _stands_alone(joined, i, end):
                first = cue_at(i)
                found.append(SplitTerm(hint.term, first, cue_at(end - 1), i - starts[first]))
            i = joined.find(hint.term, i + 1)
    found.sort(key=lambda s: (s.first_cue, s.offset, s.term))
    return tuple(found)
