"""The Whisper cross-check of cue boundaries (design section 11.2 step 6).

At each boundary between two placed cues, the service's boundary is compared with Whisper's, where the
neighbouring words match. The neighbouring words are the last word of the cue before and the first word
of the cue after, as the spoken text has them (whether or not the aligner placed them). They match when
Whisper heard both, one after the other: the spoken words and Whisper's words are lined up with a
longest-matching-blocks alignment (``difflib``) on a plain form of each word (letters and digits, accents
dropped, lower case), and the two words must land on two consecutive Whisper words. A word of a hinted term
is never used: Whisper spells names unpredictably, and its timing of them is looser still.

Both boundaries are intervals. The service's runs from the end of the cue before to the start of the cue
after (the pause they were snapped to, or the gap between the CTC times). Whisper's runs from the end of the
last word to the start of the next. **The disagreement is the distance between the two intervals**: zero
when they overlap, otherwise the gap between them. Whisper's word times come from cross-attention and often
stretch a word into the pause after it, so comparing edge by edge would report its looseness as a
disagreement; two boundaries in the same pause agree.
"""

from __future__ import annotations

import difflib
import unicodedata
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from narration.contracts.worker import AsrWord


@dataclass(frozen=True, slots=True)
class Boundary:
    """A boundary between two placed cues: the service's interval, and the words on either side."""

    before_cue: int
    after_cue: int
    end_s: float
    start_s: float
    last_word: tuple[int, int]
    first_word: tuple[int, int]


@dataclass(frozen=True, slots=True)
class BoundaryCheck:
    """One boundary compared with Whisper: its disagreement, and Whisper's interval."""

    boundary: Boundary
    disagreement_s: float
    asr_end_s: float
    asr_start_s: float


def plain(text: str) -> str:
    """A word's plain form for matching: letters and digits only, accents dropped, lower case."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed.casefold() if ch.isalnum() and not unicodedata.combining(ch))


def interval_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """The gap between two intervals (each given in either order); zero when they overlap or touch."""
    a_lo, a_hi = sorted(a)
    b_lo, b_hi = sorted(b)
    return max(0.0, b_lo - a_hi, a_lo - b_hi)


def cross_check(
    boundaries: Sequence[Boundary],
    spoken_words: Sequence[tuple[int, int, str]],
    asr_words: Sequence[AsrWord],
    names: Collection[tuple[int, int]],
) -> tuple[BoundaryCheck, ...]:
    """Every boundary that can be compared with Whisper, with its disagreement (section 11.2 step 6).

    ``spoken_words`` are the segment's words in order as (cue, word, text); ``names`` the (cue, word) pairs
    of hinted terms, which are never used. A boundary whose neighbouring words Whisper did not hear one
    after the other, or heard without times, is not compared.
    """
    if not boundaries or not asr_words:
        return ()
    spoken_plain = [plain(text) for _, _, text in spoken_words]
    asr_plain = [plain(w["text"]) for w in asr_words]
    matcher = difflib.SequenceMatcher(a=spoken_plain, b=asr_plain, autojunk=False)
    heard_at: dict[tuple[int, int], int] = {}
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            cue, word, _ = spoken_words[block.a + k]
            if spoken_plain[block.a + k]:
                heard_at[(cue, word)] = block.b + k
    checks: list[BoundaryCheck] = []
    for boundary in boundaries:
        if boundary.last_word in names or boundary.first_word in names:
            continue
        i, j = heard_at.get(boundary.last_word), heard_at.get(boundary.first_word)
        if i is None or j is None or j != i + 1:
            continue
        asr_end, asr_start = asr_words[i]["end_s"], asr_words[j]["start_s"]
        if asr_end is None or asr_start is None:
            continue
        distance = interval_distance((boundary.end_s, boundary.start_s), (asr_end, asr_start))
        checks.append(BoundaryCheck(boundary, round(distance, 3), asr_end, asr_start))
    return tuple(checks)
