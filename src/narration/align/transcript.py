"""The aligner's transcript and its guard (design section 11.2 steps 1 and 3).

The transcript is the segment's **spoken** text, cue by cue and word by word, in the model's alphabet
(``alphabet.spell``), with ``|`` between words. A word is what ``narration.text.words`` says it is, so the
token → (cue, word) map uses the same word indices as exact spans and QA (section 7.2).

A word that cannot be spelled in the alphabet (a digit or a symbol in it) is listed in ``dropped``. Each run
of such words inside one cue becomes one wildcard token, ``*`` (plan.md DC-11), so the speech the words stand
for has a token to go to; without it, that speech pulled the neighbouring words off by up to 1.8 s (spike
(b)). The wildcard maps to the run's first word; ``wildcard_runs`` gives every word of the run. A run never
crosses a cue boundary.

A hinted term is spelled from its ``align_as`` letters if the hint gives them, otherwise from its spoken
letters (never from its respelling, which is the engine's). A term may cover several words, and may start
or end inside one (``Brindlewick`` in ``Brindlewick's``). ``align_as`` is split at whitespace into parts, and
the parts go to the term's words in order: part *i* to word *i*, and every part left over to the term's
last word. A word of the term that gets no part has no letters and is listed in ``dropped``, but gets no
wildcard: the term's other words already spell its speech. Whatever the word has outside the term (``'s``,
a prefix before a hyphen) is spelled from the spoken text.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from math import gcd
from typing import Final

from narration.contracts.interfaces import AlignTranscript
from narration.contracts.models import Hint, SegmentText
from narration.text import Word, canonical_form, words

from .alphabet import WILDCARD, WORD_SEPARATOR, spell

SAMPLE_RATE: Final = 16_000
"""The aligner's input rate: the worker resamples every take to it (section 11.2)."""
FRAME_STRIDE: Final = 320
FRAME_WINDOW: Final = 400
"""wav2vec2's feature encoder: a frame every 320 samples, each seeing 400 (``narration_worker_qa.align``)."""


def repeats(tokens: Sequence[str]) -> int:
    """``R``: how many tokens equal the token before them. CTC needs a blank frame between the two."""
    return sum(1 for a, b in itertools.pairwise(tokens) if a == b)


def has_letters(transcript: AlignTranscript) -> bool:
    """Whether any token is a word's letter: a transcript of wildcards alone places no cue (DC-11, DC-12)."""
    return any(
        token != WILDCARD and owner is not None
        for token, owner in zip(transcript.tokens, transcript.token_words, strict=True)
    )


def guard(transcript: AlignTranscript, num_frames: int) -> bool:
    """``T ≥ L + R`` (section 11.2 step 3): frames at least tokens plus repeats, and a letter to align.

    With no letter there is nothing to send to the worker: ``CtcAligner.resolve`` without a reply makes every
    cue ``no_alignable_words``, and raises no ``ALIGNMENT_ERROR``.
    """
    return has_letters(transcript) and num_frames >= len(transcript.tokens) + repeats(transcript.tokens)


def ctc_frames(samples: int, sample_rate: int) -> int:
    """How many emission frames the worker computes for a take of ``samples`` samples at ``sample_rate``.

    The worker resamples to 16 kHz with ``resample_poly`` (``ceil(samples · up / down)`` samples), then
    the encoder gives a frame for every 320 samples after the first 400.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, not {sample_rate}")
    g = gcd(SAMPLE_RATE, sample_rate)
    n = -(-samples * (SAMPLE_RATE // g) // (sample_rate // g))
    return 0 if n < FRAME_WINDOW else (n - FRAME_WINDOW) // FRAME_STRIDE + 1


def _align_as_parts(hint: Hint) -> tuple[str, ...] | None:
    if hint.align_as is None:
        return None
    parts = tuple(canonical_form(hint.align_as).split(" "))
    return parts if parts != ("",) else None


def _align_as_by_term(hints: Sequence[Hint]) -> dict[str, tuple[str, ...]]:
    """Each canonical term's ``align_as`` parts, for the terms that have them; independent of the order."""
    given: dict[str, set[tuple[str, ...] | None]] = {}
    for hint in hints:
        given.setdefault(canonical_form(hint.term), set()).add(_align_as_parts(hint))
    by_term: dict[str, tuple[str, ...]] = {}
    for term, options in given.items():
        if len(options) > 1:
            raise ValueError(f"the hints give the term {term!r} more than one align_as; send each term once")
        (parts,) = options
        if parts is not None:
            by_term[term] = parts
    return by_term


def _word_sources(
    text: str, cue_words: Sequence[Word], terms: Sequence[tuple[int, int, tuple[str, ...]]]
) -> tuple[list[str], set[int]]:
    """Each word's text to spell: its core, with every ``align_as`` term replaced by its parts; and the words
    of a term that got no part, whose speech the term's other words already spell."""
    inserts: dict[int, list[tuple[int, int, str]]] = {}
    for start, end, parts in terms:
        touched = [w for w in cue_words if w.core_start < end and w.core_end > start]
        for i, word in enumerate(touched):
            given = parts[i : i + 1] if i < len(touched) - 1 else parts[i:]
            inserts.setdefault(word.index, []).append(
                (max(start, word.core_start), min(end, word.core_end), "-".join(given))
            )
    absorbed = {index for index, pieces in inserts.items() if any(not letters for _, _, letters in pieces)}
    sources: list[str] = []
    for word in cue_words:
        pieces: list[str] = []
        pos = word.core_start
        for start, end, letters in sorted(inserts.get(word.index, ())):
            pieces.append(text[pos:start])
            pieces.append(letters)
            pos = end
        pieces.append(text[pos : word.core_end])
        sources.append("".join(pieces))
    return sources, absorbed


def build_transcript(segment: SegmentText, hints: Sequence[Hint]) -> AlignTranscript:
    """The aligner's transcript of ``segment`` (section 11.2 step 1), with its token → (cue, word) map.

    ``hints`` are the hints used in the segment; only their ``align_as`` matters here. Their order does not:
    hints are a set (the analysis key sorts them), so the transcript is the same for any order. Two hints of
    one term with different ``align_as`` raise ``ValueError``; the text planner refuses a term sent twice
    before this can see one. A word whose letters fall outside the alphabet is listed in ``dropped``, and
    each run of them in a cue is one ``*``; the ``|`` separators join the words and wildcards.
    """
    by_term = _align_as_by_term(hints)
    tokens: list[str] = []
    token_words: list[tuple[int, int] | None] = []
    all_words: list[tuple[int, int, str]] = []
    dropped: list[tuple[int, int, str]] = []
    term_words: set[tuple[int, int]] = set()

    def separate() -> None:
        if tokens:
            tokens.append(WORD_SEPARATOR)
            token_words.append(None)

    for cue in segment.cues:
        cue_words = words(cue.spoken)
        for applied in cue.hints_applied:
            start, end = applied.offset, applied.offset + len(applied.term)
            term_words.update((cue.index, w.index) for w in cue_words if w.core_start < end and w.core_end > start)
        terms = [
            (applied.offset, applied.offset + len(applied.term), parts)
            for applied in cue.hints_applied
            if (parts := by_term.get(canonical_form(applied.term))) is not None
        ]
        in_run = False
        sources, absorbed = _word_sources(cue.spoken, cue_words, terms)
        for word, source in zip(cue_words, sources, strict=True):
            entry = (cue.index, word.index, word.text)
            all_words.append(entry)
            spelled = spell(source)
            if spelled is None:
                dropped.append(entry)
                if word.index in absorbed:  # its term's letters already cover its speech: no wildcard
                    in_run = False
                elif not in_run:
                    separate()
                    tokens.append(WILDCARD)
                    token_words.append((cue.index, word.index))
                    in_run = True
                continue
            in_run = False
            separate()
            for token in spelled:
                tokens.append(token)
                token_words.append(None if token == WORD_SEPARATOR else (cue.index, word.index))
    return AlignTranscript(
        tokens=tuple(tokens),
        token_words=tuple(token_words),
        words=tuple(all_words),
        cue_count=len(segment.cues),
        term_words=tuple(sorted(term_words)),
        dropped=tuple(dropped),
    )


def wildcard_runs(transcript: AlignTranscript) -> dict[int, tuple[tuple[int, int], ...]]:
    """For each ``*`` token's index, the (cue, word) pairs of the run it stands for: its first word and the
    left-out words that follow it in the same cue."""
    starts = {
        owner for token, owner in zip(transcript.tokens, transcript.token_words, strict=True) if token == WILDCARD
    }
    left_out = {(cue, word) for cue, word, _ in transcript.dropped} - starts
    runs: dict[int, tuple[tuple[int, int], ...]] = {}
    for i, (token, owner) in enumerate(zip(transcript.tokens, transcript.token_words, strict=True)):
        if token != WILDCARD or owner is None:
            continue
        cue, word = owner
        run = [owner]
        while (cue, word + len(run)) in left_out:
            run.append((cue, word + len(run)))
        runs[i] = tuple(run)
    return runs
