"""Exact spans → word ranges (design section 7.2, R14).

A span is ``[start, end)`` in code points into the cue's text **as sent**. It must lie inside the cue,
cover at least one word, not overlap another span of the cue, and not cut a word: neither end may fall
strictly inside a word's core (a word is a whitespace-separated token with its leading and trailing
punctuation set aside, ``canonical.words``). An end may therefore fall anywhere in the punctuation or
whitespace around a word: ``seven`` and ``seven.`` are both whole, and so is ``(forty)``.

The service keeps a span as the range of word indices it covers, ``[first, end)``, which canonical form
does not change.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.models import ExactSpan, ExactWords

from .canonical import words

_SPAN_HINT = "Mark whole words: start and end must fall between words or on the punctuation around one."


def resolve_exact(received: str, spans: Sequence[ExactSpan], field: str) -> tuple[ExactWords, ...]:
    """Each span of one cue, with its word range; ``field`` names the cue (e.g. ``segments[0].cues[1]``).

    Raises ``NarrationError(INVALID_ARGUMENT)`` naming ``<field>.exact[<j>]`` for a span that is empty,
    outside the cue, covers no word, cuts a word, or overlaps an earlier-starting span.
    """
    cue_words = words(received)
    n = len(received)
    out: list[ExactWords] = []
    for j, span in enumerate(spans):
        where = f"{field}.exact[{j}]"
        if not 0 <= span.start < span.end <= n:
            raise NarrationError(
                codes.INVALID_ARGUMENT,
                f"{where} [{span.start}, {span.end}) is not a non-empty range inside the cue, "
                f"which has {n} code points",
                field=where,
                hint="Give start < end, both counted in Unicode code points into the cue's text as sent.",
                details={"start": span.start, "end": span.end, "cue_code_points": n},
            )
        for w in cue_words:
            for edge in (span.start, span.end):
                if w.core_start < edge < w.core_end:
                    raise NarrationError(
                        codes.INVALID_ARGUMENT,
                        f"{where} [{span.start}, {span.end}) cuts the word '{w.text}' at code point {edge}",
                        field=where,
                        hint=_SPAN_HINT,
                        details={"start": span.start, "end": span.end, "word": w.text, "word_index": w.index},
                    )
        covered = [w.index for w in cue_words if span.start <= w.core_start and w.core_end <= span.end]
        if not covered:
            raise NarrationError(
                codes.INVALID_ARGUMENT,
                f"{where} [{span.start}, {span.end}) covers no word",
                field=where,
                hint=_SPAN_HINT,
                details={"start": span.start, "end": span.end},
            )
        out.append(ExactWords(start=span.start, end=span.end, words=(covered[0], covered[-1] + 1)))

    order = sorted(range(len(spans)), key=lambda j: (spans[j].start, spans[j].end, j))
    for earlier, later in pairwise(order):
        if spans[later].start < spans[earlier].end:
            where = f"{field}.exact[{later}]"
            raise NarrationError(
                codes.INVALID_ARGUMENT,
                f"{where} [{spans[later].start}, {spans[later].end}) overlaps {field}.exact[{earlier}] "
                f"[{spans[earlier].start}, {spans[earlier].end})",
                field=where,
                hint="Exact spans of one cue must not overlap; merge them into one span or mark separate words.",
                details={"span": later, "overlaps": earlier},
            )
    return tuple(out)
