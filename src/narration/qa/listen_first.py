"""What to listen to first (design section 11.1, "Report and listen-first").

``listen_first`` is what QA found, in this order:

1. fails;
2. exact-span and term flags;
3. cue-alignment flags;
4. insertion, similarity and pace warnings, and consistency outliers;
5. cues with a text warning, and segments over the voice's reliable length.

What is listed (this module's reading of the design):

* For each segment, the flags of its **suggested take** (its ``qa.flags``, ``alignment.flags`` and the take's
  own ``flags``), because that is the take a caller hears first; every take's flags stay in its ``qa`` and in
  ``report.md``. A segment with no suggested take lists only its segment-level items.
* The segment's own ``flags`` (``SEGMENT_TOO_LONG``, ``TERM_SPLIT_ACROSS_CUES``) and each cue's text warnings.
* Other QA warnings (``WER_HIGH``, ``SILENCE_LONG``, ``CLIPPING``) go with group 4. Info flags are left out,
  except ``SPK_OUTLIER`` (group 4) and a lone letter's text warning (group 5), which exist for a listener.
  Execution errors (severity ``error``) have nothing to hear and are left out.

Items are ordered by group, then by the segment's position in the request, then by cue.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence

from narration.contracts import codes
from narration.contracts.models import CueTiming, Flag, ListenFirstItem, SegmentResult, TakeResult

__all__ = ["group_of", "listen_first"]

_EXACT_AND_TERMS = frozenset({codes.EXACT_SPAN_MISMATCH, codes.TERM_UNVERIFIED})
_CUE_ALIGNMENT = frozenset({codes.CUE_UNALIGNED, codes.CUE_LOW_CONFIDENCE, codes.CUE_ALIGNMENT_DISAGREE})
_INSERTION_SIMILARITY_PACE = frozenset(
    {codes.HEAD_INSERTION, codes.END_INSERTION, codes.SPK_SIM_LOW, codes.PACE_FAST, codes.PACE_SLOW}
)
_TEXT = frozenset({codes.WRITTEN_FORM_TOKEN, codes.TERM_SPLIT_ACROSS_CUES, codes.SEGMENT_TOO_LONG})


def group_of(flag: Flag) -> int | None:
    """The flag's group (1–5) in section 11.1's order, or None if it is not listed."""
    if flag.severity == "error":
        return None
    if flag.severity == "fail":
        return 1
    if flag.code in _EXACT_AND_TERMS:
        return 2
    if flag.code in _CUE_ALIGNMENT and flag.severity == "warn":
        return 3
    if flag.code == codes.SPK_OUTLIER or (flag.code in _INSERTION_SIMILARITY_PACE and flag.severity == "warn"):
        return 4
    if flag.code in _TEXT:
        return 5
    if flag.severity == "warn":
        return 4
    return None


def _cue(take: TakeResult, index: int | None) -> CueTiming | None:
    if index is None:
        return None
    return next((c for c in take.cues if c.index == index), None)


def _window(flag: Flag, take: TakeResult | None) -> tuple[float | None, float | None]:
    """Where in the take to listen: the cue's times, the head or the tail, or the whole take."""
    if take is None:
        return None, None
    duration = take.delivery.duration_s
    if flag.code == codes.HEAD_INSERTION:
        first = _cue(take, flag.cue if flag.cue is not None else 0)
        return 0.0, first.end_s if first is not None and first.end_s is not None else duration
    if flag.code == codes.END_INSERTION:
        last = _cue(take, flag.cue)
        return (last.start_s if last is not None and last.start_s is not None else 0.0), duration
    cue = _cue(take, flag.cue)
    if flag.cue is not None:
        return (cue.start_s, cue.end_s) if cue is not None else (None, None)
    return 0.0, duration


def _take_flags(take: TakeResult) -> Iterable[Flag]:
    if take.qa is not None:
        yield from take.qa.flags
    if take.alignment is not None:
        yield from take.alignment.flags
    yield from take.flags


def listen_first(segments: Sequence[SegmentResult]) -> tuple[ListenFirstItem, ...]:
    """The listen-first list for a job's segments, in section 11.1's order."""
    ranked: list[tuple[int, int, int, str, ListenFirstItem]] = []
    for position, segment in enumerate(segments):
        take = next((t for t in segment.takes if t.take_id == segment.suggested_take_id), None)
        found: list[tuple[Flag, TakeResult | None]] = []
        if take is not None:
            found += [(f, take) for f in _take_flags(take)]
        found += [(f, take) for f in segment.flags]
        for cue in segment.text.cues:
            found += [(f if f.cue is not None else dataclasses.replace(f, cue=cue.index), take) for f in cue.warnings]
        seen: set[tuple[str, str, int | None, str]] = set()
        for flag, source in found:
            group = group_of(flag)
            key = (flag.code, flag.severity, flag.cue, flag.message)
            if group is None or key in seen:
                continue
            seen.add(key)
            start, end = _window(flag, source)
            item = ListenFirstItem(
                segment_id=segment.segment_id,
                take_id=source.take_id if source is not None else None,
                cue=flag.cue,
                reason=f"{flag.code} ({flag.severity}): {flag.message}",
                from_s=start,
                to_s=end,
            )
            ranked.append((group, position, -1 if flag.cue is None else flag.cue, flag.code, item))
    ranked.sort(key=lambda r: r[:4])
    return tuple(r[4] for r in ranked)
