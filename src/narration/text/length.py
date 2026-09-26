"""The over-long segment warning (design sections 3.2, 7.3, R8).

A voice's ``max_segment_chars`` is advice, never a refusal: a longer segment is rendered, and flagged
``SEGMENT_TOO_LONG`` (warn) with the voice's limits, the segment's spoken length, how far over it is, and
each cue's spoken length, so a caller that splits knows where it stands. How long a paragraph is remains
the caller's decision.
"""

from __future__ import annotations

from narration.contracts import codes
from narration.contracts.models import Flag, SegmentText


def segment_too_long(
    segment: SegmentText, *, max_segment_chars: int, max_segment_seconds: float | None = None
) -> Flag | None:
    """``SEGMENT_TOO_LONG`` (warn) when the segment's spoken length is over the voice's reliable length,
    else None. It never raises: the segment is rendered either way."""
    if segment.spoken_chars <= max_segment_chars:
        return None
    return Flag(
        code=codes.SEGMENT_TOO_LONG,
        severity="warn",
        segment_id=segment.segment_id,
        message=(
            f"{segment.segment_id} is {segment.spoken_chars} spoken characters; this voice read up to "
            f"{max_segment_chars} reliably. It will be rendered."
        ),
        details={
            "max_segment_chars": max_segment_chars,
            "max_segment_seconds": max_segment_seconds,
            "spoken_chars": segment.spoken_chars,
            "over_by_chars": segment.spoken_chars - max_segment_chars,
            "cue_chars": [len(cue.spoken) for cue in segment.cues],
        },
    )
