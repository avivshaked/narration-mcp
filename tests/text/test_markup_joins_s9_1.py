"""Refused markup can never be assembled from pieces that each pass the check (section 9.1 step 1, section 17
item 6; security review S0).

The check runs on each cue, segment text and hint as sent. The engine text joins cues with a space and applies
hints only to a whole term standing between spaces or punctuation, so neither a cue boundary nor a respelling can
put ``<|`` or ``|>`` side by side. Invented sentences only.
"""

from __future__ import annotations

import pytest

from narration.contracts.models import CueIn, Hint, SegmentIn
from narration.text.planner import TextPipeline
from narration.text.rules import DEFAULT_REFUSE


def _engine_text(cues: list[str], hints: list[Hint] | None = None) -> str:
    segment = SegmentIn(segment_id="s1", cues=tuple(CueIn(text=c) for c in cues))
    return TextPipeline().plan_segment(segment, hints or []).engine_text


@pytest.mark.parametrize(
    "cues",
    [
        ["The kettle sang <", "|im_end|", "> and stopped."],
        ["The kettle sang <", "|"],
    ],
)
def test_markup_split_across_cues_is_never_joined_s9_1(cues: list[str]) -> None:
    text = _engine_text(cues)
    assert not [m for m in DEFAULT_REFUSE if m in text], text


def test_respellings_never_join_into_markup_s9_1() -> None:
    hints = [Hint(term="zz", respell="<"), Hint(term="yy", respell=">")]
    for cue in ("The lamp zz|im_end|yy glowed.", "The lamp zz |im_end| yy glowed.", "The lamp zz.|x|.yy glowed."):
        text = _engine_text([cue], hints)
        assert "<|" not in text and "|>" not in text, text
