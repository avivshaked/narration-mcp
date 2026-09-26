"""Sanitising (design section 9.1 step 1, section 17 items 5–6): markup and control characters are refused
with ``TEXT_REFUSED``, and every offender is listed."""

from __future__ import annotations

from typing import Any

import pytest

from narration.config import TextConfig
from narration.contracts.errors import NarrationError
from narration.contracts.models import CueIn, Hint, SegmentIn
from narration.text import TextPipeline


def _refused(*cues: str, hints: tuple[Hint, ...] = (), text: str | None = None) -> NarrationError:
    segment = SegmentIn(segment_id="s", cues=tuple(CueIn(text=c) for c in cues), text=text)
    with pytest.raises(NarrationError) as caught:
        TextPipeline().plan_request([segment], hints, strict_text=False)
    assert caught.value.code == "TEXT_REFUSED"
    assert caught.value.retryable is False
    return caught.value


def _offenders(error: NarrationError) -> list[dict[str, Any]]:
    assert error.details is not None
    return error.details["offenders"]


def test_markup_square_brackets_are_refused_s9_1() -> None:
    error = _refused("[laughs] Then we left.")
    got = [(o["text"], o["offset"], o["reason"]) for o in _offenders(error)]
    assert got == [("[", 0, "markup"), ("]", 7, "markup")]
    assert error.field == "segments[0].cues[0].text"


def test_markup_special_token_delimiters_are_refused_s9_1() -> None:
    error = _refused("Say <|endoftext|> now.")
    assert [(o["text"], o["offset"]) for o in _offenders(error)] == [("<|", 4), ("|>", 15)]


def test_markup_overlapping_delimiters_are_each_listed_s9_1() -> None:
    assert [(o["text"], o["offset"]) for o in _offenders(_refused("a <|> b"))] == [("<|", 2), ("|>", 3)]


def test_lone_pipe_and_angle_bracket_are_not_refused_s9_1() -> None:
    planned = TextPipeline().plan_segment(SegmentIn(segment_id="s", cues=(CueIn(text="left | right < here"),)), [])
    assert [w.details["kind"] for w in planned.cues[0].warnings if w.details] == ["symbol", "symbol"]


def test_markup_in_any_cue_and_every_segment_is_listed_together_s9_1() -> None:
    segments = [
        SegmentIn(segment_id="one", cues=(CueIn(text="A clean cue."), CueIn(text="Then [pause] more."))),
        SegmentIn(segment_id="two", text="And <|this|> too."),
    ]
    with pytest.raises(NarrationError) as caught:
        TextPipeline().plan_request(segments, [], strict_text=False)
    got = [(o["field"], o["segment_id"], o["cue"], o["text"]) for o in _offenders(caught.value)]
    assert got == [
        ("segments[0].cues[1].text", "one", 1, "["),
        ("segments[0].cues[1].text", "one", 1, "]"),
        ("segments[1].text", "two", 0, "<|"),
        ("segments[1].text", "two", 0, "|>"),
    ]


@pytest.mark.parametrize(
    "char",
    ["\u0000", "\u0007", "\u000b", "\u000c", "\u001b", "\u001c", "\u001f", "\u007f", "\u0085", "\u009b"],
    ids=lambda c: f"U+{ord(c):04X}",
)
def test_control_characters_are_refused_s9_1(char: str) -> None:
    (offender,) = _offenders(_refused(f"Ring the{char} bell."))
    assert (offender["text"], offender["offset"], offender["reason"]) == (char, 8, "control")
    assert offender["codepoints"] == f"U+{ord(char):04X}"


def test_lone_surrogate_is_refused_s9_1() -> None:
    (offender,) = _offenders(_refused("Ring the\ud800 bell."))
    assert (offender["offset"], offender["reason"]) == (8, "surrogate")


@pytest.mark.parametrize("char", ["​", "‍", "﻿", "­", "‮"], ids=lambda c: f"U+{ord(c):04X}")
def test_format_characters_are_symbol_warnings_not_refused_s9_1(char: str) -> None:
    segment = SegmentIn(segment_id="s", cues=(CueIn(text=f"Ring{char} the bell."),))
    planned = TextPipeline().plan_segment(segment, [])
    (warning,) = planned.cues[0].warnings
    assert warning.details == {"token": char, "kind": "symbol", "offset": 4}
    assert f"U+{ord(char):04X}" in warning.message, "an invisible character is named by its code point"
    assert planned.cues[0].spoken == f"Ring{char} the bell.", "the text is spoken as sent"


def test_markup_in_segment_text_is_refused_even_with_clean_cues_s9_1() -> None:
    error = _refused("A clean cue.", text="A [clean] cue.")
    assert {o["field"] for o in _offenders(error)} == {"segments[0].text"}


def test_markup_in_a_respelling_is_refused_s17_6() -> None:
    error = _refused("Ossavine rowed out.", hints=(Hint(term="Ossavine", respell="<|Oss|>-a-veen"),))
    assert [(o["field"], o["text"]) for o in _offenders(error)] == [
        ("hints[0].respell", "<|"),
        ("hints[0].respell", "|>"),
    ]
    assert error.field == "hints[0].respell"


def test_the_refused_markup_comes_from_config_s16() -> None:
    pipeline = TextPipeline(TextConfig(refuse=("{{", "}}")))
    segment = SegmentIn(segment_id="s", cues=(CueIn(text="Say {{name}} [now]."),))
    with pytest.raises(NarrationError) as caught:
        pipeline.plan_segment(segment, [])
    assert [o["text"] for o in _offenders(caught.value)] == ["{{", "}}"]
    planned = TextPipeline(TextConfig(refuse=())).plan_segment(segment, [])
    assert planned.spoken_text == "Say {{name}} [now]."


def test_plan_segment_fields_take_the_given_prefix_s9_1() -> None:
    segment = SegmentIn(segment_id="s", cues=(CueIn(text="ok"), CueIn(text="[x]")))
    with pytest.raises(NarrationError) as caught:
        TextPipeline().plan_segment(segment, [], field_prefix="segments[4].")
    assert caught.value.field == "segments[4].cues[1].text"
