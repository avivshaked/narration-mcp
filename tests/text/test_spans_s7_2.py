"""Exact spans → word ranges (design section 7.2, R14; section 9.3's fourth bullet): code-point offsets in
non-ASCII text, a span that cuts a word refused, and word ranges that survive canonical form."""

from __future__ import annotations

import pytest

from narration.contracts.errors import NarrationError
from narration.contracts.models import CueIn, ExactSpan, ExactWords, SegmentIn
from narration.text import TextPipeline, words


def _exact(text: str, *spans: tuple[int, int]) -> tuple[ExactWords, ...]:
    cue = CueIn(text=text, exact=tuple(ExactSpan(start=s, end=e) for s, e in spans))
    planned = TextPipeline().plan_request([SegmentIn(segment_id="s", cues=(cue,))], [], strict_text=False)[0]
    return planned.cues[0].exact


def _refused(text: str, *spans: tuple[int, int]) -> NarrationError:
    with pytest.raises(NarrationError) as caught:
        _exact(text, *spans)
    assert caught.value.code == "INVALID_ARGUMENT"
    return caught.value


def _words(text: str, exact: ExactWords) -> list[str]:
    first, end = exact.words
    return [w.text for w in words(text)[first:end]]


def test_exact_span_design_example_is_words_3_to_7_s7_2() -> None:
    text = "By sunrise, some three thousand two hundred of them are back in the rock."
    (span,) = _exact(text, (17, 43))
    assert span == ExactWords(start=17, end=43, words=(3, 7))
    assert _words(text, span) == ["three", "thousand", "two", "hundred"]


def test_exact_span_offsets_are_code_points_in_accented_text_s7_2() -> None:
    text = "Zoë and Anaïs counted forty-two cranes."
    (span,) = _exact(text, (22, 31))
    assert (span.words, _words(text, span)) == ((4, 5), ["forty-two"])


def test_exact_span_offsets_are_code_points_beyond_the_bmp_s7_2() -> None:
    (span,) = _exact("𐌷𐌰𐌿𐍃 held nineteen lamps.", (10, 18))
    assert span.words == (2, 3)


def test_exact_span_word_range_survives_canonical_form_s7_2() -> None:
    sent = "  Café   held   nineteen  lamps. "
    (span,) = _exact(sent, (17, 25))
    assert (span.start, span.end, span.words) == (17, 25, (2, 3)), "start and end are echoed as sent"
    planned_words = words("Café held nineteen lamps.")
    assert [w.text for w in planned_words[2:3]] == ["nineteen"]


def test_exact_span_over_several_words_s7_2() -> None:
    (span,) = _exact("They counted three hundred and twelve geese.", (13, 37))
    assert span.words == (2, 6)


def test_exact_span_may_end_before_or_after_trailing_punctuation_s7_2() -> None:
    assert _exact("They counted seven.", (13, 18))[0].words == (2, 3)
    assert _exact("They counted seven.", (13, 19))[0].words == (2, 3)


def test_exact_span_may_start_before_leading_punctuation_s7_2() -> None:
    assert _exact("They saw (forty) geese.", (9, 16))[0].words == (2, 3)
    assert _exact("They saw (forty) geese.", (10, 15))[0].words == (2, 3)


def test_exact_spans_keep_the_order_sent_s7_2() -> None:
    spans = _exact("one two three four", (14, 18), (0, 3))
    assert [s.words for s in spans] == [(3, 4), (0, 1)]


def test_exact_span_that_cuts_a_hyphenated_word_is_refused_s7_2() -> None:
    error = _refused("They counted forty-two cranes.", (13, 18))
    assert error.field == "segments[0].cues[0].exact[0]"
    assert error.details is not None and error.details["word"] == "forty-two"


def test_exact_span_that_cuts_letters_is_refused_s7_2() -> None:
    assert _refused("They counted nineteen cranes.", (17, 21)).field == "segments[0].cues[0].exact[0]"
    assert _refused("They counted nineteen cranes.", (13, 20)).field == "segments[0].cues[0].exact[0]"


def test_exact_span_that_cuts_a_word_in_non_ascii_text_is_refused_s7_2() -> None:
    _refused("Zoë and Anaïs", (8, 11))


def test_exact_spans_that_overlap_are_refused_s7_2() -> None:
    error = _refused("They counted three hundred and twelve geese.", (13, 26), (19, 37))
    assert error.field == "segments[0].cues[0].exact[1]"


def test_exact_span_outside_the_cue_is_refused_s7_2() -> None:
    _refused("They counted seven.", (13, 24))


def test_exact_span_that_is_empty_is_refused_s7_2() -> None:
    _refused("They counted seven.", (13, 13))


def test_exact_span_over_whitespace_or_punctuation_only_is_refused_s7_2() -> None:
    _refused("then — nothing", (4, 7))


def test_exact_span_in_a_later_cue_names_that_cue_s7_2() -> None:
    cues = (CueIn(text="one two"), CueIn(text="three four", exact=(ExactSpan(start=0, end=3),)))
    with pytest.raises(NarrationError) as caught:
        TextPipeline().plan_request([SegmentIn(segment_id="s", cues=cues)], [], strict_text=False)
    assert caught.value.field == "segments[0].cues[1].exact[0]"
