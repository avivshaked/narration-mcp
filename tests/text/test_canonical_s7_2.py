"""Canonical form, words, the join and spoken length (design section 7.2; section 9.3's first bullet)."""

from __future__ import annotations

import pytest

from narration.contracts.errors import NarrationError
from narration.contracts.models import CueIn, SegmentIn
from narration.text import TextPipeline, canonical_form, join, spoken_length, words


def _plan(*cues: str, text: str | None = None):
    segment = SegmentIn(segment_id="s", cues=tuple(CueIn(text=c) for c in cues), text=text)
    return TextPipeline().plan_request([segment], [], strict_text=False)[0]


def test_canonical_form_collapses_every_whitespace_run_and_trims_s7_2() -> None:
    sent = "  The  tide  turned 　at dusk.  "
    assert canonical_form(sent) == "The tide turned at dusk."


def test_canonical_form_tab_line_feed_and_carriage_return_are_whitespace_s7_2() -> None:
    # Lead ruling, 2026-09-26: they are collapsed, not refused.
    assert canonical_form("The tide\tturned\r\nat\ndusk.") == "The tide turned at dusk."
    assert _plan("The tide\tturned\r\nat dusk.").spoken_text == "The tide turned at dusk."


def test_canonical_form_applies_nfc_s7_2() -> None:
    assert canonical_form("Café by the quay.") == "Café by the quay."
    assert canonical_form("Ångström") == "Ångström", "ANGSTROM SIGN is a singleton of NFC"


def test_canonical_form_adds_no_punctuation_or_capital_s7_2() -> None:
    assert canonical_form("the ferry left at dawn") == "the ferry left at dawn"
    assert canonical_form("and then — nothing") == "and then — nothing"


def test_canonical_form_is_idempotent_s7_2() -> None:
    for sent in ("  a  b ", "Café", "́ lone mark", "x ́y", ""):
        once = canonical_form(sent)
        assert canonical_form(once) == once


def test_words_set_leading_and_trailing_punctuation_aside_s7_2() -> None:
    got = [w.text for w in words("“Seven,” she said — (forty-eight) of them; it’s done.")]
    assert got == ["Seven", "she", "said", "forty-eight", "of", "them", "it’s", "done"]


def test_words_a_token_of_punctuation_only_is_not_a_word_s7_2() -> None:
    ws = words("then — nothing …")
    assert [w.text for w in ws] == ["then", "nothing"]
    assert [w.index for w in ws] == [0, 1]


def test_words_have_the_same_indices_before_and_after_canonical_form_s7_2() -> None:
    sent = "  Zoë   counted\tforty-two   (cranes). "
    before, after = words(sent), words(canonical_form(sent))
    assert [w.index for w in before] == [w.index for w in after]
    assert [canonical_form(w.text) for w in before] == [w.text for w in after]


def test_join_is_the_cues_joined_by_one_space_s7_2() -> None:
    planned = _plan("The ferry left at", "dawn, and nobody", "saw it go.")
    assert planned.spoken_text == join(["The ferry left at", "dawn, and nobody", "saw it go."])
    assert planned.spoken_text == "The ferry left at dawn, and nobody saw it go."
    assert [c.spoken_span for c in planned.cues] == [(0, 17), (18, 34), (35, 45)]


def test_join_adds_nothing_at_a_cue_boundary_s9_1() -> None:
    planned = _plan("a sentence may run", "across two cues")
    assert planned.spoken_text == "a sentence may run across two cues"
    assert planned.engine_text == planned.spoken_text


def test_join_uses_each_cue_in_canonical_form_s7_2() -> None:
    planned = _plan("  Wind  ", " over the\n fell. ")
    assert [c.received for c in planned.cues] == ["  Wind  ", " over the\n fell. "]
    assert [c.spoken for c in planned.cues] == ["Wind", "over the fell."]
    assert planned.spoken_text == "Wind over the fell."


def test_text_only_segment_is_one_cue_s9_1() -> None:
    planned = _plan(text="  One cue,  only. ")
    assert len(planned.cues) == 1
    assert planned.cues[0].received == "  One cue,  only. "
    assert planned.cues[0].spoken == planned.spoken_text == "One cue, only."


def test_text_equal_to_the_join_is_accepted_r2_s7_2() -> None:
    assert _plan("The ferry left at", "dawn.", text="The ferry left at dawn.").spoken_text == "The ferry left at dawn."


def test_text_whose_canonical_form_is_the_join_is_accepted_r2_s7_2() -> None:
    assert _plan("Café at", "dawn.", text=" Café  at dawn. ").spoken_text == "Café at dawn."


def test_text_that_differs_from_the_join_is_refused_r2_s7_2() -> None:
    with pytest.raises(NarrationError) as caught:
        _plan("The ferry left at", "dawn.", text="The ferry left at dawn")
    error = caught.value
    assert (error.code, error.field, error.retryable) == ("INVALID_ARGUMENT", "segments[0].text", False)
    assert error.details is not None and error.details["first_difference"] == 22


def test_a_cue_of_whitespace_only_is_refused_s7_2() -> None:
    with pytest.raises(NarrationError) as caught:
        _plan("The ferry left.", " \t ")
    assert (caught.value.code, caught.value.field) == ("INVALID_ARGUMENT", "segments[0].cues[1].text")


def test_spoken_length_counts_code_points_not_utf16_units_s7_2() -> None:
    planned = _plan("𐌷𐌰𐌿𐍃 stands by the water.")
    assert spoken_length("𐌷𐌰𐌿𐍃") == 4
    assert planned.spoken_chars == 25
    assert planned.cues[0].warnings == (), "Gothic letters are letters"


def test_spoken_length_is_the_join_in_nfc_code_points_s7_2() -> None:
    planned = _plan("Café", "on the quay.")
    assert planned.spoken_chars == len("Café on the quay.") == 17
