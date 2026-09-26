"""The four text checks (design section 9.1 step 3.2; section 9.3's third bullet): every warning kind, the
exclusions, lone capitals as info, and the punctuation that must pass. Warnings never change the text."""

from __future__ import annotations

from typing import Any

import pytest

from narration.contracts.models import CueIn, SegmentIn
from narration.text import TextPipeline
from narration.text.rules import UNIT_EXCLUSIONS, UNIT_SYMBOLS


def _findings(text: str) -> list[dict[str, Any]]:
    planned = TextPipeline().plan_segment(SegmentIn(segment_id="s", cues=(CueIn(text=text),)), [])
    (cue,) = planned.cues
    assert cue.spoken == cue.engine, "a warning never changes the text"
    out: list[dict[str, Any]] = []
    for warning in cue.warnings:
        assert warning.code == "WRITTEN_FORM_TOKEN"
        assert (warning.segment_id, warning.cue) == ("s", 0)
        assert warning.details is not None
        out.append({**warning.details, "severity": warning.severity})
    return out


def _tokens(text: str) -> list[tuple[str, str, int]]:
    return [(f["token"], f["kind"], f["offset"]) for f in _findings(text)]


def test_warn_digit_s9_1() -> None:
    assert _findings("The tower is 40 metres tall.") == [
        {"token": "40", "kind": "digit", "offset": 13, "severity": "warn"}
    ]


@pytest.mark.parametrize("digit", ["٣", "²", "¹", "३", "７"], ids=lambda d: f"U+{ord(d):04X}")
def test_warn_digit_any_unicode_digit_once_s9_1(digit: str) -> None:
    assert _tokens(f"Mark {digit} here.") == [(digit, "digit", 5)], "a digit is reported as digit only"


def test_warn_digit_token_is_the_run_of_digits_s9_1() -> None:
    assert _tokens("In 2026, 3,200 geese.") == [("2026", "digit", 3), ("3", "digit", 9), ("200", "digit", 11)]


@pytest.mark.parametrize("symbol", ["&", "%", "/", "×", "°", "·", "#", "£", "€", "$", "@", "*", "+", "="])
def test_warn_symbol_s9_1(symbol: str) -> None:
    assert _tokens(f"one {symbol} two") == [(symbol, "symbol", 4)]


def test_warn_symbol_inside_a_word_is_the_symbol_itself_s9_1() -> None:
    assert _tokens("The north/south road.") == [("/", "symbol", 9)]


def test_warn_symbol_run_is_one_finding_s9_1() -> None:
    assert _tokens("Wait --> there") == [(">", "symbol", 7)], "- is prosody; > is the symbol"
    assert _tokens("Salt && pepper") == [("&&", "symbol", 5)]


def test_emoji_sequence_is_one_symbol_finding_s9_1() -> None:
    family = "\U0001f469‍\U0001f469‍\U0001f467"
    assert _tokens(f"A {family} came.") == [(family, "symbol", 2)]


def test_warn_unit_like_s9_1() -> None:
    got = _tokens("ten km , four kg , sixty mph , nine ft , fifty Hz , two kW , five µs , six μs , one kΩ")
    assert [(t, k) for t, k, _ in got] == [
        ("km", "unit_like"),
        ("kg", "unit_like"),
        ("mph", "unit_like"),
        ("ft", "unit_like"),
        ("Hz", "unit_like"),
        ("kW", "unit_like"),
        ("µs", "unit_like"),
        ("μs", "unit_like"),
        ("kΩ", "unit_like"),
    ]


def test_warn_unit_like_with_punctuation_set_aside_s9_1() -> None:
    assert _tokens("It weighs four kg.") == [("kg", "unit_like", 15)]
    assert _tokens("four (kg), then") == [("kg", "unit_like", 6)]


def test_unit_like_is_matched_case_sensitively_s9_1() -> None:
    assert _tokens("KM and KG and MPH") == []


def test_unit_like_needs_a_standalone_token_s9_1() -> None:
    assert _tokens("kilograms and kgs") == []


def test_warn_unit_like_lone_lower_case_m_s_g_s9_1() -> None:
    assert _tokens("two m , five s , ten g") == [("m", "unit_like", 4), ("s", "unit_like", 13), ("g", "unit_like", 21)]


def test_no_warning_for_a_A_I_in_am_as_at_s9_1() -> None:
    assert _findings("A cat and I am in a boat, as at dawn.") == []


def test_unit_exclusions_are_ordinary_words_s9_1() -> None:
    for word in ("in", "am", "as", "at", "Ah", "Pa", "Ms", "TV", "PC"):
        assert word in UNIT_EXCLUSIONS and word not in UNIT_SYMBOLS
    assert _findings("Ah, Pa said the TV was on at noon, as Ms Harl knew.") == []


def test_lone_capitals_are_letter_info_s9_1() -> None:
    assert _findings("Plan B failed, so we tried C and then J.") == [
        {"token": "B", "kind": "letter", "offset": 5, "severity": "info"},
        {"token": "C", "kind": "letter", "offset": 27, "severity": "info"},
        {"token": "J", "kind": "letter", "offset": 38, "severity": "info"},
    ]


def test_lone_lower_case_letters_are_letter_info_s9_1() -> None:
    assert _tokens("Mark it with an x and an e.") == [("x", "letter", 16), ("e", "letter", 25)]


def test_lone_non_ascii_letter_is_letter_info_s9_1() -> None:
    assert _tokens("Then é, then Ω.") == [("é", "letter", 5), ("Ω", "letter", 13)]


def test_letters_with_combining_marks_are_letters_not_symbols_s9_1() -> None:
    # q + COMBINING DOT ABOVE has no precomposed form; it stays two code points and is one letter.
    assert _tokens("The q̇ sound in Nguyễn and हिन्दी.") == [("q̇", "letter", 4)]


def test_a_combining_mark_after_a_space_is_a_symbol_s9_1() -> None:
    assert _tokens("a ́ b") == [("́", "symbol", 2), ("b", "letter", 4)]


def test_passing_punctuation_raises_no_warning_s9_1() -> None:
    text = 'He said, "Wait — the tide… it\'s turning; isn’t it?" ‘Yes’: the east–west road (quiet) is well-worn! “Go.”'
    assert _findings(text) == []


def test_every_finding_kind_in_one_cue_in_offset_order_s9_1() -> None:
    assert _tokens("At 5 & C, walk 2 km.") == [
        ("5", "digit", 3),
        ("&", "symbol", 5),
        ("C", "letter", 7),
        ("2", "digit", 15),
        ("km", "unit_like", 17),
    ]


def test_findings_offsets_are_in_the_spoken_text_after_canonical_form_s9_1() -> None:
    assert _tokens("  The   tower\tis 40 m.") == [("40", "digit", 13), ("m", "unit_like", 16)]
