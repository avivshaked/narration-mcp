"""The aligner's alphabet (design section 11.2 step 1): letters folded to A–Z with accents dropped, the
apostrophe kept, ``|`` for an in-word hyphen, and a word with a digit or symbol left out."""

from __future__ import annotations

import pytest

from narration.align import ALPHABET, fold_letter, spell

PINNED_VOCABULARY = ("<pad>", "<s>", "</s>", "<unk>", "|", *"ETAONIHSRDLUMWCFGYPBVK'XJQZ")
"""``vocab.json`` of facebook/wav2vec2-large-960h-lv60-self at the pinned revision (spike (b))."""


def _s(word: str) -> str | None:
    tokens = spell(word)
    return None if tokens is None else "".join(tokens)


def test_alphabet_is_the_pinned_models_letters_s11_2() -> None:
    assert frozenset(PINNED_VOCABULARY) - {"<pad>", "<s>", "</s>", "<unk>"} == ALPHABET


@pytest.mark.parametrize(
    ("word", "spelled"),
    [
        ("ferry", "FERRY"),
        ("Café", "CAFE"),
        ("naïve", "NAIVE"),
        ("Zoë", "ZOE"),
        ("Straße", "STRASSE"),
        ("Æsir", "AESIR"),
        ("Øresund", "ORESUND"),
        ("Łódź", "LODZ"),
        ("ﬁne", "FINE"),  # a ligature decomposes
        ("Ｑuenby", "QUENBY"),  # a full-width letter decomposes
    ],
)
def test_letters_folded_to_a_z_accents_dropped_s11_2(word: str, spelled: str) -> None:
    assert _s(word) == spelled


def test_combining_accent_is_dropped_s11_2() -> None:
    assert _s("Tolvénny") == "TOLVENNY"


@pytest.mark.parametrize("word", ["don't", "don’t", "donʼt"])
def test_apostrophe_kept_s11_2(word: str) -> None:
    assert _s(word) == "DON'T"


@pytest.mark.parametrize(
    ("word", "spelled"),
    [
        ("forty-eight", "FORTY|EIGHT"),
        ("well–known", "WELL|KNOWN"),
        ("stop—then", "STOP|THEN"),
        ("rock--solid", "ROCK|SOLID"),
        ("rock-'n'-roll", "ROCK|'N'|ROLL"),
    ],
)
def test_in_word_hyphen_becomes_separator_s11_2(word: str, spelled: str) -> None:
    assert _s(word) == spelled


def test_silent_punctuation_inside_a_word_is_dropped_s11_2() -> None:
    assert _s("U.S") == "US"
    assert _s("e.g") == "EG"


@pytest.mark.parametrize("word", ["12", "3,050", "38%", "and/or", "B2B", "3rd", "π", "東京", "5.5", "x²", "#1"])
def test_word_with_a_digit_or_symbol_is_left_out_s11_2(word: str) -> None:
    assert spell(word) is None


def test_word_with_no_letter_is_left_out_s11_2() -> None:
    assert spell("") is None
    assert spell("--") is None


def test_fold_letter_has_no_spelling_for_another_script_s11_2() -> None:
    assert fold_letter("é") == "E"
    assert fold_letter("ß") == "SS"
    assert fold_letter("Ж") is None
