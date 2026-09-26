"""The number reader of design section 11.3: Whisper's English normaliser, vendored, plus the "nought" rule."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from narration.contracts.names import NUMBER_READER
from narration.qa.normaliser import (
    NumberReader,
    fold_apostrophes,
    letters_and_digits,
    letters_only,
    phrases,
    remove_apostrophes,
)

REPO = Path(__file__).resolve().parents[2]
VENDORED = REPO / "src" / "narration" / "qa" / "normaliser" / "_whisper"
UPSTREAM_COMMIT = "31243bad24cc746f07d4c8bfdd2d974872cb1803"
UPSTREAM_MAP_SHA256 = "6607f948be9824d2e1b2fa2223cd94c06c45afa4e05ea0e3d5e1f2bdffde2465"

READER = NumberReader()

# Section 11.3 step 1, as the design lists them (KNOW there: eval/normaliser_check.py, transformers 5.17.0).
NUMBER_FORMS = [
    ("fourteen per cent", "14%"),
    ("fourteen percent", "14%"),
    ("14%", "14%"),
    ("two oh one", "201"),
    ("two o one", "201"),
    ("two zero one", "201"),
    ("four thousand and forty-nine", "4049"),
    ("four thousand forty-nine", "4049"),
    ("thirty-five fifty-nine", "3559"),
    ("3,200", "3200"),
    # "What this confirms": the number, not its wording.
    ("thirty-two hundred", "3200"),
    ("three thousand two hundred", "3200"),
]


@pytest.mark.parametrize(("text", "expected"), NUMBER_FORMS)
def test_number_forms_read_alike_s11_3(text: str, expected: str) -> None:
    assert READER.read(text) == expected


def test_nought_reads_as_zero_s11_3() -> None:
    # Whisper's normaliser alone misses "nought" (the design's one recorded miss); the extra rule fixes it.
    assert READER.whisper("nought point five four") == "nought .54"
    assert READER.read("nought point five four") == "0.54"
    assert READER.read("Nought point five four") == READER.read("zero point five four") == "0.54"


def test_nought_rule_is_whole_word_only_s11_3() -> None:
    assert READER.read("noughts and crosses") == "noughts and crosses"


@pytest.mark.parametrize(
    ("british", "american"),
    [
        ("the colour of the centre", "the color of the center"),
        ("grey metres and kilometres", "gray meters and kilometers"),
        ("they analyse a favourite neighbour", "they analyze a favorite neighbor"),
        ("it travelled", "it traveled"),
    ],
)
def test_spelling_variants_read_alike_s11_3(british: str, american: str) -> None:
    # plan.md section 1.3 item 2: the normaliser treats spelling variants alike only with its spelling map.
    assert READER.read(british) == READER.read(american) == american


def test_number_reader_version_is_the_contracts_s10_2() -> None:
    assert READER.version == NUMBER_READER


def test_words_splits_the_reading() -> None:
    assert READER.words("Room three hundred and seven, still empty.") == ("room", "307", "still", "empty")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("In the year two thousand, forty ships sailed.", "in the year 2000 40 ships sailed"),
        ("By nineteen ninety, five hundred ships had gone.", "by 1990 500 ships had gone"),
        ("She counted one, two, three and stopped.", "she counted one 2 3 and stopped"),
        ("It rose two hundred; forty fell.", "it rose 200 40 fell"),
        ("Two thousand — forty ships.", "2000 40 ships"),
    ],
)
def test_number_words_never_merge_across_punctuation_s11_3(text: str, expected: str) -> None:
    # Reader @2 (plan.md DC-7): Whisper's normaliser alone deletes the comma and reads one number.
    assert READER.read(text) == expected
    assert READER.whisper("In the year two thousand, forty ships sailed.") == "in the year 2040 ships sailed"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("3,200 and 3.5", "3200 and 3.5"),  # a comma or period between digits
        ("forty-eight well-known", "48 well known"),  # a hyphen inside a word
        ("fourteen per cent, then 14%", "14% then 14%"),  # the percent sign stays with its number
        ("the guide's boat", "the guide is boat"),  # the apostrophe stays inside its word
    ],
)
def test_punctuation_inside_a_number_or_word_does_not_split_s11_3(text: str, expected: str) -> None:
    assert READER.read(text) == expected


def test_phrases_split_at_punctuation_only_s11_3() -> None:
    assert phrases("One, two; three (four) — five!") == ("One", " two", " three ", "four", " five")
    assert phrases("forty-eight, 3.5") == ("forty-eight", " 3.5")
    assert phrases("") == ()


@pytest.mark.parametrize(
    ("spoken", "written"),
    [
        ("The ferry left at four thirty sharp.", "The ferry left at 4:30 sharp."),
        ("ten thirty", "10:30"),
        ("twelve oh five", "12:05"),
        ("three pounds fifty", "£3.50"),
        ("three dollars and fifty cents", "$3.50"),
        ("three dollars fifty", "$3.50"),
        ("minus five", "-5"),
        ("five degrees", "5°"),
        ("five degrees celsius", "5°C"),
        ("five degrees celsius", "5℃"),
        ("minus four degrees fahrenheit", "-4°F"),
        ("five degrees Celsius", "5°Celsius"),
        ("It fell to minus five (minus five) degrees.", "It fell to -5 (-5) degrees."),
    ],
)
def test_times_money_and_minus_read_as_they_are_spoken_s11_3(spoken: str, written: str) -> None:
    assert READER.read(spoken) == READER.read(written)


def test_a_hyphen_inside_a_word_or_range_is_not_minus_s11_3() -> None:
    assert READER.read("from 1990-1995") == "from 1990 1995"
    assert READER.read("COVID-19") == "covid 19"


def test_a_comma_inside_one_spoken_number_splits_it_s11_3() -> None:
    # A known trade-off of reader @2 (plan.md DC-7): read in phrases, "three thousand, two hundred" is two
    # numbers, while a transcript's "3,200" is one. Pinned so a change to it is deliberate.
    assert READER.read("three thousand, two hundred") == "3000 200"
    assert READER.read("3,200") == "3200"


def test_curly_apostrophes_read_as_straight_s11_3() -> None:
    assert fold_apostrophes("Brannoc’s ‘boat’ ʼtis") == "Brannoc's 'boat' 'tis"
    assert READER.read("Brannoc’s boat") == READER.read("Brannoc's boat") == READER.read("Brannocʼs boat")


def test_words_in_parentheses_are_read_and_annotations_are_not_s11_3() -> None:
    # The text is spoken as sent, parentheses included; Whisper's normaliser alone would drop "the keeper".
    assert READER.read("He (the keeper) waited.") == "he the keeper waited"
    assert READER.whisper("He (the keeper) waited.") == "he waited"
    # Bracketed annotations only a transcript can hold are removed, as Whisper's normaliser removes them.
    assert READER.read("He waited [music] <noise>.") == "he waited"


def test_letters_and_digits_keep_numbers_inside_terms_s11_1() -> None:
    assert letters_and_digits("7 Sisters, Café!") == "7sisterscafe"


def test_remove_apostrophes_joins_possessives_s11_1() -> None:
    assert remove_apostrophes("the guide's and the guide’s") == "the guides and the guides"


def test_letters_only_drops_everything_but_letters_s11_1() -> None:
    assert letters_only("Tavvy-Mokrell, 2!") == "tavvymokrell"
    assert letters_only("Café") == "cafe"


def test_vendored_spelling_map_is_upstreams() -> None:
    data = (VENDORED / "english.json").read_bytes()
    assert hashlib.sha256(data).hexdigest() == UPSTREAM_MAP_SHA256


def test_vendored_sources_name_their_origin_and_licence() -> None:
    licence = (VENDORED / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in licence
    assert "Copyright (c) 2022 OpenAI" in licence
    for name in ("__init__.py", "basic.py", "english.py"):
        head = (VENDORED / name).read_text(encoding="utf-8")[:1200]
        assert UPSTREAM_COMMIT in head, name
    notices = (REPO / "THIRD_PARTY_NOTICES").read_text(encoding="utf-8")
    assert UPSTREAM_COMMIT in notices


@pytest.mark.evidence
def test_bakeoff_normaliser_check_reproduced_s11_3(bakeoff_root: Path) -> None:
    """Every case of the bake-off's eval/normaliser_check.txt reads the same through the vendored copy.

    The bake-off built Whisper's normaliser with an empty spelling map; none of its cases holds a mapped word,
    so the vendored normaliser (with its map) must give the same output. With the "nought" rule, the one miss
    becomes the reading of "zero point five four".
    """
    lines = (bakeoff_root / "eval" / "normaliser_check.txt").read_text(encoding="utf-8").splitlines()
    cases = [tuple(ast.literal_eval(part.strip()) for part in line.split(" -> ")) for line in lines[1:] if line]
    assert len(cases) == 21
    for text, expected in cases:
        assert READER.whisper(text) == expected, text
        if text.startswith("nought"):
            assert READER.read(text) == READER.read("zero point five four") == "0.54"
        else:
            assert READER.read(text) == expected, text
