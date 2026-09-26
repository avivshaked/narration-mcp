"""The positive-only lint of voice descriptions (design section 3.5): the word list, the allowed
morphological negatives, the suggestion table, and the warn-only policy."""

from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator

from narration.contracts import interfaces
from narration.contracts.models import LintFinding
from narration.contracts.schemas import TOOLS_BY_NAME
from narration.contracts.serial import to_json
from narration.lint import NegationLinter, lint
from narration.lint.negation import ALLOWED_NEGATIVES, DEFAULT_SUGGESTION, NOTE, SUGGESTIONS

DEFAULT = DEFAULT_SUGGESTION


def _found(description: str) -> list[tuple[str, int, str]]:
    return [(f.phrase, f.offset, f.suggestion) for f in lint(description).findings]


def test_the_linter_implements_the_description_linter_protocol_s3_5() -> None:
    assert isinstance(NegationLinter(), interfaces.DescriptionLinter)


def test_lint_policy_is_warn_and_the_note_explains_why_s3_5() -> None:
    result = lint("A warm voice, not theatrical.")
    assert result.policy == "warn"
    assert result.note == NOTE
    assert lint("A warm voice.").findings == ()


def test_lint_design_example_s7_6() -> None:
    description = "A deep, unhurried voice. " + "x" * 188 + " Not gravelly, not theatrical."
    assert _found(description) == [
        ("Not gravelly", 214, "smooth, clean tone"),
        ("not theatrical", 228, "natural, understated delivery"),
    ]


@pytest.mark.parametrize(("phrases", "suggestion"), SUGGESTIONS, ids=lambda v: v if isinstance(v, str) else "/".join(v))
def test_lint_every_suggestion_table_row_s3_5(phrases: tuple[str, ...], suggestion: str) -> None:
    for phrase in phrases:
        assert _found(f"Warm, {phrase}.") == [(phrase, 6, suggestion)]


@pytest.mark.parametrize("word", ["not", "no", "never", "without", "avoid", "don't", "dont", "nor", "neither", "none"])
def test_lint_every_trigger_word_s3_5(word: str) -> None:
    ((phrase, offset, _),) = _found(f"Calm, {word} breathy.")
    assert (phrase, offset) == (f"{word} breathy", 6)


@pytest.mark.parametrize("word", ["isn't", "aren't", "wasn't", "won't", "can't", "shouldn’t", "doesn’t", "don’t"])
def test_lint_the_other_nt_forms_with_either_apostrophe_s3_5(word: str) -> None:
    assert _found(f"It {word} rush.") == [(f"{word} rush", 3, DEFAULT)]


def test_lint_is_case_insensitive_s3_5() -> None:
    assert _found("NOT gravelly and Never theatrical") == [
        ("NOT gravelly", 0, "smooth, clean tone"),
        ("Never theatrical", 17, "natural, understated delivery"),
    ]


def test_lint_matches_whole_words_only_s3_5() -> None:
    assert _found("Notable warmth, a knowing smile, nimble, nonchalant, annotated, snore, nonetheless.") == []


def test_lint_non_prefix_needs_its_hyphen_s3_5() -> None:
    assert _found("A non-breathy tone.") == [("non-breathy", 2, DEFAULT)]
    assert _found("A nonbreathy tone.") == []


def test_lint_allowed_morphological_negatives_pass_s3_5() -> None:
    assert _found(", ".join(ALLOWED_NEGATIVES) + ", unforced, careless.") == []


def test_lint_what_it_misses_is_not_reported_s3_5() -> None:
    assert _found("Less theatrical, anything but shrill, and free of rasp.") == []


def test_lint_false_alarm_is_reported_and_only_warns_s3_5() -> None:
    result = lint("A no-nonsense tone.")
    assert result.policy == "warn"
    assert [(f.phrase, f.offset) for f in result.findings] == [("no-nonsense", 2)]


def test_lint_phrase_stops_at_punctuation_s3_5() -> None:
    assert _found("Bright, not. Rough.") == [("not", 8, DEFAULT)]
    assert _found("never (ever) loud") == [("never", 0, DEFAULT)]
    assert _found("none") == [("none", 0, DEFAULT)]


def test_lint_longest_table_phrase_wins_s3_5() -> None:
    assert _found("Deep, with no movie-trailer delivery.") == [
        ("no movie-trailer delivery", 11, "even, conversational documentary delivery")
    ]
    assert _found("with no movie-trailer, delivery") == [("no movie-trailer", 5, DEFAULT)]


def test_lint_offsets_are_code_points_into_the_description_as_sent_s3_5() -> None:
    assert _found("Zoë’s 𐌷 voice, not rushed") == [("not rushed", 15, "unhurried, measured pace")]


def test_lint_result_validates_against_design_voice_output_s7_6() -> None:
    result = {"job_id": "job_x", "status": "queued", "design_id": "d", "lint": to_json(lint("not rough"))}
    schema = TOOLS_BY_NAME["design_voice"].output_schema
    assert [e.message for e in Draft202012Validator(schema).iter_errors(result)] == []
    assert lint("not rough").findings == (LintFinding(phrase="not rough", offset=0, suggestion="smooth, clean tone"),)
