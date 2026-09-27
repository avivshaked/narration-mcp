"""The transcript check (design section 3.2; ``narration.measure.transcript``). Invented sentences only."""

from __future__ import annotations

from narration.contracts import codes
from narration.measure.transcript import check_transcript, mismatch_error, word_errors
from narration.qa.normaliser import NumberReader

READER = NumberReader()
SAID = "The ferry leaves at seven, and the harbour lights come on one by one along the wall."


def test_the_same_words_match_whatever_their_case_and_punctuation_s3_2() -> None:
    check, errors, words = check_transcript(SAID, SAID.upper().rstrip("."), READER)
    assert (check.ok, check.wer, errors, words) == (True, 0.0, 0, 17)
    assert check.heard == SAID.upper().rstrip(".")


def test_one_word_heard_differently_passes_s3_2() -> None:
    check, errors, _ = check_transcript(SAID, SAID.replace("ferry", "fairy"), READER)
    assert errors == 1
    assert check.ok  # over 6 % in a short clip, but only one word: QA's warn, not its fail


def test_two_words_heard_differently_in_a_short_clip_do_not_s3_2() -> None:
    heard = SAID.replace("ferry", "fairy").replace("lights", "flights")
    check, errors, _ = check_transcript(SAID, heard, READER)
    assert errors == 2
    assert not check.ok


def test_a_transcript_of_another_clip_does_not_match_s3_2() -> None:
    check, errors, words = check_transcript(
        SAID, "Bright gulls wheel over the quay while the boats come home at dusk.", READER
    )
    assert not check.ok
    assert errors > words / 2


def test_a_missing_sentence_does_not_match_s3_2() -> None:
    check, _, _ = check_transcript(SAID + " Nobody waits for the last bus.", SAID, READER)
    assert not check.ok


def test_numbers_are_compared_as_numbers_s11_3() -> None:
    errors, _ = word_errors(
        "The tide rose fourteen per cent in two thousand and twenty.", "The tide rose 14% in 2020.", READER
    )
    assert errors == 0


def test_a_transcript_with_no_words_never_matches_s3_2() -> None:
    check, errors, words = check_transcript("...", "", READER)
    assert (errors, words) == (0, 0)
    assert not check.ok


def test_the_mismatch_error_says_what_was_heard_and_what_to_do_s5() -> None:
    check, errors, words = check_transcript(SAID, "Something else entirely.", READER)
    error = mismatch_error(check, errors, words)
    assert error.code == codes.REF_TEXT_MISMATCH
    assert error.field == "voice.transcript"
    assert error.hint and not error.retryable
    assert error.details == {
        "heard": "Something else entirely.",
        "wer": check.wer,
        "word_errors": errors,
        "words": words,
    }
