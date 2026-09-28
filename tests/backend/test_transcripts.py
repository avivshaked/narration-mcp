"""A transcript that is not the one the clip was measured with (design sections 3.2, 10.2 and 14).

The voice's hash covers its transcript character for character, so a transcript sent with a trailing newline, a
double space or curly quotes is another voice, and ``submit_job`` refuses it with ``VOICE_NOT_MEASURED``. When
the same clip is measured under a near spelling of it, the error says where the transcript first differs, and
that this is not a voice to measure again. Neither transcript is quoted in the error.

Every text is invented for these tests.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

from narration import keys
from narration.backend.measures import (
    MEASURE_HINT,
    REWRITES,
    describe_char,
    differs_in,
    first_difference,
    transcript_variants,
)
from narration.config import MeasurementConfig
from narration.contracts import codes, names
from narration.contracts.errors import NarrationError
from narration.contracts.models import TranscriptCheck
from tests.jobs.support import ENGINE_HASH, LAMPS, VOICE_TRANSCRIPT, measurement, voice_hash

from .conftest import Service

QUOTED: str = 'The keeper\'s dog waits by the gate and says "come back soon" with its tail.'
"""A transcript with straight quotes, measured for the test clip in the tests that need one."""


def refused(service: Service, transcript: str) -> NarrationError:
    request = service.request(LAMPS)
    request["voice"] = service.voice(transcript=transcript)
    with pytest.raises(NarrationError) as caught:
        service.backend.submit_job_sync(request)
    error = caught.value
    assert error.code == codes.VOICE_NOT_MEASURED
    return error


def mismatch(error: NarrationError) -> dict[str, Any]:
    assert error.details is not None
    found = error.details["transcript_mismatch"]
    assert isinstance(found, dict)
    return found  # pyright: ignore[reportUnknownVariableType]


def quotes_nothing(error: NarrationError, *transcripts: str) -> None:
    """Neither transcript is echoed: not in the message, the hint or the details."""
    shown = json.dumps([error.message, error.hint, error.details], ensure_ascii=False)
    for transcript in transcripts:
        assert transcript.strip() not in shown


def measure_under(service: Service, transcript: str) -> str:
    """Measure the test clip under ``transcript`` too; its voice hash."""
    clip = service.world.clip_sha256
    hashed = keys.voice_hash(
        model=names.MODEL_QWEN_BASE,
        clip_sha256=clip,
        transcript=transcript,
        language=names.LANGUAGE,
        x_vector_only_mode=False,
    )
    key = keys.measurement_key(
        voice_hash=hashed,
        engine_profile_hash=ENGINE_HASH,
        corpus_version=f"{names.CORPUS}@sha256:{'2' * 64}",
        settings=MeasurementConfig(),
    )
    record = dataclasses.replace(
        measurement(clip, service.world.anchor),
        voice_hash=hashed,
        measurement_key=key,
        transcript_check=TranscriptCheck(heard=transcript, wer=0.0, ok=True),
    )
    service.world.store.put_measurement(record)
    return hashed


# ======================================================================== submit_job's VOICE_NOT_MEASURED


def test_a_trailing_newline_is_named_as_the_difference_not_a_new_voice_s10_2(service: Service) -> None:
    sent = VOICE_TRANSCRIPT + "\n"
    error = refused(service, sent)
    assert error.field == "voice.transcript"
    found = mismatch(error)
    assert found["measured_voice_hash"] == voice_hash(service.world.clip_sha256)
    assert (found["rewrites"], found["differs_in"]) == (["trim_edges"], "whitespace")
    assert found["first_difference"] == len(VOICE_TRANSCRIPT)
    assert (found["sent"], found["measured"]) == ("U+000A LINE FEED", "the end of the text")
    assert (found["sent_chars"], found["measured_chars"]) == (len(sent), len(VOICE_TRANSCRIPT))
    assert error.hint is not None
    assert REWRITES["trim_edges"] in error.hint, "the hint says what to do, not a one-character swap"
    assert f"at character {len(VOICE_TRANSCRIPT)}" in error.hint and "do not measure it again" in error.hint
    assert f"first at character {len(VOICE_TRANSCRIPT)}" in error.message
    assert not error.retryable
    quotes_nothing(error, sent)
    assert service.platform.read_paths == [], "nothing of the clip is read"
    assert service.world.store.queued_jobs() == ()


def test_a_double_space_points_at_the_extra_space_s10_2(service: Service) -> None:
    first_space = VOICE_TRANSCRIPT.index(" ")
    sent = VOICE_TRANSCRIPT.replace(" ", "  ", 1)
    error = refused(service, sent)
    found = mismatch(error)
    assert (found["rewrites"], found["differs_in"]) == (["collapse_whitespace"], "whitespace")
    assert found["first_difference"] == first_space + 1
    assert found["sent"] == "U+0020 SPACE"
    assert error.hint is not None and REWRITES["collapse_whitespace"] in error.hint
    quotes_nothing(error, sent)


def test_a_leading_space_is_named_at_character_0_s10_2(service: Service) -> None:
    sent = " " + VOICE_TRANSCRIPT
    error = refused(service, sent)
    found = mismatch(error)
    assert (found["rewrites"], found["first_difference"]) == (["trim_edges"], 0)
    assert (found["sent"], found["measured"]) == ("U+0020 SPACE", describe_char(VOICE_TRANSCRIPT, 0))
    quotes_nothing(error, sent)


def test_an_ellipsis_for_three_dots_is_named_s10_2(service: Service) -> None:
    dotted = "The tide came in... and the harbour went quiet."
    measured = measure_under(service, dotted)
    sent = dotted.replace("...", "\u2026")
    error = refused(service, sent)
    found = mismatch(error)
    assert found["measured_voice_hash"] == measured
    assert (found["rewrites"], found["differs_in"]) == (["plain_punctuation"], "punctuation")
    assert found["first_difference"] == dotted.index("...")
    assert (found["sent"], found["measured"]) == ("U+2026 HORIZONTAL ELLIPSIS", "U+002E FULL STOP")
    assert (found["sent_chars"], found["measured_chars"]) == (len(sent), len(dotted))
    assert error.hint is not None and REWRITES["plain_punctuation"] in error.hint
    quotes_nothing(error, sent, dotted)


def test_a_newline_the_measured_transcript_ended_with_is_named_s10_2(service: Service) -> None:
    kept = "A heron stood in the shallows until the light was gone.\n"
    measure_under(service, kept)
    sent = kept.rstrip("\n")
    error = refused(service, sent)
    found = mismatch(error)
    assert (found["rewrites"], found["first_difference"]) == (["add_trailing_newline"], len(sent))
    assert (found["sent"], found["measured"]) == ("the end of the text", "U+000A LINE FEED")
    assert error.hint is not None and REWRITES["add_trailing_newline"] in error.hint


def test_curly_quotes_for_straight_ones_are_named_s10_2(service: Service) -> None:
    measured = measure_under(service, QUOTED)
    sent = QUOTED.replace("'", "\u2019").replace('"', "\u201c", 1).replace('"', "\u201d", 1)
    error = refused(service, sent)
    found = mismatch(error)
    assert found["measured_voice_hash"] == measured
    assert (found["rewrites"], found["differs_in"]) == (["plain_punctuation"], "punctuation")
    assert found["first_difference"] == QUOTED.index("'")
    assert (found["sent"], found["measured"]) == ("U+2019 RIGHT SINGLE QUOTATION MARK", "U+0027 APOSTROPHE")
    quotes_nothing(error, sent, QUOTED)


def test_straight_quotes_for_curly_ones_and_an_edge_space_are_named_s10_2(service: Service) -> None:
    curly = QUOTED.replace("'", "\u2019").replace('"', "\u201c", 1).replace('"', "\u201d", 1)
    service.world.store.put_measurement(
        dataclasses.replace(
            measurement(service.world.clip_sha256, service.world.anchor),
            voice_hash=keys.voice_hash(
                model=names.MODEL_QWEN_BASE,
                clip_sha256=service.world.clip_sha256,
                transcript=curly,
                language=names.LANGUAGE,
                x_vector_only_mode=False,
            ),
            measurement_key="sha256:" + "3" * 64,
        )
    )
    error = refused(service, " " + QUOTED)
    found = mismatch(error)
    assert found["rewrites"] == ["trim_edges", "typographic_quotes"]
    assert found["differs_in"] == "whitespace and punctuation"
    assert found["first_difference"] == 0
    assert error.hint is not None
    assert f"{REWRITES['trim_edges']}; then {REWRITES['typographic_quotes']}" in error.hint, "the steps in order"


def test_another_transcript_gets_the_hint_to_measure_or_send_the_measured_one_s14(service: Service) -> None:
    sent = VOICE_TRANSCRIPT.replace("ferry", "barge")
    error = refused(service, sent)
    assert error.field == "voice"
    assert error.hint == MEASURE_HINT
    assert error.details is not None and "transcript_mismatch" not in error.details
    assert error.details["clip_sha256"] == service.world.clip_sha256
    quotes_nothing(error, sent)


# ======================================================================== the spellings tried


def test_the_spellings_tried_are_those_a_slip_makes_fewest_changes_first_s10_2() -> None:
    sent = ' A "quiet" tide\u2019s  edge. \n'
    variants = dict(transcript_variants(sent))
    assert variants['A "quiet" tide\u2019s  edge.'] == ("trim_edges",)
    assert variants['A "quiet" tide\u2019s edge.'] == ("collapse_whitespace",)
    assert variants[sent + "\n"] == ("add_trailing_newline",)
    assert variants[' A "quiet" tide\'s  edge. \n'] == ("plain_punctuation",)
    assert variants[" A \u201cquiet\u201d tide\u2019s  edge. \n"] == ("typographic_quotes",)
    assert variants["A \u201cquiet\u201d tide\u2019s edge."] == ("collapse_whitespace", "typographic_quotes")
    assert sent not in variants and "" not in variants
    assert next(iter(variants)) == 'A "quiet" tide\u2019s  edge.', "the edges trimmed first"
    assert all(len(set(names)) == len(names) and set(names) <= set(REWRITES) for names in variants.values())


def test_a_plain_transcript_has_only_its_trailing_newline_spelling_s10_2() -> None:
    assert transcript_variants("plain words only") == [("plain words only\n", ("add_trailing_newline",))]
    assert transcript_variants("   ") == [("   \n", ("add_trailing_newline",))], "never an empty transcript"


def test_what_the_rewrites_change_is_named_s10_2() -> None:
    assert differs_in(("trim_edges",)) == "whitespace"
    assert differs_in(("plain_punctuation",)) == "punctuation"
    assert differs_in(("collapse_whitespace", "typographic_quotes")) == "whitespace and punctuation"


@pytest.mark.parametrize(
    ("sent", "other", "index"),
    [("abc", "abd", 2), ("abc", "abc\n", 3), (" abc", "abc", 0), ("abc", "abc", 3)],
)
def test_first_difference_is_an_index_into_the_transcript_sent(sent: str, other: str, index: int) -> None:
    assert first_difference(sent, other) == index


def test_a_character_is_described_by_code_point_and_name() -> None:
    assert describe_char("a\tb", 1) == "U+0009 CHARACTER TABULATION"
    assert describe_char("\u2014", 0) == "U+2014 EM DASH"
    assert describe_char("ab", 2) == "the end of the text"
