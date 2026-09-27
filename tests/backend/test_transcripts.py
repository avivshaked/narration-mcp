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
    describe_char,
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


def measure_quoted(service: Service) -> str:
    """Measure the test clip under ``QUOTED`` too; its voice hash."""
    clip = service.world.clip_sha256
    hashed = keys.voice_hash(
        model=names.MODEL_QWEN_BASE,
        clip_sha256=clip,
        transcript=QUOTED,
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
        transcript_check=TranscriptCheck(heard=QUOTED, wer=0.0, ok=True),
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
    assert found["differs_in"] == "whitespace"
    assert found["first_difference"] == len(VOICE_TRANSCRIPT)
    assert (found["sent"], found["measured"]) == ("U+000A LINE FEED", "the end of the text")
    assert (found["sent_chars"], found["measured_chars"]) == (len(sent), len(VOICE_TRANSCRIPT))
    assert error.hint is not None
    assert f"at character {len(VOICE_TRANSCRIPT)}" in error.hint and "do not measure it again" in error.hint
    assert f"first at character {len(VOICE_TRANSCRIPT)}" in error.message
    assert not error.retryable
    quotes_nothing(error, sent)
    assert service.platform.read_paths == [], "nothing of the clip is read"
    assert service.world.store.queued_jobs() == ()


def test_a_double_space_points_at_the_extra_space_s10_2(service: Service) -> None:
    first_space = VOICE_TRANSCRIPT.index(" ")
    sent = VOICE_TRANSCRIPT.replace(" ", "  ", 1)
    found = mismatch(refused(service, sent))
    assert found["differs_in"] == "whitespace"
    assert found["first_difference"] == first_space + 1
    assert found["sent"] == "U+0020 SPACE"


def test_curly_quotes_for_straight_ones_are_named_s10_2(service: Service) -> None:
    measured = measure_quoted(service)
    sent = QUOTED.replace("'", "\u2019").replace('"', "\u201c", 1).replace('"', "\u201d", 1)
    error = refused(service, sent)
    found = mismatch(error)
    assert found["measured_voice_hash"] == measured
    assert found["differs_in"] == "punctuation"
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
    found = mismatch(refused(service, " " + QUOTED))
    assert found["differs_in"] == "whitespace and punctuation"
    assert found["first_difference"] == 0


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
    assert variants['A "quiet" tide\u2019s  edge.'] == "whitespace"
    assert variants['A "quiet" tide\u2019s edge.'] == "whitespace"
    assert variants[' A "quiet" tide\'s  edge. \n'] == "punctuation"
    assert variants[" A \u201cquiet\u201d tide\u2019s  edge. \n"] == "punctuation"
    assert variants["A \u201cquiet\u201d tide\u2019s edge."] == "whitespace and punctuation"
    assert sent not in variants and "" not in variants
    assert next(iter(variants)) == 'A "quiet" tide\u2019s  edge.', "the edges trimmed first"


def test_a_plain_transcript_has_only_its_whitespace_and_quote_spellings_s10_2() -> None:
    assert transcript_variants("plain words only") == []
    assert transcript_variants("   ") == [], "never an empty transcript"


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
