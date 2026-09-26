"""The planner as a whole (design section 9.1): strict_text, duplicate ids, the rules hash, the over-long
segment warning (section 3.2), and its output against the published schemas (sections 7.2, 7.6)."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import pytest
import rfc8785
from jsonschema import Draft202012Validator

from narration.config import TextConfig
from narration.contracts import interfaces
from narration.contracts.errors import ConfigError, NarrationError
from narration.contracts.models import CueIn, Hint, SegmentIn, SegmentText
from narration.contracts.names import TEXT_CHECKS_VERSION
from narration.contracts.schemas import TOOLS_BY_NAME, fragment, record_schema
from narration.contracts.serial import to_json
from narration.text import TextPipeline, rules_sha256, segment_too_long
from narration.text.rules import DEFAULT_REFUSE, rules_document

RULES_SHA256_TEXT_1_1_0 = "f6b3749c2284c8968aa1e469a18998fcdc730d3ad2dcfb043b17b1a739fb1850"
"""The rules hash of text-1.1.0 with the default markup list. If a rule changes, this changes: decide whether
the change needs a new TEXT_CHECKS_VERSION (a contract change request), then update this value."""


def _segment(sid: str, *cues: str) -> SegmentIn:
    return SegmentIn(segment_id=sid, cues=tuple(CueIn(text=c) for c in cues))


def _errors(schema: dict[str, Any], value: Any) -> list[str]:
    return [e.message for e in Draft202012Validator(schema).iter_errors(value)]


def test_the_pipeline_implements_the_text_planner_protocol_s9_1() -> None:
    assert isinstance(TextPipeline(), interfaces.TextPlanner)


def test_checks_info_names_the_version_and_the_rules_hash_s9() -> None:
    info = TextPipeline().checks_info
    assert info.version == TEXT_CHECKS_VERSION == "text-1.1.0"
    assert info.rules_sha256 == hashlib.sha256(rfc8785.dumps(rules_document(DEFAULT_REFUSE))).hexdigest()
    assert info.rules_sha256 == RULES_SHA256_TEXT_1_1_0


def test_rules_hash_follows_the_configured_markup_s9() -> None:
    assert rules_sha256(("[", "]")) != rules_sha256(DEFAULT_REFUSE)
    assert TextPipeline(TextConfig(refuse=("[", "]"))).checks_info.rules_sha256 == rules_sha256(("[", "]"))


def test_every_segment_carries_the_checks_info_s9() -> None:
    pipeline = TextPipeline()
    (planned,) = pipeline.plan_request([_segment("s", "ok")], [], strict_text=False)
    assert planned.text_checks == pipeline.checks_info


def test_config_naming_another_checks_version_is_refused_s16() -> None:
    with pytest.raises(ConfigError, match=re.escape(TEXT_CHECKS_VERSION)):
        TextPipeline(TextConfig(checks="text-9.0.0"))


def test_config_with_an_empty_markup_string_is_refused_s16() -> None:
    with pytest.raises(ConfigError):
        TextPipeline(TextConfig(refuse=("[", "")))


def test_duplicate_segment_ids_are_refused_s7_3() -> None:
    with pytest.raises(NarrationError) as caught:
        TextPipeline().plan_request(
            [_segment("p1", "a"), _segment("p2", "b"), _segment("p1", "c")], [], strict_text=False
        )
    assert (caught.value.code, caught.value.field) == ("INVALID_ARGUMENT", "segments[2].segment_id")


def test_a_segment_with_neither_cues_nor_text_is_refused_s7_2() -> None:
    with pytest.raises(NarrationError) as caught:
        TextPipeline().plan_request([SegmentIn(segment_id="s")], [], strict_text=False)
    assert (caught.value.code, caught.value.field) == ("INVALID_ARGUMENT", "segments[0].cues")


def test_segments_come_back_in_request_order_s9_1() -> None:
    planned = TextPipeline().plan_request([_segment("b", "two"), _segment("a", "one")], [], strict_text=False)
    assert [p.segment_id for p in planned] == ["b", "a"]


def test_without_strict_text_warnings_do_not_refuse_s9_1() -> None:
    (planned,) = TextPipeline().plan_request([_segment("s", "The tower is 40 m tall.")], [], strict_text=False)
    assert [w.severity for w in planned.cues[0].warnings] == ["warn", "warn"]


def test_strict_text_refuses_warnings_listing_every_offender_s9_1() -> None:
    segments = [_segment("one", "The tower is 40 metres tall."), SegmentIn(segment_id="two", text="Salt & C.")]
    with pytest.raises(NarrationError) as caught:
        TextPipeline().plan_request(segments, [], strict_text=True)
    error = caught.value
    assert (error.code, error.field, error.retryable) == ("TEXT_REFUSED", "segments[0].cues[0].text", False)
    assert error.details == {
        "offenders": [
            {
                "field": "segments[0].cues[0].text",
                "segment_id": "one",
                "cue": 0,
                "code": "WRITTEN_FORM_TOKEN",
                "severity": "warn",
                "token": "40",
                "kind": "digit",
                "offset": 13,
            },
            {
                "field": "segments[1].text",
                "segment_id": "two",
                "cue": 0,
                "code": "WRITTEN_FORM_TOKEN",
                "severity": "warn",
                "token": "&",
                "kind": "symbol",
                "offset": 5,
            },
        ]
    }, "the lone letter C is info and is not an offender"


def test_strict_text_does_not_refuse_letter_info_s9_1() -> None:
    (planned,) = TextPipeline().plan_request([_segment("s", "Plan B failed.")], [], strict_text=True)
    assert [w.severity for w in planned.cues[0].warnings] == ["info"]


def test_strict_text_refuses_a_term_split_across_cues_s9_1() -> None:
    hints = [Hint(term="Velmora Pass", respell="Vell-mora Pahss")]
    with pytest.raises(NarrationError) as caught:
        TextPipeline().plan_request([_segment("s", "towards Velmora", "Pass we went.")], hints, strict_text=True)
    assert caught.value.details is not None
    (offender,) = caught.value.details["offenders"]
    assert offender["code"] == "TERM_SPLIT_ACROSS_CUES" and offender["term"] == "Velmora Pass"


def test_text_refused_error_validates_against_the_error_fragment_s7_2() -> None:
    with pytest.raises(NarrationError) as caught:
        TextPipeline().plan_request([_segment("s", "[x]")], [], strict_text=False)
    assert _errors(fragment("Error"), to_json(caught.value.error)) == []


def test_planned_segment_validates_against_its_record_schema_s7_5() -> None:
    hints = [Hint(term="Ossavine", respell="Oss-a-veen"), Hint(term="Velmora Pass")]
    (planned,) = TextPipeline().plan_request(
        [_segment("s", "The Ossavine at 5 km & B.", "Then Velmora", "Pass.")], hints, strict_text=False
    )
    assert planned.warnings and planned.cues[0].warnings, "the sample exercises every kind of flag"
    assert _errors(record_schema(SegmentText), to_json(planned)) == []


def test_planned_segment_fits_the_check_text_output_s7_6() -> None:
    (planned,) = TextPipeline().plan_request([_segment("s", "The Ossavine at 5 km.")], [], strict_text=False)
    segment = to_json(planned)
    segment.update({"max_segment_chars": None, "over_by_chars": None, "est_duration_s": None})
    result = {"segments": [segment], "text_checks_version": planned.text_checks.version if planned.text_checks else ""}
    assert _errors(TOOLS_BY_NAME["check_text"].output_schema, result) == []


# ---------------------------------------------------------------- SEGMENT_TOO_LONG (sections 3.2, 7.3)


def test_segment_too_long_is_a_warning_with_the_limits_s3_2() -> None:
    pipeline = TextPipeline()
    (planned,) = pipeline.plan_request([_segment("p11", "a" * 300, "b" * 311)], [], strict_text=True)
    flag = segment_too_long(planned, max_segment_chars=450, max_segment_seconds=31.5)
    assert flag is not None
    assert (flag.code, flag.severity, flag.segment_id) == ("SEGMENT_TOO_LONG", "warn", "p11")
    assert flag.message == "p11 is 612 spoken characters; this voice read up to 450 reliably. It will be rendered."
    assert flag.details == {
        "max_segment_chars": 450,
        "max_segment_seconds": 31.5,
        "spoken_chars": 612,
        "over_by_chars": 162,
        "cue_chars": [300, 311],
    }
    assert _errors(fragment("Flag"), to_json(flag)) == []


def test_segment_at_or_under_the_limit_is_not_flagged_s3_2() -> None:
    (planned,) = TextPipeline().plan_request([_segment("p1", "x" * 450)], [], strict_text=False)
    assert segment_too_long(planned, max_segment_chars=450) is None
    assert segment_too_long(planned, max_segment_chars=449) is not None


def test_an_over_long_segment_is_never_refused_s3_2() -> None:
    # The planner has no length limit of its own: 1200 spoken characters plan like any other segment.
    (planned,) = TextPipeline().plan_request([_segment("p1", *(["y" * 99] * 12))], [], strict_text=True)
    assert planned.spoken_chars == 12 * 99 + 11
