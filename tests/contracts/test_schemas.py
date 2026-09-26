"""The published tool schemas (design sections 5, 7 and 14; plan.md WP01 and WP03's schema checks)."""

from __future__ import annotations

import json
from typing import Any

import jsonschema
import pytest

from narration.contracts import names, schemas

VALIDATOR = jsonschema.Draft202012Validator


def _walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            keys.append(k)
            keys.extend(_walk_keys(v))
    elif isinstance(value, list):
        for v in value:
            keys.extend(_walk_keys(v))
    return keys


def test_schema_tools_are_in_the_fixed_order_of_section_7_1() -> None:
    assert tuple(t.name for t in schemas.TOOLS) == names.TOOL_NAMES


def test_schema_no_ref_anywhere_in_tools_list_s5() -> None:
    published = [
        {"name": t.name, "inputSchema": t.input_schema, "outputSchema": t.output_schema} for t in schemas.TOOLS
    ]
    assert "$ref" not in _walk_keys(published)
    assert '"$ref"' not in json.dumps(published)


@pytest.mark.parametrize("tool", schemas.TOOLS, ids=lambda t: t.name)
def test_schema_every_output_schema_is_an_object_with_an_optional_error_s7_2(tool: schemas.ToolSchema) -> None:
    out = tool.output_schema
    assert out["type"] == "object"
    assert "error" in out["properties"]
    assert "error" not in out.get("required", [])
    assert out["properties"]["error"]["required"] == ["code", "message", "retryable"]


@pytest.mark.parametrize("tool", schemas.TOOLS, ids=lambda t: t.name)
def test_schema_every_schema_is_valid_2020_12(tool: schemas.ToolSchema) -> None:
    VALIDATOR.check_schema(tool.input_schema)
    VALIDATOR.check_schema(tool.output_schema)
    assert tool.input_schema["type"] == "object"
    assert tool.input_schema["additionalProperties"] is False


def _valid(tool: str, args: dict[str, Any]) -> bool:
    return VALIDATOR(schemas.TOOLS_BY_NAME[tool].input_schema).is_valid(args)


VOICE = {"path": "C:\\voices\\narrator.wav", "sha256": "5b1e" + "0" * 60, "transcript": "Good bread asks for patience."}


def test_schema_the_submit_job_example_of_section_7_3_validates() -> None:
    args = {
        "voice": VOICE,
        "hints": [{"term": "Ossavine", "respell": "Oss-a-veen"}],
        "segments": [
            {
                "segment_id": "p03",
                "cues": [
                    {"text": "Before dawn, the reef belongs to the Ossavine shrimp."},
                    {
                        "text": "By sunrise, some three thousand two hundred of them are back in the rock.",
                        "exact": [{"start": 17, "end": 43}],
                    },
                ],
            }
        ],
        "label": "reef film, draft 4",
        "options": {"takes": 3},
    }
    assert _valid("submit_job", args)


def test_schema_instruct_is_an_unknown_field_s3_3() -> None:
    base = {"voice": VOICE, "segments": [{"segment_id": "p1", "text": "Hello there."}]}
    assert _valid("submit_job", base)
    assert not _valid("submit_job", {**base, "instruct": "sound happy"})


def test_schema_text_mode_written_is_refused_s9_2() -> None:
    base = {"voice": VOICE, "segments": [{"segment_id": "p1", "text": "Hello."}]}
    assert _valid("submit_job", {**base, "text_mode": "spoken"})
    assert not _valid("submit_job", {**base, "text_mode": "written"})


def test_schema_a_segment_needs_cues_or_text_s7_2() -> None:
    assert not _valid("submit_job", {"voice": VOICE, "segments": [{"segment_id": "p1"}]})


def test_schema_takes_are_one_to_three_and_retakes_zero_to_three_s7_3() -> None:
    base = {"voice": VOICE, "segments": [{"segment_id": "p1", "text": "Hi."}]}
    assert _valid("submit_job", {**base, "options": {"takes": 3, "max_retakes": 0}})
    assert not _valid("submit_job", {**base, "options": {"takes": 4}})
    assert not _valid("submit_job", {**base, "options": {"max_retakes": 4}})


def test_schema_segment_id_follows_the_id_pattern_s7_2() -> None:
    ok = {"voice": VOICE, "segments": [{"segment_id": "p03.intro-2", "text": "Hi."}]}
    bad = {"voice": VOICE, "segments": [{"segment_id": "P03", "text": "Hi."}]}
    assert _valid("submit_job", ok)
    assert not _valid("submit_job", bad)


def test_schema_get_job_wait_is_at_most_55_seconds_s7_4() -> None:
    assert _valid("get_job", {"job_id": "job_01JBXQ7Z3M8V4T2R9K6N5P0W1C", "wait_s": 55})
    assert not _valid("get_job", {"job_id": "job_01JBXQ7Z3M8V4T2R9K6N5P0W1C", "wait_s": 56})


def test_schema_design_voice_takes_one_to_four_s7_6() -> None:
    assert _valid("design_voice", {"name": "n", "description": "a warm, unhurried voice", "takes": 4})
    assert not _valid("design_voice", {"name": "n", "description": "d", "takes": 5})


def test_schema_audition_takes_at_most_four_variants_s7_6() -> None:
    variants = [{"label": f"v{i}", "respell": f"r{i}"} for i in range(5)]
    assert not _valid("audition_pronunciation", {"voice": VOICE, "term": "t", "variants": variants})
    assert _valid("audition_pronunciation", {"voice": VOICE, "term": "t", "variants": variants[:4]})


def test_schema_error_carries_retry_after_dc2() -> None:
    error = schemas.fragment("Error")
    assert "retry_after_s" in error["properties"]
    VALIDATOR(error).validate({"code": "QUEUE_FULL", "message": "full", "retryable": True, "retry_after_s": 30})


def test_schema_submit_and_get_job_return_poll_after_dc2() -> None:
    for name in ("submit_job", "get_job"):
        assert "poll_after_s" in schemas.TOOLS_BY_NAME[name].output_schema["properties"]
    assert "admission" in schemas.TOOLS_BY_NAME["get_server_status"].output_schema["properties"]


def test_schema_copies_are_independent() -> None:
    a = schemas.tool_schema("submit_job")
    a.input_schema["properties"]["voice"]["properties"]["path"]["type"] = "integer"
    assert schemas.TOOLS_BY_NAME["submit_job"].input_schema["properties"]["voice"]["properties"]["path"]["type"] == (
        "string"
    )


def test_schema_controls_pace_refuses_unknown_fields_s3_3() -> None:
    pace = schemas.fragment("Controls")["properties"]["pace"]
    assert pace["additionalProperties"] is False, "no open object in any input schema (section 3.3)"


def test_schema_job_id_inputs_have_the_id_pattern_s6() -> None:
    for tool in ("get_job", "get_results", "cancel_job"):
        job_id = schemas.TOOLS_BY_NAME[tool].input_schema["properties"]["job_id"]
        assert job_id["pattern"] == names.id_schema_pattern("job_id"), tool


def _open_objects(node: object, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node and node.get("additionalProperties") is not False:
            found.append(path or "/")
        for key, value in node.items():
            found += _open_objects(value, f"{path}/{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found += _open_objects(value, f"{path}/{i}")
    return found


def test_schema_every_input_object_is_closed_s14() -> None:
    for tool in schemas.TOOLS_BY_NAME.values():
        assert _open_objects(tool.input_schema) == [], tool.name
