"""The MCP front-end against a fake backend, over real JSON-RPC in both protocol eras (sections 5, 7, 14).

Acceptance (plan.md WP17): sending ``instruct`` returns ``INVALID_ARGUMENT`` with ``field: "instruct"`` and a
hint to section 3.3; ``text_mode: "written"`` is refused; no ``$ref`` in ``tools/list``; an unknown tool is
JSON-RPC -32602; a failure inside a call is an ``INTERNAL`` tool error (ADR 0001).
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import anyio
import pytest
from jsonschema import Draft202012Validator
from mcp.shared.uri_template import UriTemplate

from narration.config import RetentionConfig
from narration.contracts import codes
from narration.contracts.errors import NarrationError, UnsupportedPlatform
from narration.contracts.names import (
    ID_PATTERNS,
    PROMPTS,
    RESOURCE_CACHE_SCOPE,
    RESOURCES,
    SERVER_NAME,
    TOOL_NAMES,
)
from narration.contracts.schemas import DIALECT, TOOLS_BY_NAME
from narration.mcp import FrontEnd, build_front_end, results
from narration.mcp.descriptions import (
    BACKOFF_RULE,
    LENGTH_IS_YOURS,
    NO_CALLER_STATE,
    RESPELLING_IS_A_HINT,
    retention_clause,
)
from tests.mcp.fake_backend import DESIGN_ID, JOB_ID, TAKE_ID, VALID_ARGUMENTS, VOICE, VOICE_HASH, FakeBackend
from tests.mcp.wire import ERAS, Era, Wire, open_wire

INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
OTHER_JOB_ID = "job_01JBXQ7Z3M8V4T2R9K6N5P0W1D"
"""A well-formed job id the fake does not know."""


@pytest.fixture
def backend(tmp_path: Path) -> FakeBackend:
    return FakeBackend(tmp_path / "store")


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "narration-mcp.log"


def over_wire(front: FrontEnd, era: Era, body: Callable[[Wire], Awaitable[None]]) -> None:
    """Run ``body`` against ``front`` over a fresh connection in ``era``."""

    async def main() -> None:
        async with open_wire(front.server, era) as wire:
            await body(wire)

    anyio.run(main)


def tool_result(reply: dict[str, Any]) -> dict[str, Any]:
    """The ``tools/call`` result of a reply that must not be a JSON-RPC error."""
    assert "result" in reply, reply
    return reply["result"]


def error_of(reply: dict[str, Any]) -> dict[str, Any]:
    """The structured Error of a tool error, after checking its shape (section 14)."""
    result = tool_result(reply)
    assert result["isError"] is True, result
    assert set(result["structuredContent"]) == {"error"}, "a tool error carries only error"
    assert json.loads(result["content"][0]["text"]) == result["structuredContent"], "the text copy"
    return result["structuredContent"]["error"]


def schema_failures(tool: str, structured: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(TOOLS_BY_NAME[tool].output_schema)
    return [f"{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in validator.iter_errors(structured)]


# ---------------------------------------------------------------- tools/list (sections 5, 7.1, 7.2)


@pytest.mark.parametrize("era", ERAS)
def test_tools_list_is_every_v1_tool_in_order_s7_1(backend: FakeBackend, era: Era) -> None:
    async def body(wire: Wire) -> None:
        tools = (await wire.result("tools/list"))["tools"]
        assert tuple(t["name"] for t in tools) == TOOL_NAMES
        for tool in tools:
            contract = TOOLS_BY_NAME[tool["name"]]
            assert tool["inputSchema"] == contract.input_schema
            assert tool["outputSchema"] == contract.output_schema
            assert tool["annotations"]["readOnlyHint"] is contract.read_only
            assert tool["annotations"]["idempotentHint"] is contract.idempotent
            assert tool["annotations"]["openWorldHint"] is False

    over_wire(build_front_end(backend), era, body)


@pytest.mark.parametrize("era", ERAS)
def test_tools_list_has_no_ref_and_declares_2020_12_s5(backend: FakeBackend, era: Era) -> None:
    async def body(wire: Wire) -> None:
        reply = await wire.request("tools/list")
        assert "$ref" not in json.dumps(reply), "published schemas are fully dereferenced (section 5)"
        for tool in reply["result"]["tools"]:
            assert tool["inputSchema"]["$schema"] == DIALECT
            assert tool["outputSchema"]["$schema"] == DIALECT
            assert tool["outputSchema"]["type"] == "object"

    over_wire(build_front_end(backend), era, body)


def test_every_description_states_the_callers_rules_s7_1(backend: FakeBackend) -> None:
    retention = RetentionConfig(retention_days=14, measurement_retention_days=400)

    async def body(wire: Wire) -> None:
        tools = {t["name"]: t for t in (await wire.result("tools/list"))["tools"]}
        for name, tool in tools.items():
            description = tool["description"]
            for rule in (NO_CALLER_STATE, LENGTH_IS_YOURS, RESPELLING_IS_A_HINT):
                assert rule in description, (name, rule)
            retryable = not TOOLS_BY_NAME[name].read_only
            assert (BACKOFF_RULE in description) is retryable, f"{name}: the DC-2 backoff rule"
            assert tool["title"]
        for name in ("get_job", "get_results", "submit_job", "design_voice", "measure_voice"):
            assert retention_clause(retention) in tools[name]["description"], name
        assert "14 days" in tools["submit_job"]["description"]
        assert "400 days" in tools["measure_voice"]["description"]

    over_wire(build_front_end(backend, retention=retention), "modern", body)


def test_backoff_rule_covers_every_tool_that_writes_s7_1() -> None:
    writers = {name for name in TOOL_NAMES if not TOOLS_BY_NAME[name].read_only}
    assert writers == {
        "release_gpu",
        "cancel_job",
        "design_voice",
        "profile_voice",
        "measure_voice",
        "audition_pronunciation",
        "submit_job",
    }


@pytest.mark.parametrize("era", ERAS)
def test_server_identifies_itself_with_instructions_s7(backend: FakeBackend, era: Era) -> None:
    async def body(wire: Wire) -> None:
        result = wire.init_result if era == "legacy" else await wire.result("server/discover")
        assert result is not None
        info = result["serverInfo"] if era == "legacy" else result["_meta"]["io.modelcontextprotocol/serverInfo"]
        assert (info["name"], info["title"]) == (SERVER_NAME, "Narration")
        assert NO_CALLER_STATE in result["instructions"]

    over_wire(build_front_end(backend), era, body)


# ---------------------------------------------------------------- argument validation (sections 3.3, 14)


@pytest.mark.parametrize("era", ERAS)
def test_instruct_is_invalid_argument_naming_the_field_with_a_section_3_3_hint_s14(
    backend: FakeBackend, era: Era
) -> None:
    async def body(wire: Wire) -> None:
        arguments = {**VALID_ARGUMENTS["submit_job"], "instruct": "speak calmly"}
        error = error_of(await wire.call("submit_job", arguments))
        assert error["code"] == codes.INVALID_ARGUMENT
        assert error["field"] == "instruct"
        assert "section 3.3" in error["hint"]
        assert error["retryable"] is False
        assert "speak calmly" not in json.dumps(error), "messages never repeat the caller's values"

    over_wire(build_front_end(backend), era, body)
    assert backend.calls == [], "the backend is never called with invalid arguments"


@pytest.mark.parametrize("tool", TOOL_NAMES)
@pytest.mark.parametrize("field", ["instruct", "style", "emotion", "voice_notes"])
def test_every_tool_refuses_instruction_style_emotion_and_unknown_fields_s3_3(
    backend: FakeBackend, tool: str, field: str
) -> None:
    async def body(wire: Wire) -> None:
        error = error_of(await wire.call(tool, {**VALID_ARGUMENTS[tool], field: "x"}))
        assert (error["code"], error["field"]) == (codes.INVALID_ARGUMENT, field)
        assert "section 3.3" in error["hint"]

    over_wire(build_front_end(backend), "modern", body)


def test_a_nested_unknown_field_is_named_by_its_path_s3_3(backend: FakeBackend) -> None:
    arguments = json.loads(json.dumps(VALID_ARGUMENTS["submit_job"]))
    arguments["segments"][0]["cues"][0]["emotion"] = "sad"

    async def body(wire: Wire) -> None:
        error = error_of(await wire.call("submit_job", arguments))
        assert error["field"] == "segments[0].cues[0].emotion"
        assert "section 3.3" in error["hint"]

    over_wire(build_front_end(backend), "modern", body)


@pytest.mark.parametrize("era", ERAS)
def test_text_mode_written_is_refused_with_a_section_9_2_hint_s3_3(backend: FakeBackend, era: Era) -> None:
    async def body(wire: Wire) -> None:
        error = error_of(await wire.call("submit_job", {**VALID_ARGUMENTS["submit_job"], "text_mode": "written"}))
        assert (error["code"], error["field"]) == (codes.INVALID_ARGUMENT, "text_mode")
        assert "section 9.2" in error["hint"]
        ok = tool_result(await wire.call("submit_job", {**VALID_ARGUMENTS["submit_job"], "text_mode": "spoken"}))
        assert ok["isError"] is False

    over_wire(build_front_end(backend), era, body)


def test_missing_and_malformed_arguments_name_the_field_s14(backend: FakeBackend) -> None:
    async def body(wire: Wire) -> None:
        missing = error_of(await wire.call("get_job", {}))
        assert (missing["field"], missing["hint"]) == ("job_id", "Add job_id; the tool's inputSchema lists its fields.")
        wait = error_of(await wire.call("get_job", {"job_id": JOB_ID, "wait_s": 56}))
        assert wait["field"] == "wait_s" and "at most 55" in wait["message"]
        bad_hash = {**VALID_ARGUMENTS["measure_voice"], "voice": {**VOICE, "sha256": "nope"}}
        assert error_of(await wire.call("measure_voice", bad_hash))["field"] == "voice.sha256"
        no_words = {**VALID_ARGUMENTS["submit_job"], "segments": [{"segment_id": "p01"}]}
        either = error_of(await wire.call("submit_job", no_words))
        assert either["field"] == "segments[0]" and "cues, text" in either["message"]

    over_wire(build_front_end(backend), "modern", body)


@pytest.mark.parametrize(
    ("tool", "change", "field"),
    [
        ("design_voice", {"takes": 2.0}, "takes"),
        ("submit_job", {"options": {"takes": 1.0}}, "options.takes"),
        ("get_job", {"wait_s": float("nan")}, "wait_s"),
        ("get_job", {"wait_s": float("inf")}, "wait_s"),
        ("get_job", {"job_id": JOB_ID + "\n"}, "job_id"),
        ("get_results", {"job_id": JOB_ID.lower()}, "job_id"),
        ("measure_voice", {"voice": {**VOICE, "sha256": VOICE["sha256"] + "\n"}}, "voice.sha256"),
        ("submit_job", {"segments": [{"segment_id": "p01", "text": "Hi.", "scene_seconds": float("nan")}]}, None),
    ],
)
def test_integers_numbers_and_patterns_are_exact_s14(
    backend: FakeBackend, tool: str, change: dict[str, Any], field: str | None
) -> None:
    """``2.0`` is not an integer, NaN and infinity are not numbers, and a pattern's ``$`` is the string's end."""
    field = field or "segments[0].scene_seconds"

    async def body(wire: Wire) -> None:
        error = error_of(await wire.call(tool, {**VALID_ARGUMENTS[tool], **change}))
        assert (error["code"], error["field"]) == (codes.INVALID_ARGUMENT, field)

    over_wire(build_front_end(backend), "modern", body)
    assert backend.calls == []


def _segments(n: int) -> list[dict[str, Any]]:
    return [{"segment_id": f"p{i:03d}", "cues": [{"text": "Before dawn."}]} for i in range(n)]


@pytest.mark.parametrize(
    ("tool", "change", "field"),
    [
        ("submit_job", {"segments": _segments(201)}, "segments"),
        ("check_text", {"segments": _segments(201)}, "segments"),
        ("submit_job", {"segments": [{"segment_id": "p01", "cues": [{"text": "a"}] * 41}]}, "segments[0].cues"),
        (
            "submit_job",
            {"segments": [{"segment_id": "p01", "cues": [{"text": "a" * 601}]}]},
            "segments[0].cues[0].text",
        ),
        ("check_text", {"segments": [{"segment_id": "p01", "text": "a" * 1201}]}, "segments[0].text"),
        ("submit_job", {"hints": [{"term": "Ossavine"}] * 501}, "hints"),
        ("check_text", {"hints": [{"term": "Ossavine"}] * 501}, "hints"),
    ],
)
def test_a_request_size_bound_is_limit_exceeded_with_a_hint_to_split_s14(
    backend: FakeBackend, tool: str, change: dict[str, Any], field: str
) -> None:
    async def body(wire: Wire) -> None:
        error = error_of(await wire.call(tool, {**VALID_ARGUMENTS[tool], **change}))
        assert (error["code"], error["field"], error["retryable"]) == (codes.LIMIT_EXCEEDED, field, False)
        assert "split" in error["hint"].lower(), error["hint"]
        assert field in error["hint"] or field == "hints", error["hint"]

    over_wire(build_front_end(backend), "modern", body)
    assert backend.calls == []


@pytest.mark.parametrize(
    ("tool", "change", "field"),
    [
        ("measure_voice", {"voice": {**VOICE, "transcript": "a" * 601}}, "voice.transcript"),
        ("design_voice", {"description": "a" * 601}, "description"),
        ("submit_job", {"label": "a" * 101}, "label"),
        ("audition_pronunciation", {"variants": [{"label": "a", "respell": "b"}] * 5}, "variants"),
        ("submit_job", {"segments": [{"segment_id": "p01", "text": "Hi.", "attempts": [0, 1, 2, 3]}]}, None),
    ],
)
def test_other_size_bounds_stay_invalid_argument_s14(
    backend: FakeBackend, tool: str, change: dict[str, Any], field: str | None
) -> None:
    async def body(wire: Wire) -> None:
        error = error_of(await wire.call(tool, {**VALID_ARGUMENTS[tool], **change}))
        assert (error["code"], error["field"]) == (codes.INVALID_ARGUMENT, field or "segments[0].attempts")

    over_wire(build_front_end(backend), "modern", body)


def test_a_long_segment_is_not_refused_by_the_front_end_s3_2(backend: FakeBackend) -> None:
    """Length is the caller's decision: nothing but the schema's own bounds is checked here."""
    text = "word " * 230
    arguments = {**VALID_ARGUMENTS["submit_job"], "segments": [{"segment_id": "p01", "text": text.strip()}]}

    async def body(wire: Wire) -> None:
        result = tool_result(await wire.call("submit_job", arguments))
        assert result["isError"] is False
        assert result["structuredContent"]["warnings"][0]["code"] == codes.SEGMENT_TOO_LONG

    over_wire(build_front_end(backend), "modern", body)


def test_the_backend_gets_the_arguments_as_sent_s14(backend: FakeBackend) -> None:
    async def body(wire: Wire) -> None:
        for name in TOOL_NAMES:
            assert tool_result(await wire.call(name, VALID_ARGUMENTS[name]))["isError"] is False, name

    over_wire(build_front_end(backend), "modern", body)
    assert backend.calls == [(name, VALID_ARGUMENTS[name]) for name in TOOL_NAMES]


# ---------------------------------------------------------------- errors (section 14, ADR 0001)


@pytest.mark.parametrize("era", ERAS)
def test_an_unknown_tool_is_json_rpc_invalid_params_s14(backend: FakeBackend, era: Era) -> None:
    async def body(wire: Wire) -> None:
        reply = await wire.call("no_such_tool", {})
        assert reply["error"]["code"] == INVALID_PARAMS

    over_wire(build_front_end(backend), era, body)


@pytest.mark.parametrize("era", ERAS)
def test_an_exception_inside_a_call_is_an_internal_tool_error_s14(
    backend: FakeBackend, log_path: Path, era: Era
) -> None:
    backend.raise_for["check_text"] = RuntimeError("secret caller text")

    async def body(wire: Wire) -> None:
        error = error_of(await wire.call("check_text", VALID_ARGUMENTS["check_text"]))
        assert error["code"] == codes.INTERNAL and error["retryable"] is False
        assert error["details"] == {"tool": "check_text", "exception": "RuntimeError", "log": str(log_path)}
        assert "secret caller text" not in json.dumps(error), "the exception's text goes to the log only"
        assert tool_result(await wire.call("get_job", {"job_id": JOB_ID}))["isError"] is False, "still serving"

    over_wire(build_front_end(backend, log_path=log_path), era, body)


def test_a_result_off_its_output_schema_is_an_internal_tool_error_s14(backend: FakeBackend) -> None:
    backend.result_for["get_job"] = {"job_id": JOB_ID, "status": "exploded"}
    backend.result_for["release_gpu"] = ["not", "an", "object"]

    async def body(wire: Wire) -> None:
        error = error_of(await wire.call("get_job", {"job_id": JOB_ID}))
        assert error["code"] == codes.INTERNAL
        assert error["details"]["output_schema_failures"] == [{"path": "status", "rule": "enum"}]
        assert "exploded" not in json.dumps(error)
        assert error_of(await wire.call("release_gpu", {}))["code"] == codes.INTERNAL

    over_wire(build_front_end(backend), "modern", body)


def test_a_non_finite_number_in_a_result_is_an_internal_tool_error_s14(backend: FakeBackend) -> None:
    """JSON has no NaN or infinity: a result holding one becomes INTERNAL naming the path, never the value."""
    backend.result_for["get_job"] = {**backend.job_json(), "eta_s": float("nan")}
    backend.result_for["get_results"] = {"licence": {"generation_model": float("inf")}}
    backend.raise_for["release_gpu"] = NarrationError(codes.GPU_UNAVAILABLE, "busy", retry_after_s=float("inf"))
    backend.raise_for["cancel_job"] = NarrationError(
        codes.STORE_FULL, "disk", retry_after_s=5.0, details={"free_gb": float("-inf")}
    )
    expected = {
        "get_job": {"path": "eta_s", "rule": "type"},
        "get_results": {"path": "licence.generation_model", "rule": "finite"},
        "release_gpu": {"path": "error.retry_after_s", "rule": "type"},
        "cancel_job": {"path": "error.details.free_gb", "rule": "finite"},
    }

    async def body(wire: Wire) -> None:
        for tool, failure in expected.items():
            reply = await wire.call(tool, VALID_ARGUMENTS[tool])
            error = error_of(reply)
            assert error["code"] == codes.INTERNAL, tool
            assert failure in error["details"]["output_schema_failures"], (tool, error)
            assert "NaN" not in json.dumps(reply) and "Infinity" not in json.dumps(reply)

    over_wire(build_front_end(backend), "modern", body)


def test_non_finite_paths_finds_every_nan_and_infinity_s14() -> None:
    value = {"a": [1.0, float("nan")], "b": {"c": float("-inf"), "d": 2}, "e": "NaN", "f": True}
    assert results.non_finite_paths(value) == ["a[1]", "b.c"]
    assert results.non_finite_paths(float("inf")) == ["$"]


@pytest.mark.parametrize("era", ERAS)
def test_a_narration_error_is_a_tool_error_with_its_hint_s14(backend: FakeBackend, era: Era) -> None:
    backend.raise_for["submit_job"] = NarrationError(codes.VOICE_NOT_MEASURED, "no measurement for this voice")
    backend.raise_for["design_voice"] = NarrationError(
        codes.QUEUE_FULL, "the job queue is full", retry_after_s=42.0, details={"est_drain_s": 42.0}
    )

    async def body(wire: Wire) -> None:
        refused = error_of(await wire.call("submit_job", VALID_ARGUMENTS["submit_job"]))
        assert refused["code"] == codes.VOICE_NOT_MEASURED
        assert refused["hint"] == codes.error_code(codes.VOICE_NOT_MEASURED).hint
        assert "retry_after_s" not in refused and "field" not in refused, "unset optionals are left out"
        full = error_of(await wire.call("design_voice", VALID_ARGUMENTS["design_voice"]))
        assert (full["code"], full["retryable"], full["retry_after_s"]) == (codes.QUEUE_FULL, True, 42.0)

    over_wire(build_front_end(backend), era, body)


def test_a_reused_idempotency_key_is_the_backends_invalid_argument_s7_3(backend: FakeBackend) -> None:
    """DC-6: the backend refuses a key reused for a different request; the front-end passes it on as is."""
    backend.raise_for["submit_job"] = NarrationError(
        codes.INVALID_ARGUMENT,
        "idempotency_key is in use by a different request",
        field="idempotency_key",
        hint="use a new key for a different request",
    )
    arguments = {**VALID_ARGUMENTS["submit_job"], "options": {"idempotency_key": "draft-4"}}

    async def body(wire: Wire) -> None:
        error = error_of(await wire.call("submit_job", arguments))
        assert (error["code"], error["field"], error["retryable"]) == (codes.INVALID_ARGUMENT, "idempotency_key", False)
        assert error["hint"] == "use a new key for a different request"

    over_wire(build_front_end(backend), "modern", body)


def test_unsupported_platform_is_surfaced_like_any_narration_error_s14(backend: FakeBackend) -> None:
    backend.raise_for["release_gpu"] = UnsupportedPlatform("release_gpu", "linux")

    async def body(wire: Wire) -> None:
        error = error_of(await wire.call("release_gpu", {}))
        assert (error["code"], error["retryable"]) == (codes.DAEMON_UNAVAILABLE, False)
        assert error["hint"] == UnsupportedPlatform.HINT
        assert error["details"] == {"operation": "release_gpu", "platform": "linux"}

    over_wire(build_front_end(backend), "modern", body)


# ---------------------------------------------------------------- results (sections 5, 7.2, 7.4, 7.5)


def test_every_fake_result_and_every_tool_error_validates_against_its_output_schema_s7_2(
    backend: FakeBackend, log_path: Path
) -> None:
    """The SDK client checks structuredContent against outputSchema, so every result must pass (section 7.2)."""
    produced: list[tuple[str, dict[str, Any]]] = []

    async def body(wire: Wire) -> None:
        for status in ("completed", "failed", "running"):
            backend.job_status = status
            for name in TOOL_NAMES:
                produced.append((name, tool_result(await wire.call(name, VALID_ARGUMENTS[name]))))
        dry = {**VALID_ARGUMENTS["submit_job"], "options": {"dry_run": True}}
        produced.append(("submit_job", tool_result(await wire.call("submit_job", dry))))
        for name in TOOL_NAMES:
            produced.append((name, tool_result(await wire.call(name, {**VALID_ARGUMENTS[name], "instruct": "x"}))))
            backend.raise_for[name] = NarrationError(codes.RATE_LIMITED, "too many", retry_after_s=3.0)
            produced.append((name, tool_result(await wire.call(name, VALID_ARGUMENTS[name]))))
            backend.raise_for[name] = KeyError("boom")
            produced.append((name, tool_result(await wire.call(name, VALID_ARGUMENTS[name]))))
            backend.raise_for[name] = UnsupportedPlatform(name, "linux")
            produced.append((name, tool_result(await wire.call(name, VALID_ARGUMENTS[name]))))
            del backend.raise_for[name]

    over_wire(build_front_end(backend, log_path=log_path), "modern", body)
    assert len(produced) == 3 * len(TOOL_NAMES) + 1 + 4 * len(TOOL_NAMES)
    kinds = {(name, r["isError"]) for name, r in produced}
    assert kinds == {(name, e) for name in TOOL_NAMES for e in (False, True)}, "each tool, both kinds"
    for name, result in produced:
        assert schema_failures(name, result["structuredContent"]) == [], (name, result["structuredContent"])


@pytest.mark.parametrize("mode", ["legacy", "auto"])
def test_the_sdk_client_accepts_every_result_s5(backend: FakeBackend, mode: str) -> None:
    """The SDK's own client validates every structured result against the tool's outputSchema."""
    from mcp import Client

    async def main() -> None:
        front = build_front_end(backend)
        async with Client(front.server, mode=mode) as client:
            for name in TOOL_NAMES:
                result = await client.call_tool(name, VALID_ARGUMENTS[name])
                assert result.is_error is False and result.structured_content is not None, name
            refused = await client.call_tool("submit_job", {**VALID_ARGUMENTS["submit_job"], "instruct": "x"})
            assert refused.is_error is True

    anyio.run(main)


@pytest.mark.parametrize("era", ERAS)
def test_a_failed_job_is_a_successful_call_carrying_the_jobs_error_s7_4(backend: FakeBackend, era: Era) -> None:
    backend.job_status = "failed"

    async def body(wire: Wire) -> None:
        job = tool_result(await wire.call("get_job", {"job_id": JOB_ID}))
        assert job["isError"] is False
        assert job["structuredContent"]["status"] == "failed"
        assert job["structuredContent"]["error"]["code"] == codes.ENGINE_DRIFT
        results = tool_result(await wire.call("get_results", {"job_id": JOB_ID}))
        assert results["isError"] is False
        assert results["structuredContent"]["job"]["error"]["code"] == codes.ENGINE_DRIFT
        assert "error" not in results["structuredContent"], "the job's error is job.error, not a tool error"

    over_wire(build_front_end(backend), era, body)


@pytest.mark.parametrize("era", ERAS)
def test_results_carry_a_text_copy_and_file_links_never_audio_s5(backend: FakeBackend, era: Era) -> None:
    async def body(wire: Wire) -> None:
        result = tool_result(await wire.call("get_results", {"job_id": JOB_ID}))
        text, *links = result["content"]
        assert text["type"] == "text"
        assert json.loads(text["text"]) == result["structuredContent"]
        assert all(item["type"] == "resource_link" for item in links), "audio is never inlined"
        by_mime = {item["mimeType"]: item for item in links}
        delivery = result["structuredContent"]["segments"][0]["takes"][0]["delivery"]["path"]
        assert by_mime["audio/wav"]["uri"] == Path(delivery).as_uri()
        assert by_mime["audio/wav"]["name"] == f"{TAKE_ID}/delivery.wav"
        assert by_mime["text/markdown"]["uri"] == Path(result["structuredContent"]["report_md"]).as_uri()
        assert all(item["uri"].startswith("file:///") for item in links)

    over_wire(build_front_end(backend), era, body)


def test_a_structured_result_keeps_nulls_on_the_wire_s11_2(backend: FakeBackend) -> None:
    async def body(wire: Wire) -> None:
        result = tool_result(await wire.call("get_results", {"job_id": JOB_ID}))
        cues = result["structuredContent"]["segments"][0]["takes"][0]["cues"]
        assert cues[1]["start_s"] is None and cues[1]["end_s"] is None, "an unplaced cue has null times"

    over_wire(build_front_end(backend), "modern", body)


# ---------------------------------------------------------------- progress and cancellation (section 5)


@pytest.mark.parametrize("era", ERAS)
def test_get_job_forwards_progress_and_keeps_it_monotonic_s5(backend: FakeBackend, era: Era) -> None:
    backend.progress_steps = [(1.0, 10.0, "one"), (0.5, 10.0, "back"), (1.0, 10.0, "same"), (4.0, 12.0, "four")]

    async def body(wire: Wire) -> None:
        reply = await wire.call("get_job", {"job_id": JOB_ID, "wait_s": 5}, meta={"progressToken": "p-9"})
        assert tool_result(reply)["isError"] is False
        progress = [n["params"] for n in wire.notifications if n.get("method") == "notifications/progress"]
        assert [(p["progressToken"], p["progress"], p["total"], p["message"]) for p in progress] == [
            ("p-9", 1.0, 10.0, "one"),
            ("p-9", 4.0, 12.0, "four"),
        ]

    over_wire(build_front_end(backend), era, body)


def test_get_job_without_a_wait_gets_no_progress_callback_s5(backend: FakeBackend) -> None:
    backend.progress_steps = [(1.0, 10.0, "one")]

    async def body(wire: Wire) -> None:
        await wire.call("get_job", {"job_id": JOB_ID}, meta={"progressToken": "p-1"})
        await wire.call("get_job", {"job_id": JOB_ID, "wait_s": 1})
        assert not [n for n in wire.notifications if n.get("method") == "notifications/progress"]

    over_wire(build_front_end(backend), "modern", body)
    assert backend.get_job_progress[0] is None
    assert backend.get_job_progress[1] is not None, "a wait without a token still gets a (silent) callback"


@pytest.mark.parametrize("era", ERAS)
def test_cancelling_get_job_ends_only_the_wait_s5(backend: FakeBackend, era: Era) -> None:
    backend.block_get_job = True

    async def body(wire: Wire) -> None:
        request_id = await wire.send("tools/call", {"name": "get_job", "arguments": {"job_id": JOB_ID, "wait_s": 55}})
        with anyio.fail_after(5):
            await backend.wait_started.wait()
        await wire.notify("notifications/cancelled", {"requestId": request_id, "reason": "user"})
        with anyio.fail_after(5):
            await backend.wait_cancelled.wait()
        backend.block_get_job = False
        assert tool_result(await wire.call("get_server_status"))["isError"] is False
        assert request_id not in wire.replies, "a cancelled request is never answered"

    over_wire(build_front_end(backend), era, body)
    assert [name for name, _ in backend.calls] == ["get_job", "get_server_status"], "the job is not cancelled"


@pytest.mark.parametrize("era", ERAS)
@pytest.mark.parametrize("tool", ["submit_job", "cancel_job", "design_voice"])
def test_a_cancelled_call_still_finishes_its_backend_work_s5(backend: FakeBackend, era: Era, tool: str) -> None:
    """``notifications/cancelled`` must not cut a call off between backend steps; only the reply is dropped."""
    backend.gated = {tool}

    async def body(wire: Wire) -> None:
        request_id = await wire.send("tools/call", {"name": tool, "arguments": VALID_ARGUMENTS[tool]})
        with anyio.fail_after(5):
            await backend.gate_reached.wait()
        await wire.notify("notifications/cancelled", {"requestId": request_id, "reason": "user"})
        # A round trip after the notification: the server has read it, and so has cancelled the handler.
        assert tool_result(await wire.call("get_server_status"))["isError"] is False
        backend.gate.set()
        with anyio.fail_after(5):
            while tool not in backend.passed_gate:
                await anyio.sleep(0.01)
        assert tool_result(await wire.call("get_server_status"))["isError"] is False
        assert request_id not in wire.replies, "a cancelled request is never answered"

    over_wire(build_front_end(backend), era, body)
    if tool == "submit_job":
        assert backend.finished == ["submit_job"], "the backend saw the whole call"


# ---------------------------------------------------------------- resources (section 7.7)


@pytest.mark.parametrize("era", ERAS)
def test_resources_and_templates_are_the_section_7_7_set_s7_7(backend: FakeBackend, era: Era) -> None:
    async def body(wire: Wire) -> None:
        resources = (await wire.result("resources/list"))["resources"]
        templates = (await wire.result("resources/templates/list"))["resourceTemplates"]
        assert [r["uri"] for r in resources] == ["narration://status"]
        assert [t["uriTemplate"] for t in templates] == [r.uri_template for r in RESOURCES if "{" in r.uri_template]
        for item in (*resources, *templates):
            assert item["mimeType"] and item["description"]

    over_wire(build_front_end(backend), era, body)


@pytest.mark.parametrize(
    ("uri", "mime_type", "ttl_ms"),
    [
        ("narration://status", "application/json", 5_000),
        (f"narration://jobs/{JOB_ID}", "application/json", 86_400_000),
        (f"narration://jobs/{JOB_ID}/report", "text/markdown", 86_400_000),
        (f"narration://designs/{DESIGN_ID}", "application/json", 60_000),
        (f"narration://takes/{TAKE_ID}", "application/json", 86_400_000),
        (f"narration://measurements/{VOICE_HASH}", "application/json", 60_000),
    ],
)
def test_a_resource_read_carries_its_ttl_and_private_scope_s7_7(
    backend: FakeBackend, uri: str, mime_type: str, ttl_ms: int
) -> None:
    async def body(wire: Wire) -> None:
        result = await wire.result("resources/read", {"uri": uri})
        (content,) = result["contents"]
        assert (content["uri"], content["mimeType"]) == (uri, mime_type)
        assert (result["ttlMs"], result["cacheScope"]) == (ttl_ms, RESOURCE_CACHE_SCOPE)

    over_wire(build_front_end(backend), "modern", body)


def test_a_running_jobs_resource_is_fresh_for_two_seconds_s7_7(backend: FakeBackend) -> None:
    backend.job_status = "running"

    async def body(wire: Wire) -> None:
        result = await wire.result("resources/read", {"uri": f"narration://jobs/{JOB_ID}"})
        assert result["ttlMs"] == 2_000
        assert json.loads(result["contents"][0]["text"])["status"] == "running"

    over_wire(build_front_end(backend), "modern", body)


@pytest.mark.parametrize("era", ERAS)
def test_a_missing_resource_is_json_rpc_invalid_params_s14(backend: FakeBackend, era: Era) -> None:
    unknown_job = f"narration://jobs/{OTHER_JOB_ID}"

    async def body(wire: Wire) -> None:
        for uri in ("narration://nothing", "narration://jobs/", "https://example.com/", unknown_job):
            reply = await wire.request("resources/read", {"uri": uri})
            assert reply["error"]["code"] == INVALID_PARAMS, uri

    over_wire(build_front_end(backend), era, body)
    assert backend.resource_reads == [unknown_job], "a URI that matches no template never reaches the backend"


@pytest.mark.parametrize("era", ERAS)
@pytest.mark.parametrize(
    "uri",
    [
        "narration://takes/%2e%2e%2f%2e%2e",
        "narration://takes/C:%5CWindows",
        f"narration://takes/{TAKE_ID}\n",
        f"narration://takes/{TAKE_ID}%0A",
        "narration://takes/tk_%38c41d2e07a9b3f55",
        f"narration://takes/{TAKE_ID.upper()}",
        "narration://jobs/job_01jbxq7z3m8v4t2r9k6n5p0w1c",
        f"narration://jobs/{JOB_ID}?x=1",
        "narration://designs/..",
        "narration://measurements/sha256:" + "3F" * 32,
    ],
)
def test_a_malformed_resource_id_is_invalid_params_before_the_backend_s7_7(
    backend: FakeBackend, era: Era, uri: str
) -> None:
    async def body(wire: Wire) -> None:
        reply = await wire.request("resources/read", {"uri": uri})
        assert reply["error"]["code"] == INVALID_PARAMS

    over_wire(build_front_end(backend), era, body)
    assert backend.resource_reads == [], "an id that is not well formed never reaches the backend"


def test_every_resource_template_variable_is_an_id_kind_s7_7() -> None:
    names = {v for r in RESOURCES for v in UriTemplate.parse(r.uri_template).variable_names}
    assert names == {"design_id", "voice_hash", "job_id", "take_id"}
    assert names <= set(ID_PATTERNS)


def test_a_resource_read_failure_is_json_rpc_internal_error_s14(backend: FakeBackend) -> None:
    async def body(wire: Wire) -> None:
        backend.raise_for["read_resource"] = NarrationError(codes.STORE_FULL, "disk", retry_after_s=60.0)
        full = await wire.request("resources/read", {"uri": "narration://status"})
        assert full["error"]["code"] == INTERNAL_ERROR
        assert full["error"]["data"]["error"]["code"] == codes.STORE_FULL
        backend.raise_for["read_resource"] = RuntimeError("secret")
        crash = await wire.request("resources/read", {"uri": "narration://status"})
        assert crash["error"]["code"] == INTERNAL_ERROR and "secret" not in json.dumps(crash)

    over_wire(build_front_end(backend), "modern", body)


def test_job_updates_are_published_for_subscribers_s7_7(backend: FakeBackend) -> None:
    front = build_front_end(backend)
    seen: list[Any] = []
    front.bus.subscribe(seen.append)
    anyio.run(front.job_updated, JOB_ID)
    assert [getattr(e, "uri", None) for e in seen] == [f"narration://jobs/{JOB_ID}"]


# ---------------------------------------------------------------- prompts (section 7.8)


@pytest.mark.parametrize("era", ERAS)
def test_prompts_are_the_section_7_8_set_with_required_arguments_s7_8(backend: FakeBackend, era: Era) -> None:
    async def body(wire: Wire) -> None:
        prompts = (await wire.result("prompts/list"))["prompts"]
        assert [(p["name"], [a["name"] for a in p["arguments"]]) for p in prompts] == [
            (s.name, list(s.arguments)) for s in PROMPTS
        ]
        assert all(a["required"] is True for p in prompts for a in p["arguments"])

    over_wire(build_front_end(backend), era, body)


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("narrate_script", {"voice_path": "C:\\voices\\n.wav"}, ["C:\\voices\\n.wav", "measure_voice", "listen_first"]),
        ("resolve_flags", {"job_id": JOB_ID}, [JOB_ID, "get_results", "never a guarantee"]),
        ("design_narrator_voice", {"brief": "a calm reef guide"}, ["a calm reef guide", "takes 3", "choose"]),
        ("add_pronunciation", {"term": "Ossavine"}, ["Ossavine", "audition_pronunciation", "decides by ear"]),
    ],
)
def test_a_prompt_fills_its_arguments_s7_8(
    backend: FakeBackend, name: str, arguments: dict[str, str], expected: list[str]
) -> None:
    async def body(wire: Wire) -> None:
        result = await wire.result("prompts/get", {"name": name, "arguments": arguments})
        (message,) = result["messages"]
        assert message["role"] == "user"
        for text in expected:
            assert text in message["content"]["text"], text
        assert "{" not in message["content"]["text"], "every placeholder is filled"

    over_wire(build_front_end(backend), "modern", body)


def test_a_bad_prompt_request_is_json_rpc_invalid_params_s14(backend: FakeBackend) -> None:
    async def body(wire: Wire) -> None:
        for params in (
            {"name": "no_such_prompt", "arguments": {}},
            {"name": "resolve_flags", "arguments": {}},
            {"name": "resolve_flags", "arguments": {"job_id": "  "}},
            {"name": "resolve_flags", "arguments": {"job_id": JOB_ID, "style": "x"}},
        ):
            reply = await wire.request("prompts/get", params)
            assert reply["error"]["code"] == INVALID_PARAMS, params

    over_wire(build_front_end(backend), "modern", body)


def test_lists_are_cacheable_and_private_s7_7(backend: FakeBackend) -> None:
    async def body(wire: Wire) -> None:
        for method in ("tools/list", "prompts/list", "resources/list", "resources/templates/list"):
            result = await wire.result(method)
            assert result["cacheScope"] == RESOURCE_CACHE_SCOPE and result["ttlMs"] > 0, method

    over_wire(build_front_end(backend), "modern", body)
