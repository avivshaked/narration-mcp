"""What the MCP surface tells a calling agent (design sections 7, 9.1 and 14).

The texts must name the things that decide whether real use goes well, by the names the schemas really use:
names sent as hints (section 9.1), job size and ``include_words``, ``suggested_take_id``, the voice kept as
is, and the options. A tool this build cannot run is marked wherever it is advertised, and no text enters a
key.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

import narration.keys
from narration.backend.service import KIND_OF_TOOL, RUNNABLE_KINDS
from narration.config import RetentionConfig
from narration.contracts import codes, names
from narration.contracts.errors import NarrationError
from narration.contracts.names import RESOURCES, TOOL_NAMES
from narration.contracts.schemas import TOOLS, TOOLS_BY_NAME
from narration.mcp.descriptions import (
    CLIENT_TEXT_LIMIT,
    NAMES_AS_HINTS,
    NOT_IN_THIS_BUILD,
    OPTIONS,
    PROMPT_TEXTS,
    READ_RESULTS,
    REPORT_RESOURCE,
    SERVER_INSTRUCTIONS,
    VOICE_AS_KEPT,
    not_in_this_build,
    prompt_texts,
    server_instructions,
    tool_description,
)
from narration.mcp.validation import build_validators
from tests.mcp.fake_backend import VALID_ARGUMENTS, VOICE

RETENTION = RetentionConfig()


def _properties(schema: Mapping[str, Any], path: str) -> Iterator[tuple[str, Mapping[str, Any]]]:
    for name, sub in schema.get("properties", {}).items():
        yield f"{path}.{name}", sub
        yield from _properties(sub, f"{path}.{name}")
    items = schema.get("items")
    if isinstance(items, Mapping):
        yield from _properties(items, f"{path}[]")


def _refusal(tool: str, arguments: dict[str, Any]) -> NarrationError:
    with pytest.raises(NarrationError) as caught:
        build_validators()[tool].validate(arguments)
    return caught.value


# ---------------------------------------------------------------- lengths a client shows in full


def test_the_instructions_and_every_tool_description_fit_a_clients_limit() -> None:
    """Claude Code cuts server instructions (and, we believe, tool descriptions) past 2048 characters."""
    long_retention = RetentionConfig(retention_days=99_999, measurement_retention_days=99_999)
    for unbuilt in (NOT_IN_THIS_BUILD, frozenset[str]()):
        assert len(server_instructions(unbuilt)) <= CLIENT_TEXT_LIMIT
        for tool in TOOL_NAMES:
            assert len(tool_description(tool, long_retention, unbuilt=unbuilt)) <= CLIENT_TEXT_LIMIT, tool
    assert server_instructions(NOT_IN_THIS_BUILD) == SERVER_INSTRUCTIONS


# ---------------------------------------------------------------- parameters (section 7.2)


@pytest.mark.parametrize("tool", TOOLS, ids=lambda t: t.name)
def test_every_input_parameter_has_a_description_s7_2(tool: Any) -> None:
    missing = [path for path, sub in _properties(tool.input_schema, tool.name) if not sub.get("description")]
    assert missing == []


# ---------------------------------------------------------------- names as hints (section 9.1)


def test_names_as_hints_reach_every_text_a_caller_reads_before_it_submits_s9_1() -> None:
    assert NAMES_AS_HINTS in SERVER_INSTRUCTIONS
    for tool in ("submit_job", "check_text"):
        assert NAMES_AS_HINTS in tool_description(tool, RETENTION), tool
    narrate = PROMPT_TEXTS["narrate_script"].template
    assert "invented or unusual name" in narrate and "the term alone is enough" in narrate
    hint = TOOLS_BY_NAME["submit_job"].input_schema["properties"]["hints"]
    assert "invented or unusual name" in hint["description"]
    assert "invented or unusual name" in hint["items"]["description"]


def test_names_as_hints_names_the_flag_and_the_fields_it_relies_on_s9_1() -> None:
    assert codes.FLAGS["WER_HIGH"].retake == "at_fail", "a WER_HIGH fail is what the text says costs retakes"
    hint = TOOLS_BY_NAME["submit_job"].input_schema["properties"]["hints"]["items"]
    assert "asr_aliases" in hint["properties"] and "asr_aliases" in NAMES_AS_HINTS
    assert hint["required"] == ["term"], "the term alone is a valid hint"


# ---------------------------------------------------------------- field and option names (sections 7.3, 7.5)


def test_the_options_the_texts_name_are_submit_jobs_real_options_s7_3() -> None:
    options = TOOLS_BY_NAME["submit_job"].input_schema["properties"]["options"]["properties"]
    for name in ("dry_run", "strict_text", "takes", "max_retakes", "priority"):
        assert name in options, name
        assert name in OPTIONS, name
        assert name in tool_description("submit_job", RETENTION), name
        assert options[name].get("description"), name
    assert "interactive" in options["priority"]["enum"] and "'interactive'" in OPTIONS
    assert OPTIONS in SERVER_INSTRUCTIONS


def test_the_results_guidance_names_real_fields_s7_5() -> None:
    get_results = TOOLS_BY_NAME["get_results"]
    assert "include_words" in get_results.input_schema["properties"]
    segment = get_results.output_schema["properties"]["segments"]["items"]["properties"]
    assert {"suggested_take_id", "takes"} <= set(segment)
    for text in (READ_RESULTS, tool_description("get_results", RETENTION)):
        assert "include_words false" in text
        assert "suggested_take_id" in text and "takes[0]" in text
    assert READ_RESULTS in SERVER_INSTRUCTIONS
    narrate = PROMPT_TEXTS["narrate_script"].template
    assert "include_words false" in narrate and "suggested_take_id" in narrate
    assert "8 to 10 segments" in narrate and "8 to 10 segments" in tool_description("submit_job", RETENTION)


def test_every_resource_the_texts_name_is_published_s7_7() -> None:
    templates = {r.uri_template for r in RESOURCES}
    texts = [SERVER_INSTRUCTIONS, REPORT_RESOURCE, *(tool_description(n, RETENTION) for n in TOOL_NAMES)]
    texts += [p.template for p in PROMPT_TEXTS.values()]
    named = {m for text in texts for m in re.findall(r"narration://[\w/{}]+", text)}
    assert "narration://jobs/{job_id}/report" in named
    assert named <= templates


def test_the_voice_is_sent_as_kept_because_its_transcript_is_part_of_it_s10_2() -> None:
    assert VOICE_AS_KEPT in SERVER_INSTRUCTIONS
    assert VOICE_AS_KEPT in tool_description("measure_voice", RETENTION)
    voice = TOOLS_BY_NAME["submit_job"].input_schema["properties"]["voice"]
    assert "never retyped" in voice["properties"]["transcript"]["description"]
    assert "another voice" in voice["description"]

    def voice_hash(transcript: str) -> str:
        return narration.keys.voice_hash(
            model=names.MODEL_QWEN_BASE,
            clip_sha256=VOICE["sha256"],
            transcript=transcript,
            language=names.LANGUAGE,
            x_vector_only_mode=False,
        )

    assert voice_hash("Before dawn.") != voice_hash("Before dawn"), "one character changes the voice"


def test_spec_revision_is_described_as_the_mcp_revision_s7_6() -> None:
    status = TOOLS_BY_NAME["get_server_status"].output_schema["properties"]
    assert "MCP specification revision" in status["spec_revision"]["description"]
    assert "spec_revision is the MCP specification revision" in tool_description("get_server_status", RETENTION)


# ---------------------------------------------------------------- tools not in this build


def test_exactly_the_tools_the_backend_cannot_run_are_marked_not_in_this_build() -> None:
    """A handler that lands (its kind joins ``RUNNABLE_KINDS``) must take its tool's mark off in the same
    change, and a mark taken off must come with the handler: a merge in either order fails here."""
    assert {tool for tool, kind in KIND_OF_TOOL.items() if kind not in RUNNABLE_KINDS} == NOT_IN_THIS_BUILD


@pytest.mark.parametrize("tool", sorted(NOT_IN_THIS_BUILD))
def test_a_tool_not_in_this_build_is_marked_first_in_its_description(tool: str) -> None:
    description = tool_description(tool, RETENTION)
    assert description.startswith(not_in_this_build(tool))
    assert "BACKEND_NOT_INSTALLED" in description
    assert tool in SERVER_INSTRUCTIONS.split("Not in this build yet:")[1]


def test_every_prompt_that_names_a_tool_not_in_this_build_marks_it() -> None:
    for prompt in PROMPT_TEXTS.values():
        for tool in NOT_IN_THIS_BUILD:
            if tool in prompt.template:
                assert f"{tool} is not in this build yet" in prompt.template, (prompt.title, tool)


def test_a_built_tool_carries_no_mark() -> None:
    everything: frozenset[str] = frozenset()
    for tool in TOOL_NAMES:
        assert "not in this build" not in tool_description(tool, RETENTION, unbuilt=everything).lower(), tool
    assert "not in this build" not in server_instructions(everything).lower()
    for prompt in prompt_texts(everything).values():
        assert "not in this build" not in (prompt.description + prompt.template).lower(), prompt.title
    assert "Run audition_pronunciation" in prompt_texts(everything)["add_pronunciation"].template
    assert "design_voice, then a person listens" in server_instructions(everything)


def test_while_design_voice_is_not_in_this_build_the_flow_starts_from_an_allowlisted_clip() -> None:
    unbuilt = frozenset({"design_voice"})
    assert "The flow: a synthetic clip whose sha256 the operator allowlisted" in server_instructions(unbuilt)
    design = prompt_texts(unbuilt)["design_narrator_voice"]
    assert "design_voice is not in this build yet" in design.description
    assert "design_voice is not in this build yet" in design.template


# ---------------------------------------------------------------- hints for common slips (section 14)


def test_an_options_field_sent_at_the_top_is_pointed_into_options_s14() -> None:
    for name, value in (("takes", 2), ("max_retakes", 1), ("dry_run", True), ("strict_text", True)):
        arguments = {**VALID_ARGUMENTS["submit_job"], name: value}
        error = _refusal("submit_job", arguments)
        assert (error.code, error.field) == (codes.INVALID_ARGUMENT, name)
        assert error.hint is not None and f"options.{name}" in error.hint
        assert "section 3.3" not in error.hint, "a misplaced option is not a free-text field"


def test_a_field_is_never_pointed_into_the_refused_controls_s3_3() -> None:
    segment = {**VALID_ARGUMENTS["submit_job"]["segments"][0], "pace": {"factor": 1.0}}
    error = _refusal("submit_job", {**VALID_ARGUMENTS["submit_job"], "segments": [segment]})
    assert error.field == "segments[0].pace"
    assert error.hint is not None and error.hint.startswith("Remove segments[0].pace;")
    assert "Move" not in error.hint and "section 3.3" in error.hint

    inside = {**VALID_ARGUMENTS["submit_job"]["segments"][0], "controls": {"factor": 1.0}}
    error = _refusal("submit_job", {**VALID_ARGUMENTS["submit_job"], "segments": [inside]})
    assert error.field == "segments[0].controls.factor"
    assert error.hint == (
        "Remove segments[0].controls.factor. No current engine supports any control, so every field of "
        "segments[0].controls is refused (CONTROL_UNSUPPORTED): leave segments[0].controls out "
        "(design section 3.3)."
    ), "never pointed further into the refused controls, nor shown its refused fields as accepted"


@pytest.mark.parametrize(("tool", "field"), [("submit_job", "voice"), ("profile_voice", "audio")])
def test_an_upper_case_sha256_is_told_to_lower_case_it_s14(tool: str, field: str) -> None:
    upper = VOICE["sha256"].upper()
    arguments = dict(VALID_ARGUMENTS[tool])
    arguments[field] = {**arguments[field], "sha256": upper}
    error = _refusal(tool, arguments)
    assert (error.code, error.field) == (codes.INVALID_ARGUMENT, f"{field}.sha256")
    assert error.hint is not None and "lower-case" in error.hint
    assert upper not in error.message + error.hint, "messages never repeat the caller's values"


def test_voice_not_synthetic_says_to_restart_the_daemon_and_reconnect_s17_4() -> None:
    hint = codes.ERRORS[codes.VOICE_NOT_SYNTHETIC].hint
    assert "-m narration.admin --config <service_root>/narration.toml voices allow <clip.wav>" in hint
    assert "Only a person allows a clip" in hint and "a calling agent must not run the command" in hint
    assert "with [daemon] autostart on" in hint
    assert hint.index("daemon stop") < hint.index("/mcp"), "the daemon first (WP45's restart advice)"


def _hints() -> Iterator[tuple[str, str]]:
    """Every hint the source writes out: the default hint of each code in ``codes.ERRORS``, and every literal
    ``hint=`` given to an error anywhere under ``src/narration`` (the text of an f-string without its
    fields)."""
    for code, error in codes.ERRORS.items():
        yield code, error.hint
    package = Path(narration.keys.__file__).parents[1]
    for source in sorted(package.rglob("*.py")):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.keyword) and node.arg == "hint":
                parts = [
                    p.value for p in ast.walk(node.value) if isinstance(p, ast.Constant) and isinstance(p.value, str)
                ]
                if parts:
                    yield f"{source.relative_to(package).as_posix()}:{node.value.lineno}", "".join(parts)


def test_no_error_hint_offers_a_tool_this_build_cannot_run_s7() -> None:
    """A hint may name a tool that is not in this build only as "where <tool> is available", so it is true
    before that tool lands and after."""
    hints = dict(_hints())
    assert codes.VOICE_NOT_SYNTHETIC in hints and "measure/transcript.py" in " ".join(hints), "the scan is live"
    for where, hint in hints.items():
        for tool in NOT_IN_THIS_BUILD:
            assert tool not in hint.replace(f"where {tool} is available", ""), f"{where} offers {tool}"


# ---------------------------------------------------------------- keys (section 10.2)


def test_the_keys_never_load_the_descriptions_or_the_schemas_s10_2() -> None:
    """Imported alone, in a fresh interpreter, ``narration.keys`` loads neither ``narration.mcp`` (the
    descriptions, instructions and prompts) nor ``contracts.schemas`` (the schema descriptions), directly or
    through anything it imports, so no key can hash those texts."""
    script = "import json, sys, narration.keys; print(json.dumps(sorted(sys.modules)))"
    ran = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    loaded = set(json.loads(ran.stdout))
    assert "narration.keys" in loaded
    assert not {m for m in loaded if m == "narration.mcp" or m.startswith("narration.mcp.")}
    assert "narration.contracts.schemas" not in loaded


def test_the_keys_never_name_the_error_codes_or_their_hints_s10_2() -> None:
    """``contracts.codes`` (the error hints) is loaded with ``narration.keys`` through the contracts it uses
    (``config``, ``errors``), but nothing under ``narration.keys`` imports it or names it."""
    folder = Path(narration.keys.__file__).parent
    for source in folder.glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "narration.contracts.codes", source.name
                assert "codes" not in {a.name for a in node.names}, source.name
            elif isinstance(node, ast.Import):
                assert "narration.contracts.codes" not in {a.name for a in node.names}, source.name
            elif isinstance(node, ast.Name):
                assert node.id not in {"codes", "ERRORS"}, source.name
