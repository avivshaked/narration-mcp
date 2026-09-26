"""Spike (j): a throwaway MCP server on the low-level ``Server`` of ``mcp`` 2.2.0 (plan.md WP17).

What it shows (design sections 5, 7 and 14):

- tools published with hand-written, fully dereferenced JSON Schemas (no ``$ref``), each with an
  ``outputSchema`` whose root is ``"type": "object"``;
- argument validation inside the handler (jsonschema, Draft 2020-12), so a bad argument comes back as a
  tool result with ``isError: true``, ``structuredContent: {"error": {...}}`` and a text copy;
- an unknown tool as JSON-RPC -32602;
- ``notifications/progress`` during a long call that carried a ``progressToken``, and cancellation of it;
- resources with a template, ``ttlMs`` and ``cacheScope: "private"``; a prompt; ``subscriptions/listen``;
- ``structuredContent`` with a text copy, and a ``resource_link`` content item.

This is not product code. WP17's front-end is built in ``src/narration/mcp`` once the contracts (WP01)
exist. The names here only echo the design's.

Run it on stdio: ``uv run python spikes/j-mcp-sdk/spike_server.py``. If ``SPIKE_INBOUND_LOG`` names a
file, every inbound request and notification (method, id, era, params including ``_meta``) is appended
to it as a JSON line: a way to see what a real client, such as Claude Code, sends.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import anyio
import mcp_types as types
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError, best_match
from mcp.server import Server, ServerRequestContext
from mcp.server.context import CallNext, HandlerResult
from mcp.server.stdio import stdio_server
from mcp.server.subscriptions import InMemorySubscriptionBus, ListenHandler, ResourceUpdated
from mcp.shared.exceptions import MCPError
from mcp.shared.uri_template import UriTemplate

# ---------------------------------------------------------------- schemas (design section 7.2, inlined)

ID_SCHEMA: dict[str, Any] = {"type": "string", "pattern": "^[a-z0-9][a-z0-9._-]{0,63}$"}

ERROR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["code", "message", "retryable"],
    "properties": {
        "code": {"type": "string"},
        "message": {"type": "string"},
        "retryable": {"type": "boolean"},
        "hint": {"type": "string"},
        "field": {"type": "string"},
        "details": {"type": "object"},
    },
}

VOICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["path", "sha256", "transcript"],
    "properties": {
        "path": {"type": "string", "description": "absolute path of a WAV on a local drive"},
        "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "transcript": {"type": "string", "minLength": 1, "maxLength": 600},
    },
}

CHECK_LINE_INPUT: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["voice", "text"],
    "properties": {
        "voice": VOICE_SCHEMA,
        "text": {
            "type": "string",
            "minLength": 1,
            "maxLength": 600,
            "description": "the words to be spoken, as the caller wants them heard",
        },
        "takes": {"type": "integer", "minimum": 1, "maximum": 3, "default": 1},
    },
}

CHECK_LINE_OUTPUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "spoken": {"type": "string"},
        "spoken_chars": {"type": "integer"},
        "takes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "take_id": {"type": "string"},
                    "delivery": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "sha256": {"type": "string"}},
                    },
                },
            },
        },
        "error": ERROR_SCHEMA,
    },
}

WAIT_JOB_INPUT: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["job_id"],
    "properties": {
        "job_id": ID_SCHEMA,
        "wait_s": {"type": "number", "minimum": 0, "maximum": 55, "default": 0},
    },
}

WAIT_JOB_OUTPUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "job_id": {"type": "string"},
        "status": {"enum": ["queued", "running", "completed"]},
        "waited_s": {"type": "number"},
        "error": ERROR_SCHEMA,
    },
}

RAISE_DEMO_INPUT: dict[str, Any] = {"type": "object", "additionalProperties": False, "properties": {}}

NO_STATE = "The service keeps no caller state."

TOOLS: list[types.Tool] = [
    types.Tool(
        name="check_line",
        title="Check a line (spike)",
        description=(
            "Spike stand-in for a synchronous tool. Returns the line's spoken form and a fake take with a "
            f"resource_link. {NO_STATE} A paragraph's length is the caller's decision."
        ),
        input_schema=CHECK_LINE_INPUT,
        output_schema=CHECK_LINE_OUTPUT,
        annotations=types.ToolAnnotations(read_only_hint=True),
    ),
    types.Tool(
        name="wait_job",
        title="Wait on a job (spike)",
        description=(
            "Spike stand-in for get_job with wait_s: waits, sending notifications/progress when the request "
            f"carried a progressToken. notifications/cancelled ends only the wait. {NO_STATE}"
        ),
        input_schema=WAIT_JOB_INPUT,
        output_schema=WAIT_JOB_OUTPUT,
        annotations=types.ToolAnnotations(read_only_hint=True),
    ),
    types.Tool(
        name="raise_demo",
        title="Raise (spike)",
        description="Spike only: raises inside the handler, uncaught, to show how the SDK maps it.",
        input_schema=RAISE_DEMO_INPUT,
    ),
]
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}
VALIDATORS = {tool.name: Draft202012Validator(tool.input_schema) for tool in TOOLS}

STATUS_URI = "spike://status"
JOB_TEMPLATE = "spike://jobs/{job_id}"
JOB_URI = UriTemplate.parse(JOB_TEMPLATE)

# ---------------------------------------------------------------- in-handler validation (design section 14)

HINTS = {
    "instruct": (
        "instruct is not an input of this service: a voice's delivery comes from its reference clip "
        "(design section 3.3). Remove the field."
    ),
}


def _field_path(parts: list[str | int]) -> str:
    """``["voice", "path"]`` -> ``voice.path``; ``["segments", 0, "text"]`` -> ``segments[0].text``."""
    out = ""
    for part in parts:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else part)
    return out


def argument_error(tool: types.Tool, arguments: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate ``arguments`` against the tool's own published schema; return an Error or None."""
    error: ValidationError | None = best_match(VALIDATORS[tool.name].iter_errors(dict(arguments)))
    if error is None:
        return None
    parts: list[str | int] = list(error.absolute_path)
    hint: str
    if error.validator == "additionalProperties" and isinstance(error.instance, dict):
        allowed: dict[str, Any] = error.schema.get("properties", {}) if isinstance(error.schema, dict) else {}
        extra = sorted(str(k) for k in error.instance if k not in allowed)
        name = extra[0] if extra else "?"
        field = _field_path([*parts, name])
        message = f"unknown field {field!r}"
        hint = HINTS.get(name, f"Remove {field}; accepted here: {', '.join(sorted(allowed))}.")
    elif error.validator == "required" and isinstance(error.instance, dict):
        required = error.validator_value if isinstance(error.validator_value, list) else []
        missing = [str(k) for k in required if k not in error.instance]
        field = _field_path([*parts, missing[0] if missing else "?"])
        message = f"missing required field {field!r}"
        hint = f"Add {field}."
    else:
        field = _field_path(parts)
        message = f"{field or 'arguments'}: {error.message}"
        hint = f"Fix {field or 'the arguments'} to match the tool's inputSchema."
    return {
        "code": "INVALID_ARGUMENT",
        "message": message,
        "retryable": False,
        "field": field,
        "hint": hint,
        "details": {"schema_rule": str(error.validator)},
    }


def tool_error(error: dict[str, Any]) -> types.CallToolResult:
    """A tool execution error: ``isError`` + ``structuredContent: {"error": ...}`` + a text copy."""
    structured = {"error": error}
    text = json.dumps(structured, ensure_ascii=False)
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)], structured_content=structured, is_error=True
    )


def tool_success(structured: dict[str, Any], *extra: types.ContentBlock) -> types.CallToolResult:
    """``structuredContent`` plus a text copy for older clients, then any further content items."""
    text = json.dumps(structured, ensure_ascii=False)
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text), *extra], structured_content=structured
    )


# ---------------------------------------------------------------- the server


class InboundLog:
    """Middleware that appends every inbound message, raw and before validation, to a JSON-lines file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    async def __call__(self, ctx: ServerRequestContext[Any, Any], call_next: CallNext) -> HandlerResult:
        record = {"method": ctx.method, "id": ctx.request_id, "era": ctx.protocol_version, "params": ctx.params}
        with self._path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return await call_next(ctx)


def build_server() -> Server[Any]:
    """Build the spike server. Every handler is a plain ``(ctx, params)`` coroutine."""
    bus = InMemorySubscriptionBus()

    async def list_tools(
        ctx: ServerRequestContext[Any, Any], params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=TOOLS)

    async def call_tool(
        ctx: ServerRequestContext[Any, Any], params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        tool = TOOLS_BY_NAME.get(params.name)
        if tool is None:
            # An unknown tool is a protocol error (design section 14), not a tool result.
            raise MCPError(
                code=types.INVALID_PARAMS, message=f"Unknown tool: {params.name}", data={"name": params.name}
            )
        arguments = params.arguments or {}
        if tool.name == "raise_demo":
            raise RuntimeError("raise_demo: an exception the handler did not catch")
        try:
            error = argument_error(tool, arguments)
            if error is not None:
                return tool_error(error)
            if tool.name == "check_line":
                return check_line(arguments)
            return await wait_job(ctx, arguments)
        except anyio.get_cancelled_exc_class():
            raise
        except Exception as exc:  # the front-end's own catch-all; never the SDK's (see the ADR)
            return tool_error({"code": "INTERNAL", "message": type(exc).__name__, "retryable": False})

    def check_line(arguments: Mapping[str, Any]) -> types.CallToolResult:
        spoken = " ".join(str(arguments["text"]).split())
        take_id = "tk_0123456789abcdef"
        path = f"<store_root>/takes/01/{take_id}/delivery.wav"
        structured = {
            "spoken": spoken,
            "spoken_chars": len(spoken),
            "takes": [{"take_id": take_id, "delivery": {"path": path, "sha256": "0" * 64}}],
        }
        link = types.ResourceLink(
            type="resource_link",
            uri=f"file:///store_root/takes/01/{take_id}/delivery.wav",
            name=f"{take_id}.wav",
            mime_type="audio/wav",
        )
        return tool_success(structured, link)

    async def wait_job(ctx: ServerRequestContext[Any, Any], arguments: Mapping[str, Any]) -> types.CallToolResult:
        job_id = str(arguments["job_id"])
        wait_s = float(arguments.get("wait_s", 0))
        step = 0.25
        waited = 0.0
        try:
            while waited < wait_s:
                await anyio.sleep(step)
                waited = min(wait_s, waited + step)
                # A no-op unless the request carried _meta.progressToken.
                await ctx.session.report_progress(waited, wait_s, f"waited {waited:.2f} of {wait_s:.2f} s")
        except anyio.get_cancelled_exc_class():
            print(f"spike_server: wait on {job_id} cancelled after {waited:.2f} s", file=sys.stderr, flush=True)
            raise
        await bus.publish(ResourceUpdated(uri=f"spike://jobs/{job_id}"))
        return tool_success({"job_id": job_id, "status": "completed", "waited_s": waited})

    async def list_resources(
        ctx: ServerRequestContext[Any, Any], params: types.PaginatedRequestParams | None
    ) -> types.ListResourcesResult:
        status = types.Resource(uri=STATUS_URI, name="status", mime_type="application/json")
        return types.ListResourcesResult(resources=[status], ttl_ms=60_000, cache_scope="private")

    async def list_resource_templates(
        ctx: ServerRequestContext[Any, Any], params: types.PaginatedRequestParams | None
    ) -> types.ListResourceTemplatesResult:
        job = types.ResourceTemplate(
            uri_template=JOB_TEMPLATE, name="job", description="a job (subscribable)", mime_type="application/json"
        )
        return types.ListResourceTemplatesResult(resource_templates=[job], ttl_ms=60_000, cache_scope="private")

    async def read_resource(
        ctx: ServerRequestContext[Any, Any], params: types.ReadResourceRequestParams
    ) -> types.ReadResourceResult:
        uri = str(params.uri)
        if uri == STATUS_URI:
            body: dict[str, Any] = {"state": "idle"}
            ttl_ms = 5_000
        elif (match := JOB_URI.match(uri)) is not None:
            body = {"job_id": match["job_id"], "status": "running"}
            ttl_ms = 2_000
        else:
            raise MCPError(code=types.INVALID_PARAMS, message="Resource not found", data={"uri": uri})
        contents = types.TextResourceContents(uri=uri, text=json.dumps(body), mime_type="application/json")
        return types.ReadResourceResult(contents=[contents], ttl_ms=ttl_ms, cache_scope="private")

    async def list_prompts(
        ctx: ServerRequestContext[Any, Any], params: types.PaginatedRequestParams | None
    ) -> types.ListPromptsResult:
        argument = types.PromptArgument(name="voice_path", description="the clip to narrate with", required=True)
        prompt = types.Prompt(name="narrate_script", description="Narrate a script (spike).", arguments=[argument])
        return types.ListPromptsResult(prompts=[prompt])

    async def get_prompt(
        ctx: ServerRequestContext[Any, Any], params: types.GetPromptRequestParams
    ) -> types.GetPromptResult:
        if params.name != "narrate_script":
            raise MCPError(code=types.INVALID_PARAMS, message=f"Unknown prompt: {params.name}")
        voice_path = (params.arguments or {}).get("voice_path")
        if not voice_path:
            raise MCPError(code=types.INVALID_PARAMS, message="Missing required argument: voice_path")
        text = f"1. measure_voice for {voice_path} if it has no measurement. 2. check_text. 3. submit_job."
        message = types.PromptMessage(role="user", content=types.TextContent(type="text", text=text))
        return types.GetPromptResult(description="Narrate a script (spike).", messages=[message])

    server: Server[Any] = Server(
        "narration-spike",
        version="0.0.0+spike",
        instructions=f"Spike server for plan.md WP17. {NO_STATE}",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        on_list_resources=list_resources,
        on_list_resource_templates=list_resource_templates,
        on_read_resource=read_resource,
        on_subscriptions_listen=ListenHandler(bus),
        on_list_prompts=list_prompts,
        on_get_prompt=get_prompt,
    )
    # Drop the SDK's default OpenTelemetry middleware (a no-op without an exporter); log inbound if asked.
    log_path = os.environ.get("SPIKE_INBOUND_LOG")
    server.middleware[:] = [InboundLog(Path(log_path))] if log_path else []
    return server


async def serve() -> None:
    """Serve one client on stdio, in whichever protocol era its first request opens."""
    server = build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> int:
    """Entry point."""
    anyio.run(serve)
    return 0


if __name__ == "__main__":
    sys.exit(main())
