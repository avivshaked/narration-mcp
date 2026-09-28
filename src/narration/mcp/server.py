"""The MCP front-end: tools, resources and prompts over a ``Backend`` (design sections 5, 7 and 14; ADR 0001).

It is built on the low-level ``mcp.server.Server`` of ``mcp`` 2.2.0, which answers both protocol eras: a
legacy ``initialize`` handshake and 2026-07-28's per-request ``_meta`` with ``server/discover``.

- ``tools/list`` publishes every v1 tool in the section 7.1 order, with the contract's input and output
  schemas (no ``$ref``, dialect 2020-12) and the descriptions of ``descriptions``.
- ``tools/call`` validates the arguments inside the handler and calls the backend. An argument failure, a
  ``NarrationError`` and any other exception all come back as tool errors (``isError: true``), the last as
  ``INTERNAL``; only an unknown tool is a JSON-RPC error (-32602). Every result is checked against the
  tool's ``outputSchema`` before it is sent, because the SDK client checks it too.
- Cancellation (``notifications/cancelled``, or the client going away) interrupts a read-only call at once,
  so it ends a ``get_job`` wait but never the job; nothing here catches it. A tool that writes runs its
  backend call shielded, up to ``WRITE_DEADLINE_S``: a cancelled ``submit_job`` still finishes what it
  started (writing its job row), and the SDK drops the reply, as MCP requires. A write still running at the
  deadline is cancelled and answered with ``INTERNAL``.
- Resources are the ``narration://`` templates of section 7.7, read through the backend, each read with
  its ``ttlMs`` and ``cacheScope: "private"``. A missing resource, or a URI whose ids are malformed, is
  JSON-RPC -32602; a malformed id never reaches the backend, which always gets the canonical URI.
- Prompts are the four of section 7.8.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import anyio
import mcp_types as types
from mcp.server import Server, ServerRequestContext
from mcp.server.caching import CacheHint
from mcp.server.stdio import stdio_server
from mcp.server.subscriptions import InMemorySubscriptionBus, ListenHandler, ResourceUpdated, SubscriptionBus
from mcp.shared.exceptions import MCPError
from mcp.shared.uri_template import UriTemplate

import narration
from narration.config import RetentionConfig
from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.interfaces import Backend, ProgressCallback
from narration.contracts.names import (
    ID_PATTERNS,
    PROMPTS,
    RESOURCE_CACHE_SCOPE,
    RESOURCES,
    SERVER_NAME,
    TOOL_NAMES,
    IdKind,
    ResourceTemplate,
    is_id,
)
from narration.contracts.schemas import TOOLS_BY_NAME
from narration.contracts.serial import to_json

from . import results
from .descriptions import (
    PROMPT_TEXTS,
    RESOURCE_DESCRIPTIONS,
    SERVER_INSTRUCTIONS,
    SERVER_TITLE,
    TOOL_TEXTS,
    tool_description,
)
from .validation import ArgumentValidator, ExactValidator, build_validators

logger = logging.getLogger("narration.mcp")

LIST_CACHE_TTL_MS: Final = 3_600_000
"""How long a client may cache the lists and ``server/discover``: they change only with a new server."""
DESTRUCTIVE_TOOLS: Final = frozenset({"cancel_job"})
"""Tools whose effect is not additive (MCP ``destructiveHint``): cancelling stops work that was asked for."""
RESOURCE_NOT_FOUND_CODES: Final = frozenset({codes.NOT_FOUND, codes.INVALID_ARGUMENT, codes.PATH_NOT_ALLOWED})
"""Backend errors on a resource read that mean "no such resource" (JSON-RPC -32602)."""
SHIELDED_TOOLS: Final = frozenset(
    {
        "release_gpu",
        "cancel_job",
        "design_voice",
        "profile_voice",
        "measure_voice",
        "audition_pronunciation",
        "submit_job",
    }
)
"""The tools that write (to the store, or a command to the daemon): ``release_gpu``, ``cancel_job``,
``design_voice``, ``profile_voice``, ``measure_voice``, ``audition_pronunciation`` and ``submit_job``. Their
backend call is shielded from cancellation, so it is never cut off between two backend steps. A read-only
call (``get_job``'s wait included) is interrupted at once. An explicit set, not one derived from ``read_only``:
``get_job`` is read-only and not shielded, though it may start the daemon for an active job that none serves
(a start that finishes even when the call is cancelled, since the backend runs it in a worker thread)."""
WRITE_DEADLINE_S: Final = 30.0
"""How long a shielded write may run, in seconds. **For WP36: every backend call of a tool in
``SHIELDED_TOOLS`` must finish well inside it** (it is store work, never an open-ended wait). A call still
running at the deadline is cancelled, and the tool answers ``INTERNAL`` ("the backend did not answer in
time"). It also bounds how long a closing connection waits for a write in flight."""

type Context = ServerRequestContext[Any, Any]
type ToolCall = Callable[[Mapping[str, Any], Context], Awaitable[dict[str, Any]]]


def build_tools(retention: RetentionConfig) -> list[types.Tool]:
    """The published tools, in the section 7.1 order."""
    tools: list[types.Tool] = []
    for name in TOOL_NAMES:
        schema = TOOLS_BY_NAME[name]
        tools.append(
            types.Tool(
                name=name,
                title=TOOL_TEXTS[name].title,
                description=tool_description(name, retention),
                input_schema=schema.input_schema,
                output_schema=schema.output_schema,
                annotations=types.ToolAnnotations(
                    title=TOOL_TEXTS[name].title,
                    read_only_hint=schema.read_only,
                    destructive_hint=name in DESTRUCTIVE_TOOLS,
                    idempotent_hint=schema.idempotent,
                    open_world_hint=False,
                ),
            )
        )
    return tools


def _is_template(resource: ResourceTemplate) -> bool:
    return "{" in resource.uri_template


def build_resources() -> list[types.Resource]:
    """The fixed resources (``narration://status``)."""
    return [
        types.Resource(
            uri=r.uri_template,
            name=r.name,
            description=RESOURCE_DESCRIPTIONS[r.name],
            mime_type=r.mime_type,
        )
        for r in RESOURCES
        if not _is_template(r)
    ]


def build_resource_templates() -> list[types.ResourceTemplate]:
    """The resource templates of section 7.7."""
    return [
        types.ResourceTemplate(
            uri_template=r.uri_template,
            name=r.name,
            description=RESOURCE_DESCRIPTIONS[r.name],
            mime_type=r.mime_type,
        )
        for r in RESOURCES
        if _is_template(r)
    ]


def build_prompts() -> list[types.Prompt]:
    """The prompts of section 7.8; every argument is required."""
    prompts: list[types.Prompt] = []
    for spec in PROMPTS:
        text = PROMPT_TEXTS[spec.name]
        arguments = [types.PromptArgument(name=a, description=text.arguments[a], required=True) for a in spec.arguments]
        prompts.append(
            types.Prompt(name=spec.name, title=text.title, description=text.description, arguments=arguments)
        )
    return prompts


def resolve_resource(uri: str) -> tuple[ResourceTemplate, str] | None:
    """The section 7.7 resource a URI names, with its canonical URI; None when it names none.

    Every template variable must be a well-formed id of its kind (``names.is_id``: ``job_id``,
    ``design_id``, ``take_id``, ``voice_hash``). The URI must then be one of two spellings of those ids:
    the template filled with them as they are (the canonical URI), or the template's RFC 6570 expansion,
    which a conforming client builds and which percent-encodes the ``:`` of a ``voice_hash``. Any other
    encoding, a query or a fragment is refused. So an id that is malformed or smuggled in (``%2e%2e%2f``, a
    drive path, a trailing newline) never reaches the backend, and the backend always gets the canonical URI.
    """
    for resource in RESOURCES:
        if not _is_template(resource):
            if uri == resource.uri_template:
                return resource, uri
            continue
        template = UriTemplate.parse(resource.uri_template)
        variables = template.match(uri)
        if variables is None:
            continue
        ids: dict[str, str] = {}
        for name, value in variables.items():
            if name not in ID_PATTERNS or not is_id(cast(IdKind, name), value):
                return None
            ids[name] = cast(str, value)
        canonical = resource.uri_template
        for name, value in ids.items():
            canonical = canonical.replace("{" + name + "}", value)
        if uri in (canonical, template.expand(ids)):
            return resource, canonical
    return None


def monotonic_progress(send: ProgressCallback) -> ProgressCallback:
    """Wrap a progress sender so that ``progress`` only increases, as MCP requires (section 5).

    A report that does not advance ``progress`` is dropped; ``total`` may grow (retakes).
    """
    last: list[float] = []

    async def report(progress: float, total: float | None, message: str | None) -> None:
        if last and progress <= last[0]:
            return
        last[:] = [progress]
        await send(progress, total, message)

    return report


@dataclass(frozen=True, slots=True)
class FrontEnd:
    """A built front-end: the MCP ``server`` and the ``bus`` that job-resource updates are published on."""

    server: Server[Any]
    bus: SubscriptionBus

    async def run_stdio(self) -> None:
        """Serve one client on stdin/stdout, in whichever protocol era its first message opens."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, self.server.create_initialization_options())

    async def job_updated(self, job_id: str) -> None:
        """Tell subscribers (``subscriptions/listen``) that ``narration://jobs/{job_id}`` changed."""
        await self.bus.publish(ResourceUpdated(uri=f"narration://jobs/{job_id}"))


def build_front_end(
    backend: Backend,
    *,
    retention: RetentionConfig | None = None,
    log_path: Path | None = None,
    bus: SubscriptionBus | None = None,
    check_results: bool = True,
    write_deadline_s: float = WRITE_DEADLINE_S,
) -> FrontEnd:
    """Build the MCP front-end over ``backend``.

    ``retention`` gives the periods the descriptions state (section 15). ``log_path`` is the log file an
    ``INTERNAL`` error points to. ``check_results`` checks every result against its tool's
    ``outputSchema`` and turns a mismatch into ``INTERNAL``, so that a client never rejects one.
    ``write_deadline_s`` bounds a shielded write (``WRITE_DEADLINE_S``).
    """
    retention = retention if retention is not None else RetentionConfig()
    bus = bus if bus is not None else InMemorySubscriptionBus()
    tools = build_tools(retention)
    validators: dict[str, ArgumentValidator] = build_validators()
    output_validators = {name: ExactValidator(TOOLS_BY_NAME[name].output_schema) for name in TOOL_NAMES}
    prompt_specs = {p.name: p for p in PROMPTS}

    async def get_job(args: Mapping[str, Any], ctx: Context) -> dict[str, Any]:
        progress: ProgressCallback | None = None
        if float(args.get("wait_s", 0)) > 0:
            progress = monotonic_progress(ctx.session.report_progress)
        return await backend.get_job(args, progress)

    def plain(method: Callable[[Mapping[str, Any]], Awaitable[dict[str, Any]]]) -> ToolCall:
        async def call(args: Mapping[str, Any], ctx: Context) -> dict[str, Any]:
            return await method(args)

        return call

    dispatch: dict[str, ToolCall] = {
        "get_server_status": plain(backend.get_server_status),
        "release_gpu": plain(backend.release_gpu),
        "get_job": get_job,
        "get_results": plain(backend.get_results),
        "cancel_job": plain(backend.cancel_job),
        "design_voice": plain(backend.design_voice),
        "profile_voice": plain(backend.profile_voice),
        "measure_voice": plain(backend.measure_voice),
        "check_text": plain(backend.check_text),
        "audition_pronunciation": plain(backend.audition_pronunciation),
        "submit_job": plain(backend.submit_job),
    }
    assert tuple(dispatch) == TOOL_NAMES, "one backend call per tool"

    def checked(name: str, result: types.CallToolResult) -> types.CallToolResult:
        if not check_results or result.structured_content is None:
            return result
        failures = results.output_failures(output_validators[name], result.structured_content)
        if not failures:
            return result
        logger.error("%s: result does not match its outputSchema: %s", name, failures)
        return results.tool_error(results.output_mismatch_error(name, failures, log_path=log_path))

    async def list_tools(ctx: Context, params: types.PaginatedRequestParams | None) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools)

    async def call_tool(ctx: Context, params: types.CallToolRequestParams) -> types.CallToolResult:
        name = params.name
        call = dispatch.get(name)
        if call is None:
            # An unknown tool is a protocol error (section 14), not a tool result.
            raise MCPError(code=types.INVALID_PARAMS, message=f"Unknown tool: {name}", data={"name": name})
        arguments: dict[str, Any] = dict(params.arguments or {})
        try:
            validators[name].validate(arguments)
            structured: object = None
            if name in SHIELDED_TOOLS:
                # A cancellation must not cut a write off between two backend steps (a job row written, its
                # plan not); the SDK still drops the reply to a cancelled request. The deadline keeps a
                # stuck backend from holding the server open.
                with anyio.move_on_after(write_deadline_s, shield=True) as deadline:
                    structured = await call(arguments, ctx)
                if deadline.cancelled_caught:
                    logger.error("%s: the backend did not answer within %s s", name, write_deadline_s)
                    error = results.deadline_error(name, write_deadline_s, log_path=log_path)
                    return checked(name, results.tool_error(error))
            else:
                structured = await call(arguments, ctx)
            if not isinstance(structured, dict):
                raise TypeError(f"the backend returned {type(structured).__name__}, not an object")
            return checked(name, results.success(structured))
        except NarrationError as exc:
            return checked(name, results.from_narration_error(exc))
        except Exception as exc:  # never BaseException: cancellation must pass through (ADR 0001)
            logger.exception("%s failed inside the service", name)
            return checked(name, results.tool_error(results.internal_error(name, exc, log_path=log_path)))

    async def list_resources(ctx: Context, params: types.PaginatedRequestParams | None) -> types.ListResourcesResult:
        return types.ListResourcesResult(resources=build_resources())

    async def list_resource_templates(
        ctx: Context, params: types.PaginatedRequestParams | None
    ) -> types.ListResourceTemplatesResult:
        return types.ListResourceTemplatesResult(resource_templates=build_resource_templates())

    async def read_resource(ctx: Context, params: types.ReadResourceRequestParams) -> types.ReadResourceResult:
        uri = str(params.uri)
        resolved = resolve_resource(uri)
        if resolved is None:
            raise MCPError(code=types.INVALID_PARAMS, message="Resource not found", data={"uri": uri})
        _, canonical = resolved
        try:
            content = await backend.read_resource(canonical)
        except NarrationError as exc:
            data = {"uri": uri, "error": to_json(exc.error)}
            if exc.code in RESOURCE_NOT_FOUND_CODES:
                raise MCPError(code=types.INVALID_PARAMS, message="Resource not found", data=data) from exc
            raise MCPError(code=types.INTERNAL_ERROR, message=exc.message, data=data) from exc
        except Exception as exc:
            logger.exception("reading %s failed inside the service", uri)
            data = {"uri": uri, "exception": type(exc).__name__}
            if log_path is not None:
                data["log"] = str(log_path)
            raise MCPError(code=types.INTERNAL_ERROR, message="Internal error", data=data) from exc
        contents = types.TextResourceContents(uri=uri, text=content.text, mime_type=content.mime_type)
        return types.ReadResourceResult(contents=[contents], ttl_ms=content.ttl_ms, cache_scope=RESOURCE_CACHE_SCOPE)

    async def list_prompts(ctx: Context, params: types.PaginatedRequestParams | None) -> types.ListPromptsResult:
        return types.ListPromptsResult(prompts=build_prompts())

    async def get_prompt(ctx: Context, params: types.GetPromptRequestParams) -> types.GetPromptResult:
        spec = prompt_specs.get(params.name)
        if spec is None:
            raise MCPError(code=types.INVALID_PARAMS, message=f"Unknown prompt: {params.name}")
        given = dict(params.arguments or {})
        unknown = sorted(set(given) - set(spec.arguments))
        if unknown:
            raise MCPError(code=types.INVALID_PARAMS, message=f"Unknown argument(s): {', '.join(unknown)}")
        text = PROMPT_TEXTS[spec.name]
        body = text.template
        for argument in spec.arguments:
            value = given.get(argument, "").strip()
            if not value:
                raise MCPError(code=types.INVALID_PARAMS, message=f"Missing required argument: {argument}")
            body = body.replace("{" + argument + "}", value)
        message = types.PromptMessage(role="user", content=types.TextContent(type="text", text=body))
        return types.GetPromptResult(description=text.description, messages=[message])

    list_hint = CacheHint(ttl_ms=LIST_CACHE_TTL_MS, scope=RESOURCE_CACHE_SCOPE)
    server: Server[Any] = Server(
        SERVER_NAME,
        version=narration.__version__,
        title=SERVER_TITLE,
        instructions=SERVER_INSTRUCTIONS,
        cache_hints={
            "tools/list": list_hint,
            "prompts/list": list_hint,
            "resources/list": list_hint,
            "resources/templates/list": list_hint,
            "server/discover": list_hint,
        },
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        on_list_resources=list_resources,
        on_list_resource_templates=list_resource_templates,
        on_read_resource=read_resource,
        on_subscriptions_listen=ListenHandler(bus),
        on_list_prompts=list_prompts,
        on_get_prompt=get_prompt,
    )
    # The SDK's default OpenTelemetry middleware is a no-op without an exporter; the front-end logs itself.
    server.middleware.clear()
    return FrontEnd(server=server, bus=bus)
