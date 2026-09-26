"""The MCP SDK behaviours that ADR 0001 builds the front-end on (spike j; design sections 5 and 14).

These tests drive the installed SDK's low-level ``Server`` through its real dual-era JSON-RPC loop over
in-memory streams, with raw JSON-RPC messages. If an SDK upgrade breaks one, re-run spike (j)
(``spikes/j-mcp-sdk``) and revisit the ADR before changing the test.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import anyio
import mcp_types as types
from mcp.server import Server, ServerRequestContext
from mcp.shared.exceptions import MCPError
from mcp.shared.message import SessionMessage
from mcp_types.jsonrpc import jsonrpc_message_adapter

MODERN_META = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientInfo": {"name": "test", "version": "0"},
    "io.modelcontextprotocol/clientCapabilities": {},
}
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}},
}
SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text"],
    "properties": {"text": {"type": "string", "minLength": 1}},
}
TOOL = types.Tool(name="echo", input_schema=SCHEMA, output_schema={"type": "object", "properties": {}})

Exchange = Callable[[dict[str, Any]], Awaitable[dict[str, Any] | None]]


def build_server(seen: list[dict[str, Any]]) -> Server[Any]:
    """A server whose ``echo`` tool records the arguments it receives and answers with an ``isError`` result."""

    async def list_tools(
        ctx: ServerRequestContext[Any, Any], params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[TOOL])

    async def call_tool(
        ctx: ServerRequestContext[Any, Any], params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        if params.name == "boom":
            raise RuntimeError("uncaught")
        if params.name != TOOL.name:
            raise MCPError(code=types.INVALID_PARAMS, message=f"Unknown tool: {params.name}")
        seen.append(dict(params.arguments or {}))
        error = {"code": "INVALID_ARGUMENT", "message": "m", "retryable": False, "field": "instruct"}
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="copy")], structured_content={"error": error}, is_error=True
        )

    server: Server[Any] = Server("test", on_list_tools=list_tools, on_call_tool=call_tool)
    server.middleware.clear()
    return server


@asynccontextmanager
async def connected(server: Server[Any]) -> AsyncIterator[Exchange]:
    """Run ``server`` over memory streams; yield a function that sends one message and returns its answer."""
    to_server, server_reads = anyio.create_memory_object_stream[SessionMessage | Exception](16)
    server_writes, from_server = anyio.create_memory_object_stream[SessionMessage](16)

    async def exchange(message: dict[str, Any]) -> dict[str, Any] | None:
        await to_server.send(SessionMessage(message=jsonrpc_message_adapter.validate_python(message)))
        if "id" not in message:
            return None
        with anyio.fail_after(5):
            while True:
                reply = (await from_server.receive()).message.model_dump(by_alias=True, mode="json", exclude_none=True)
                if reply.get("id") == message["id"]:
                    return reply

    async with anyio.create_task_group() as tg:
        tg.start_soon(server.run, server_reads, server_writes, server.create_initialization_options())
        yield exchange
        await to_server.aclose()


def run_legacy(messages: list[dict[str, Any]], seen: list[dict[str, Any]] | None = None) -> list[dict[str, Any] | None]:
    """Open a 2025-11-25 session with ``initialize``, then send ``messages``; return the answers."""

    async def main() -> list[dict[str, Any] | None]:
        async with connected(build_server(seen if seen is not None else [])) as exchange:
            answers = [await exchange(INITIALIZE)]
            await exchange({"jsonrpc": "2.0", "method": "notifications/initialized"})
            answers += [await exchange(m) for m in messages]
            return answers

    return anyio.run(main)


def run_modern(messages: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    """Open a 2026-07-28 connection (its first request carries the envelope) and send ``messages``."""

    async def main() -> list[dict[str, Any] | None]:
        async with connected(build_server([])) as exchange:
            return [await exchange(m) for m in messages]

    return anyio.run(main)


def call(request_id: int, name: str, arguments: dict[str, Any], meta: dict[str, Any] | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {"name": name, "arguments": arguments}
    if meta is not None:
        params["_meta"] = meta
    return {"jsonrpc": "2.0", "id": request_id, "method": "tools/call", "params": params}


def test_legacy_initialize_is_still_answered_s5() -> None:
    (init,) = run_legacy([])
    assert init is not None
    assert init["result"]["protocolVersion"] == "2025-11-25"


def test_server_discover_offers_2026_07_28_and_refuses_a_late_initialize_s5() -> None:
    discover, late_init = run_modern(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "server/discover", "params": {"_meta": MODERN_META}},
            {**INITIALIZE, "id": 2},
        ]
    )
    assert discover is not None and discover["result"]["supportedVersions"] == ["2026-07-28"]
    assert late_init is not None and late_init["error"]["code"] == types.UNSUPPORTED_PROTOCOL_VERSION


def test_tools_list_publishes_our_schema_verbatim_s5() -> None:
    _, listed = run_legacy([{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}])
    assert listed is not None
    (tool,) = listed["result"]["tools"]
    assert tool["inputSchema"] == SCHEMA


def test_the_sdk_does_not_validate_arguments_so_the_handler_can_s14() -> None:
    seen: list[dict[str, Any]] = []
    run_legacy([call(2, "echo", {"text": "", "instruct": "calm"})], seen)
    assert seen == [{"text": "", "instruct": "calm"}]


def test_an_is_error_result_reaches_the_wire_intact_in_both_eras_s14() -> None:
    _, legacy = run_legacy([call(2, "echo", {"instruct": "calm"})])
    (modern,) = run_modern([call(1, "echo", {"instruct": "calm"}, MODERN_META)])
    for answer in (legacy, modern):
        assert answer is not None
        result = answer["result"]
        assert result["isError"] is True
        assert result["structuredContent"]["error"]["field"] == "instruct"
        assert result["content"] == [{"type": "text", "text": "copy"}]


def test_an_unknown_tool_is_json_rpc_invalid_params_s14() -> None:
    _, legacy = run_legacy([call(2, "no_such_tool", {})])
    (modern,) = run_modern([call(1, "no_such_tool", {}, MODERN_META)])
    for answer in (legacy, modern):
        assert answer is not None and answer["error"]["code"] == types.INVALID_PARAMS


def test_an_uncaught_handler_exception_is_a_json_rpc_error_not_a_tool_result_s14() -> None:
    """Why the front-end catches every exception itself: the SDK's mapping differs by era."""
    _, legacy = run_legacy([call(2, "boom", {})])
    (modern,) = run_modern([call(1, "boom", {}, MODERN_META)])
    assert legacy is not None and "error" in legacy
    assert modern is not None and modern["error"]["code"] == types.INTERNAL_ERROR
