# ADR 0001: The MCP front-end is built on the low-level `Server` of `mcp` 2.2.0

- Status: proposed (WP17, spike j), for the lead to accept
- Date: 2026-09-26
- Design sections: 5 (protocol basis), 7 (MCP surface), 14 (error model)
- Evidence: [spikes/j-mcp-sdk](../../spikes/j-mcp-sdk/README.md)

## Context

Design section 5 assumes (BELIEVE) that the Python MCP SDK's v2 line implements revision 2026-07-28 and
still answers a legacy `initialize`. Section 14 puts argument validation inside our own handler, so
that a bad argument is a tool result with `isError: true`, a structured Error and a hint, and never a
JSON-RPC error. An unknown tool must still be JSON-RPC -32602. Published schemas must contain no `$ref`,
and every `outputSchema` root must be `"type": "object"`. The server dependency is declared as
`mcp>=2.2,<3`, and `uv.lock` holds 2.2.0.

## Decision

1. **Use `mcp` 2.2.0, pinned exactly** (`mcp==2.2.0`, which requires `mcp-types==2.2.0`; both are MIT).
2. **Build on the low-level `mcp.server.Server`, not on `MCPServer`.** Register constructor handlers:
   - `on_list_tools` and `on_call_tool`;
   - `on_list_resources`, `on_list_resource_templates` and `on_read_resource`;
   - `on_list_prompts` and `on_get_prompt`;
   - `on_subscriptions_listen=ListenHandler(bus)`.
3. **Publish our own schemas.** Each `types.Tool` carries the hand-written, fully dereferenced
   `input_schema` and `output_schema` from `narration.contracts`, in the fixed order of section 7.1.
4. **`on_call_tool` owns the whole call**:
   - an unknown name raises `MCPError(INVALID_PARAMS)`, which is -32602;
   - the arguments are validated with `jsonschema` (Draft 2020-12, one compiled validator per tool)
     against the same schema we publish. A failure returns `isError: true`,
     `structuredContent: {"error": Error}` with `field` and `hint`, and a text copy of it;
   - every other exception is caught. None may reach the SDK, whose mapping is wrong for us.
5. **Results.**
   - A success carries `structuredContent` that matches the tool's `outputSchema`, a JSON text copy of
     it, and a `resource_link` (`file:///`) for each audio file.
   - A resource read sets `ttl_ms` and `cache_scope="private"` explicitly.
6. **Serve stdio with `Server.run`**, the dual-era loop: 2026-07-28 for a client whose first request
   carries the `_meta` envelope, and the `initialize` handshake (up to 2025-11-25) for any other client.
7. **Clear `server.middleware`.** This drops the SDK's default OpenTelemetry middleware, a no-op without an
   exporter, so nothing implicit wraps a call.

## Evidence

**KNOW** means shown by a script, with its output saved. **KNOW (source)** means read in the installed
package's source and not run. The file is named in each case.

| Claim | Label | Where |
|---|---|---|
| `mcp` 2.2.0 is the v2 line. It knows 2024-11-05 to 2025-11-25 (handshake) and 2026-07-28 (per-request envelope). `MCPServer` exists | KNOW | `spikes/j-mcp-sdk/results/sdk-facts.json` |
| One stdio server answers both eras. The first request decides: legacy `initialize` (2025-11-25; a 2026-07-28 offer is countered with 2025-11-25) or `server/discover` with the envelope. The other era's requests are refused (-32600 / -32022) | KNOW | `results/wire-checks.json`; `wire-legacy.jsonl`, `wire-modern.jsonl`, `wire-legacy-offer-2026.jsonl` |
| `server/discover` is derived from the registered handlers and carries `supportedVersions`, capabilities, `instructions`, `ttlMs`, `cacheScope` and `serverInfo` | KNOW | `wire-modern.jsonl` |
| The low-level `Server` publishes our `inputSchema`/`outputSchema` verbatim (no `$ref`; `additionalProperties: false` kept) | KNOW | `wire-checks.json`; `tests/mcp/test_sdk_contract.py` |
| The low-level `Server` does not validate tool arguments; only `arguments` not being an object is -32602 | KNOW | `wire-checks.json`; `tests/mcp/test_sdk_contract.py` |
| In-handler validation returns `isError` + `structuredContent.error` (`INVALID_ARGUMENT`, `field: "instruct"`, hint) + a text copy, in both eras | KNOW | `wire-legacy.jsonl`, `wire-modern.jsonl`, `sdk-client-checks.json` |
| An unknown tool is -32602 when the handler raises `MCPError(INVALID_PARAMS)` | KNOW | `wire-checks.json`; `tests/mcp/test_sdk_contract.py` |
| An uncaught exception is `code: 0` with the exception's text on a legacy connection, and -32603 on a modern one | KNOW | `wire-checks.json` (`raise_demo`); `mcp/shared/jsonrpc_dispatcher.py` |
| Progress (`ctx.session.report_progress`, a no-op without a `progressToken`) and cancellation (the handler's scope is cancelled; the request is never answered) work in both eras | KNOW | `wire-checks.json`, `sdk-client-checks.json` |
| `subscriptions/listen` with `ListenHandler` sends an acknowledgement, then `notifications/resources/updated` tagged with the listen id | KNOW | `wire-modern.jsonl` |
| `ttlMs`/`cacheScope` reach 2026-07-28 clients only, and the SDK client honours them | KNOW | `wire-checks.json`, `sdk-client-checks.json` |
| The SDK client validates a success result against `outputSchema` (and raises if `structuredContent` is missing), never an `isError` one | KNOW (source) | `mcp/client/session.py`, `validate_tool_result`; the success path is exercised in `sdk-client-checks.json` |
| `MCPServer` publishes `$defs`/`$ref`, silently drops an unknown field (`instruct` succeeded), reports a wrong type as text only, and answers an unknown tool with `isError` | KNOW | `results/mcpserver-checks.json` |
| The SDK's middleware API is provisional ("may change in a 2.x minor release") | KNOW (source) | `mcp/server/lowlevel/server.py`, `mcp/server/context.py` |
| What Claude Code sends first (`initialize` or an enveloped request) | BELIEVE: either; unverified | Claude Code could not be run as a client here. `spike_server.py` logs inbound messages when `SPIKE_INBOUND_LOG` is set, for a later check |

## Consequences

- WP17 phase 2 follows `spikes/j-mcp-sdk/spike_server.py`: a tool registry of name, `Tool`, compiled
  validator and handler; one `on_call_tool` that finds the tool, validates and dispatches; one place that
  turns an Error into a tool result; and a catch-all.
- `tests/mcp/test_sdk_contract.py` runs in the default suite. An SDK upgrade that changes a behaviour this
  ADR relies on fails there first.
- **Legacy-era clients get no subscriptions.** `resources/subscribe` is not registered, so they poll
  `get_job` with `wait_s`, which the design provides anyway. Adding `on_subscribe_resource` together with
  `ServerSession.send_resource_updated` later is possible, but not planned.
- **The Tasks extension is not available** in this SDK ("tasks/* deliberately absent"). The design
  defers it anyway.
- `ttlMs`/`cacheScope` are properties of each list or read result, not of a resource. Section 7.7's table
  becomes the values that `on_read_resource` sets per URI.

## Alternatives considered

- **`MCPServer` (the high-level API).** Rejected: its schemas come from pydantic (with `$ref`), it ignores
  unknown top-level fields, its argument errors are text only, and an unknown tool is a tool result. Each
  of these could only be worked around through its internals (`Tool.fn_metadata`, `ArgModelBase`).
- **`mcp` 1.x** (the `v1.x` branch). Rejected: it has the handshake only, with no 2026-07-28,
  `server/discover` or per-request envelope.
- **A hand-written JSON-RPC layer.** Rejected: it would duplicate a maintained, MIT-licensed SDK for no
  gain, and lose the era negotiation, envelope checks, result sieving and listen streams that the SDK gets
  right (the checks above).
- **Letting the SDK map exceptions.** Rejected: legacy connections get `code: 0` and the exception's text.

## What would reverse this

- A future `mcp` release in which the low-level `Server` validates or rewrites tool schemas or arguments,
  drops `Server.run`'s dual-era loop (the legacy `initialize`) while clients still need it, or changes
  the constructor-handler API. The contract test would show it; then re-run the spike.
- An `MCPServer` that can publish a given schema verbatim, hand validation to the tool, and answer an
  unknown tool with -32602. It would then save code and could replace the low-level base.
- Evidence that Claude Code (or another client we must serve) needs something this path lacks. The
  inbound log of `spike_server.py` is the way to find out.
- A spec revision that changes where argument errors belong.

## Open points for the lead

- **Pin `mcp==2.2.0`** in `pyproject.toml`: a dependency request in `status/WP17.md` (the lead owns
  `pyproject.toml`).
- Design section 14 lists both JSON-RPC -32603 ("an internal failure") and the tool-error code
  `INTERNAL` ("a bug; the log path is included"). The proposal: a failure inside a tool call is an
  `isError` result with `INTERNAL`, so that the model sees it, and -32603 is kept for failures outside a
  tool call. WP01 or the lead decides.
