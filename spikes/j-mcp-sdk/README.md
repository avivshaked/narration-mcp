# Spike (j): the MCP Python SDK against design sections 5, 7 and 14

Plan: WP17, phase 1. Decision: [ADR 0001](../../docs/decisions/0001-mcp-sdk.md). Run on 2026-09-26 with
`mcp` 2.2.0 and `mcp-types` 2.2.0 (the versions in `uv.lock`), Python 3.12, Windows 11, stdio.

## Questions

1. Which Python MCP SDK version implements spec revision 2026-07-28? Is `mcp` 2.2.0 the v2 line, with
   `MCPServer`, stateless operation and `server/discover`? Does it still answer a legacy `initialize`?
2. What does a current client send?
3. Can argument validation happen inside our handler, so that a bad argument is a tool result
   (`isError`) and not a JSON-RPC error, with hand-written, dereferenced schemas in `tools/list`?
4. How do progress, cancellation, resource templates, prompts, `structuredContent` and `resource_link`
   work? Is the low-level or the high-level server API the better base?
5. What in design section 5 does the SDK not support?

## Files

| File | What it is |
|---|---|
| `spike_server.py` | A throwaway server on the low-level `Server`: three tools with hand-written schemas, validation inside the handler (jsonschema, Draft 2020-12), progress, cancellation, a resource template with `ttlMs`/`cacheScope`, a prompt, `subscriptions/listen`. With `SPIKE_INBOUND_LOG=<file>` it appends every inbound message, `_meta` included, to that file |
| `wire_probe.py` | A raw stdio client: starts the server as a subprocess and writes JSON-RPC lines itself, in both protocol eras. 48 checks |
| `sdk_client_probe.py` | The SDK's own `Client` over stdio, in `mode="auto"` (its default) and `mode="legacy"`, with a tap that records the wire. 22 checks |
| `mcpserver_probe.py` | The high-level `MCPServer`, in-process, on the same cases: schemas, an unknown field, a wrong type, an unknown tool |
| `sdk_facts.py` | The SDK's version tables and per-version method surface, read from the package |
| `results/` | What the scripts wrote: `wire-<session>.jsonl` and `sdk-client-<mode>.jsonl` (every JSON-RPC message, `->` sent, `<-` received), `*-checks.json`, `sdk-facts.json` |

Reproduce (each takes a few seconds; no GPU, no model, no network):

```
uv run python spikes/j-mcp-sdk/sdk_facts.py
uv run python spikes/j-mcp-sdk/wire_probe.py
uv run python spikes/j-mcp-sdk/sdk_client_probe.py
uv run python spikes/j-mcp-sdk/mcpserver_probe.py
```

The progress counts in the `.jsonl` files depend on timing, so a rerun can differ there. The checks do
not. `tests/mcp/test_sdk_contract.py` pins the behaviours the ADR depends on, in the default suite.

## Findings

Labels as in AGENTS.md section 8. **KNOW** here means shown by a script in this folder, with its output
in `results/`. **KNOW (source)** means read in the installed package's source, not run; the file is named.

### 1. The SDK and the protocol revisions

- **KNOW** `mcp` 2.2.0 is the v2 line and implements 2026-07-28. Its README says so, and
  `sdk-facts.json` lists the revisions it knows: `2024-11-05`, `2025-03-26`, `2025-06-18` and `2025-11-25`
  through the `initialize` handshake, and `2026-07-28` through the per-request `_meta` envelope.
  `MCPServer` is `mcp.server.mcpserver.MCPServer` (v1's `FastMCP`, renamed).
- **KNOW** One stdio server serves both eras. `Server.run` drives `serve_dual_era_loop`: the client's
  first request decides the era of the connection, once. A request whose `_meta` carries
  `io.modelcontextprotocol/protocolVersion` opens a 2026-07-28 connection, and anything else opens a
  legacy one (`wire-checks.json`, sessions `legacy`, `modern`, `modern-first-no-envelope`).
- **KNOW** Legacy: `initialize` at 2025-11-25 is answered. An `initialize` that proposes 2026-07-28 gets
  the counter-offer 2025-11-25. A 2026 envelope on a legacy connection gets -32600.
- **KNOW** Modern: `server/discover` is answered by a default handler that the SDK derives from the
  registered handlers. It returns `supportedVersions: ["2026-07-28"]`, the capabilities, `instructions`,
  `resultType`, `ttlMs`, `cacheScope`, and a `serverInfo` stamp in `_meta`. On a modern connection:
  - a later `initialize` gets -32022, naming the supported versions;
  - an envelope without `clientCapabilities` gets -32602;
  - an unknown version gets -32022 with `{supported, requested}`;
  - `ping` gets -32601, because 2026-07-28 removes it.
- **KNOW (source)** The modern era keeps no session state: each request gets a fresh `Connection` built
  from its envelope (`mcp/server/runner.py`, `_serve_modern_stream`). Server-initiated requests are
  refused there (`NoBackChannelError`). Notifications (progress, listen streams) still flow.

### 2. What a current client sends

- **KNOW** The SDK's `Client` defaults to `mode="auto"`. It first sends `server/discover` with the
  envelope `{protocolVersion: "2026-07-28", clientInfo, clientCapabilities: {}}`, then stamps that
  envelope on every request (`sdk-client-auto.jsonl`). It falls back to `initialize` on any error except
  -32022 with a modern-only, disjoint version list (`mcp/client/_probe.py`). In `mode="legacy"` it sends
  `initialize` with `protocolVersion: "2025-11-25"`, `capabilities: {}`, `clientInfo` and an empty
  `_meta`, then `notifications/initialized` (`sdk-client-legacy.jsonl`).
- **KNOW** The SDK client uses the request id as the `progressToken` when a progress callback is given.
  When a call is abandoned it sends `notifications/cancelled` with reason `"caller cancelled"`. It
  honours `ttlMs`: two reads of the same resource within `ttlMs` cost one request on a modern connection,
  and two on a legacy one.
- **KNOW (source)** The SDK client checks a successful tool result against the tool's `outputSchema`.
  It raises if `structuredContent` is missing or does not match. It never checks an `isError` result
  (`mcp/client/session.py`, `validate_tool_result`).
- **BELIEVE** Claude Code could not be run as a client here, and it is most likely not built on this
  Python SDK, so what it sends is unverified. A current client either opens with `initialize` or with an
  enveloped request such as `server/discover`, and the server answers both, so the decision in the ADR
  does not depend on which.
  To see what it actually sends, register `spike_server.py` as a stdio server with
  `SPIKE_INBOUND_LOG` set and read the log. That is a change to a Claude Code configuration, so it is
  for the lead or the owner to do.

### 3. Validation inside the handler

- **KNOW** The low-level `Server` publishes a `Tool`'s `inputSchema` and `outputSchema` exactly as
  written: `additionalProperties: false` is kept, the nested Voice fragment is inlined, and there is no
  `$ref`. Every `outputSchema` root is `"type": "object"`.
- **KNOW** The low-level `Server` does not validate tool arguments. The only check is the
  `CallToolRequestParams` shape: `arguments` that are not an object get -32602 with empty `data`. So the
  handler validates with `jsonschema`, and a bad argument comes back as a tool result:

  ```json
  {"content": [{"type": "text", "text": "{\"error\": {...}}"}],
   "isError": true,
   "structuredContent": {"error": {"code": "INVALID_ARGUMENT", "message": "unknown field 'instruct'",
     "retryable": false, "field": "instruct", "hint": "instruct is not an input of this service: ...",
     "details": {"schema_rule": "additionalProperties"}}}}
  ```

  This holds in both eras (`wire-legacy.jsonl`, `wire-modern.jsonl`). A wrong type gives `field: "takes"`.
- **KNOW** An unknown tool is JSON-RPC -32602 when the handler raises
  `MCPError(code=INVALID_PARAMS, ...)`. The SDK does not decide this: the handler must.
- **KNOW** An exception the handler does not catch becomes a JSON-RPC error whose code depends on the era.
  On a legacy connection it is `code: 0` with the exception's text as the message. On a modern
  connection it is -32603 `"Internal server error"`. So the front-end must catch every exception itself
  (`mcp/shared/jsonrpc_dispatcher.py` marks the legacy `code=0` as a v1-compatibility TODO).
- **KNOW (source)** The SDK's server validates results against the per-version wire schema before
  sending. A result that fails, such as a 2025-11-25 `outputSchema` whose root is not an object (the
  2025-11-25 wire model requires `type: "object"` there), becomes -32603 (`mcp/server/runner.py`,
  `_serialize`). The design's rule that every `outputSchema` root is an object
  keeps legacy clients working.

### 4. Progress, cancellation, resources, prompts, results; which API

- **KNOW** Progress: a handler calls `ctx.session.report_progress(progress, total, message)`. This is a
  no-op unless the request carried `_meta.progressToken`, and otherwise sends `notifications/progress`
  with that token. Both eras.
- **KNOW** Cancellation: `notifications/cancelled` for an in-flight request cancels the handler's anyio
  scope. The dispatcher's default `peer_cancel_mode` is `"interrupt"`. The handler sees a cancellation
  exception, and the request is never answered. Both eras. The connection stays usable.
- **KNOW** Resources: templates are listed by `on_list_resource_templates` and reads go through
  `on_read_resource`. The SDK's `UriTemplate.match` extracts the variables. `ttl_ms` and
  `cache_scope="private"` go on each `ReadResourceResult` (and on the list results). They are sent only
  on 2026-07-28 connections: the legacy wire schema has no such fields, so they are dropped there. A
  missing resource is the handler's `MCPError(INVALID_PARAMS)`: -32602.
- **KNOW** `subscriptions/listen` (2026-07-28 only): pass `ListenHandler(bus)` as
  `on_subscriptions_listen`, and `await bus.publish(ResourceUpdated(uri=...))` when a job changes. The
  client gets `notifications/subscriptions/acknowledged`, then `notifications/resources/updated` tagged
  with the listen request's id. Cancelling the listen request ends the stream. On a legacy connection
  `resources/subscribe` is -32601 unless a handler for it is registered as well.
- **KNOW** Prompts: `on_list_prompts` and `on_get_prompt`.
- **KNOW** Results: a `CallToolResult` with `structured_content`, a `TextContent` copy, and
  `ResourceLink(type="resource_link", uri="file:///...", mime_type="audio/wav")` items. The SDK passes
  them through unchanged. On 2026-07-28 it adds `resultType` and the `serverInfo` stamp.
- **KNOW** The high-level `MCPServer` does not fit design sections 5 and 14 (`mcpserver-checks.json`):
  - it publishes pydantic's schema, with `$defs` and `$ref`;
  - it **silently drops** an unknown top-level field: `instruct` went through as a success;
  - it reports a wrong type as text only, with no `structuredContent`, and the text is pydantic's message
    with a URL in it;
  - it answers an unknown tool with an `isError` result, not -32602.

  The low-level `Server` is the better base. See the ADR.

### 5. What the SDK does not do, and what to do instead

- **The Tasks extension**: absent. The SDK's method tables say "tasks/* deliberately absent". The design
  already defers it to a later phase.
- **Subscriptions for legacy clients**: `subscriptions/listen` exists only on 2026-07-28. A legacy-era
  client could only subscribe through `resources/subscribe` plus `ServerSession.send_resource_updated`.
  Without them, legacy clients poll `get_job` (long-poll `wait_s`), which the design has anyway.
- **`ttlMs` and `cacheScope` on legacy connections**: not sent, because the legacy schema has no such
  fields. They are per result, not per resource, so the design's table of TTLs per URI becomes the
  values each `resources/read` result carries.
- **Internal errors**: the SDK's own mapping of an uncaught exception is not the design's -32603 on legacy
  connections, and it leaks the exception's text. The front-end catches everything itself.
- **Argument validation**: the SDK does none on the low-level path. That is what design section 14 wants,
  and the front-end owns it with `jsonschema` (already a declared dependency).
- Everything else in section 5 is supported: stateless operation, `server/discover`, `ttlMs` +
  `cacheScope: "private"`, `subscriptions/listen`, progress, cancellation, `structuredContent` +
  `outputSchema`, `resource_link`, annotations, and legacy `initialize`. The SDK marks Sampling, Roots
  and Logging deprecated (`MCPDeprecationWarning`), and the spike uses none of them.

### Note on the recorded probes (2026-09-26)

After the runs, the probes' voice transcript was changed in the scripts and in the recorded
`results/*.jsonl` alike. It is now the first clause of the service's new default design text (design
section 16), ending in a full stop. The spike's servers check the transcript against the schema (a string
of 1 to 600 characters) and never read it. The new value meets the same limits, so no recorded outcome
changes.
