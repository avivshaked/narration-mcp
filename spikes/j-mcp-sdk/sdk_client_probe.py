"""Spike (j): the SDK's own ``Client`` against the spike server, over stdio, with the wire recorded.

Claude Code cannot be run as a client here, so this stands in for "a current client": the ``mcp`` 2.2.0
``Client`` in its default ``mode="auto"`` (probe ``server/discover``, fall back to ``initialize``), and
again in ``mode="legacy"`` (the ``initialize`` handshake a pre-2026 client sends). A tap between the
client and the stdio transport records every message both ways.

Outputs: ``results/sdk-client-<mode>.jsonl`` (the wire) and ``results/sdk-client-checks.json``. The exit
status is 1 if any check failed.

Run: ``uv run python spikes/j-mcp-sdk/sdk_client_probe.py``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import anyio
from mcp import Client, MCPError, StdioServerParameters, stdio_client
from mcp.client._transport import Transport, TransportStreams
from mcp.shared.message import SessionMessage
from mcp_types import Implementation

HERE = Path(__file__).resolve().parent
SERVER = HERE / "spike_server.py"
RESULTS = HERE / "results"
STDERR_SCRATCH = RESULTS / "_server-stderr.tmp"

GOOD_VOICE = {"path": "C:/voices/narrator.wav", "sha256": "5b" * 32, "transcript": "Good bread asks for patience."}
GOOD_LINE = {"voice": GOOD_VOICE, "text": "Before dawn, the reef belongs to the Ossavine shrimp."}

CHECKS: list[dict[str, Any]] = []


def check(mode: str, name: str, ok: bool, observed: Any) -> None:
    CHECKS.append({"mode": mode, "check": name, "ok": bool(ok), "observed": observed})


def dump(item: SessionMessage | Exception) -> dict[str, Any]:
    if isinstance(item, Exception):
        return {"exception": type(item).__name__}
    return item.message.model_dump(by_alias=True, mode="json", exclude_none=True)


@asynccontextmanager
async def tapped(inner: Transport, log: list[dict[str, Any]]) -> AsyncIterator[TransportStreams]:
    """Relay a transport's streams through two pumps that record every message."""
    async with inner as (read, write):
        up_send, up_recv = anyio.create_memory_object_stream[SessionMessage | Exception](16)
        down_send, down_recv = anyio.create_memory_object_stream[SessionMessage](16)

        async def pump_up() -> None:
            async with up_send:
                async for item in read:
                    log.append({"dir": "<-", "msg": dump(item)})
                    await up_send.send(item)

        async def pump_down() -> None:
            async with down_recv:
                async for item in down_recv:
                    log.append({"dir": "->", "msg": dump(item)})
                    await write.send(item)

        async with anyio.create_task_group() as tg:
            tg.start_soon(pump_up)
            tg.start_soon(pump_down)
            try:
                yield up_recv, down_send
            finally:
                tg.cancel_scope.cancel()


def sent(log: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return [e["msg"] for e in log if e["dir"] == "->" and e["msg"].get("method") == method]


async def probe(mode: str) -> None:
    log: list[dict[str, Any]] = []
    with STDERR_SCRATCH.open("w", encoding="utf-8") as errlog:
        params = StdioServerParameters(command=sys.executable, args=[str(SERVER)], env={"PYTHONUTF8": "1"})
        transport = cast(Transport, tapped(cast(Transport, stdio_client(params, errlog=errlog)), log))
        client = Client(
            transport,
            mode=mode,
            client_info=Implementation(name="sdk-client-probe", version="0"),
            read_timeout_seconds=30,
        )
        with anyio.fail_after(90):
            async with client:
                first = log[0]["msg"] if log else {}
                check(
                    mode,
                    "what the SDK client sends first (method and _meta)",
                    first.get("method") == ("server/discover" if mode == "auto" else "initialize"),
                    {"method": first.get("method"), "params": first.get("params")},
                )
                check(mode, "negotiated protocol version", True, client.protocol_version)

                tools = await client.list_tools()
                check(mode, "list_tools", len(tools.tools) == 3, [t.name for t in tools.tools])

                good = await client.call_tool("check_line", GOOD_LINE)
                check(
                    mode,
                    "success result passes the client's own outputSchema validation",
                    not good.is_error and isinstance(good.structured_content, dict),
                    {"is_error": good.is_error, "content_types": [c.type for c in good.content]},
                )

                bad = await client.call_tool("check_line", {**GOOD_LINE, "instruct": "calm"})
                error = (bad.structured_content or {}).get("error", {})
                check(
                    mode,
                    "unknown field comes back as a tool result (isError, INVALID_ARGUMENT, field instruct)",
                    bad.is_error and error.get("field") == "instruct",
                    error,
                )

                try:
                    await client.call_tool("no_such_tool", {})
                    unknown: Any = "no error raised"
                except MCPError as exc:
                    unknown = {"code": exc.error.code, "message": exc.error.message}
                check(mode, "unknown tool raises MCPError -32602 in the client", unknown != "no error raised", unknown)

                seen: list[tuple[float, float | None, str | None]] = []

                async def on_progress(progress: float, total: float | None, message: str | None) -> None:
                    seen.append((progress, total, message))

                done = await client.call_tool(
                    "wait_job", {"job_id": "j1", "wait_s": 1.0}, progress_callback=on_progress
                )
                token = sent(log, "tools/call")[-1]["params"]["_meta"].get("progressToken")
                check(
                    mode,
                    "progress_callback: the SDK sends a progressToken (the request id) and gets our notifications",
                    len(seen) >= 2 and not done.is_error,
                    {"progressToken": token, "calls": len(seen), "last": seen[-1] if seen else None},
                )

                with anyio.move_on_after(0.6):
                    await client.call_tool("wait_job", {"job_id": "j2", "wait_s": 5.0})
                await anyio.sleep(0.5)  # the courtesy cancel goes out asynchronously, through the tap
                cancels = sent(log, "notifications/cancelled")
                check(
                    mode,
                    "abandoning a call makes the SDK send notifications/cancelled",
                    len(cancels) == 1,
                    cancels,
                )

                await client.read_resource("spike://jobs/j9")
                await client.read_resource("spike://jobs/j9")
                reads = len(sent(log, "resources/read"))
                check(
                    mode,
                    "resources/read twice: sent over the wire (modern: once, ttlMs honoured by the client cache)",
                    reads == (1 if mode == "auto" else 2),
                    {"reads_on_the_wire": reads},
                )

                prompt = await client.get_prompt("narrate_script", {"voice_path": "C:/voices/a.wav"})
                check(mode, "get_prompt", bool(prompt.messages), prompt.messages[0].content.type)
    stderr_notes = [
        line.strip()
        for line in STDERR_SCRATCH.read_text(encoding="utf-8").splitlines()
        if line.startswith("spike_server:")
    ]
    STDERR_SCRATCH.unlink()
    check(mode, "the server saw the cancellation", bool(stderr_notes), stderr_notes)
    path = RESULTS / f"sdk-client-{mode}.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for entry in log:
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")


async def run() -> None:
    await probe("auto")
    await probe("legacy")


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    anyio.run(run)
    failed = [c for c in CHECKS if not c["ok"]]
    summary = {"checks": CHECKS, "passed": len(CHECKS) - len(failed), "failed": len(failed)}
    out = RESULTS / "sdk-client-checks.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    for c in CHECKS:
        print(f"{'ok  ' if c['ok'] else 'FAIL'} [{c['mode']}] {c['check']}")
    print(f"{summary['passed']} passed, {summary['failed']} failed; details in {out.name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
