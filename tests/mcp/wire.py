"""Raw JSON-RPC to a front-end over in-memory streams, in either protocol era (design section 5).

``open_wire(server, era)`` runs the server's real dual-era loop and yields a ``Wire``. The legacy era opens
with the ``initialize`` handshake (protocol 2025-11-25); the modern era sends the 2026-07-28 envelope in
every request's ``_meta``. A reader task files every reply by id and keeps every notification, so a test
can see progress notifications and check that a cancelled request is never answered.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.server import Server
from mcp.shared.message import SessionMessage
from mcp_types.jsonrpc import jsonrpc_message_adapter

Era = Literal["legacy", "modern"]
ERAS: tuple[Era, ...] = ("legacy", "modern")

MODERN_META: dict[str, Any] = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientInfo": {"name": "test", "version": "0"},
    "io.modelcontextprotocol/clientCapabilities": {},
}
LEGACY_VERSION = "2025-11-25"
REPLY_TIMEOUT_S = 10.0


class Wire:
    """One client connection: send requests and notifications, read replies and notifications."""

    def __init__(self, era: Era, to_server: MemoryObjectSendStream[SessionMessage | Exception]) -> None:
        self.era: Era = era
        self._to_server = to_server
        self._next_id = 100
        self.replies: dict[int | str, dict[str, Any]] = {}
        self._waiters: dict[int | str, anyio.Event] = {}
        self.notifications: list[dict[str, Any]] = []
        self.init_result: dict[str, Any] | None = None
        """The legacy era's ``initialize`` result."""

    async def _send(self, message: dict[str, Any]) -> None:
        await self._to_server.send(SessionMessage(message=jsonrpc_message_adapter.validate_python(message)))

    async def read_loop(self, from_server: MemoryObjectReceiveStream[SessionMessage]) -> None:
        """File every message from the server: replies by id, everything else as a notification."""
        async for item in from_server:
            data = item.message.model_dump(by_alias=True, mode="json", exclude_none=True)
            if "id" in data and ("result" in data or "error" in data):
                self.replies[data["id"]] = data
                if (waiter := self._waiters.pop(data["id"], None)) is not None:
                    waiter.set()
            else:
                self.notifications.append(data)

    def _params(self, params: dict[str, Any] | None, meta: dict[str, Any] | None) -> dict[str, Any]:
        out = dict(params or {})
        envelope = {**(MODERN_META if self.era == "modern" else {}), **(meta or {})}
        if envelope:
            out["_meta"] = envelope
        return out

    async def send(
        self, method: str, params: dict[str, Any] | None = None, *, meta: dict[str, Any] | None = None
    ) -> int:
        """Send a request without waiting; return its id."""
        request_id = self._next_id
        self._next_id += 1
        self._waiters[request_id] = anyio.Event()
        message = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": self._params(params, meta)}
        await self._send(message)
        return request_id

    async def reply(self, request_id: int) -> dict[str, Any]:
        """Wait for the reply to ``request_id``."""
        if request_id not in self.replies:
            with anyio.fail_after(REPLY_TIMEOUT_S):
                await self._waiters[request_id].wait()
        return self.replies[request_id]

    async def request(
        self, method: str, params: dict[str, Any] | None = None, *, meta: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a request and return its whole reply (``result`` or ``error``)."""
        return await self.reply(await self.send(method, params, meta=meta))

    async def result(
        self, method: str, params: dict[str, Any] | None = None, *, meta: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a request and return its ``result``; fail if it was a JSON-RPC error."""
        reply = await self.request(method, params, meta=meta)
        assert "result" in reply, reply
        return reply["result"]

    async def call(
        self, name: str, arguments: dict[str, Any] | None = None, *, meta: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """``tools/call`` and return its whole reply."""
        return await self.request("tools/call", {"name": name, "arguments": arguments or {}}, meta=meta)

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        await self._send(message)

    async def initialize(self) -> dict[str, Any]:
        """The legacy handshake."""
        params = {"protocolVersion": LEGACY_VERSION, "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}
        reply = await self.request("initialize", params)
        await self.notify("notifications/initialized")
        return reply


@asynccontextmanager
async def open_wire(server: Server[Any], era: Era) -> AsyncIterator[Wire]:
    """Run ``server`` over memory streams in ``era``; stop it when the block ends."""
    to_server, server_reads = anyio.create_memory_object_stream[SessionMessage | Exception](64)
    server_writes, from_server = anyio.create_memory_object_stream[SessionMessage](64)
    wire = Wire(era, to_server)
    async with anyio.create_task_group() as tg:
        tg.start_soon(server.run, server_reads, server_writes, server.create_initialization_options())
        tg.start_soon(wire.read_loop, from_server)
        try:
            if era == "legacy":
                reply = await wire.initialize()
                assert reply["result"]["protocolVersion"] == LEGACY_VERSION, reply
                wire.init_result = reply["result"]
            yield wire
        finally:
            await to_server.aclose()
            tg.cancel_scope.cancel()
