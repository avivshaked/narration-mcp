"""Spike (j): facts about the installed MCP Python SDK, read from the package itself.

Records the installed versions and licences, the protocol revisions the SDK knows (and which of them use
the ``initialize`` handshake and which the 2026-07-28 per-request envelope), the methods a client may
send and a server may send at 2025-11-25 and at 2026-07-28, the cacheable methods, and the request
methods a bare low-level ``Server`` answers.

Output: ``results/sdk-facts.json``.

Run: ``uv run python spikes/j-mcp-sdk/sdk_facts.py``.
"""

from __future__ import annotations

import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from mcp.server import MCPServer, Server
from mcp_types import methods, version

RESULTS = Path(__file__).resolve().parent / "results"


def surface(table: Any, at: str) -> list[str]:
    return sorted(method for method, v in table if v == at)


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    dist = metadata.distribution("mcp")
    legacy, modern = version.LATEST_HANDSHAKE_VERSION, version.LATEST_MODERN_VERSION
    facts: dict[str, Any] = {
        "packages": {
            name: {"version": metadata.version(name), "license": metadata.metadata(name).get("License")}
            for name in ("mcp", "mcp-types")
        },
        "mcp requires": sorted(dist.requires or []),
        "protocol versions": {
            "known": list(version.KNOWN_PROTOCOL_VERSIONS),
            "handshake (initialize)": list(version.HANDSHAKE_PROTOCOL_VERSIONS),
            "modern (per-request _meta envelope)": list(version.MODERN_PROTOCOL_VERSIONS),
            "latest handshake": legacy,
            "latest modern": modern,
        },
        "client requests": {
            legacy: surface(methods.CLIENT_REQUESTS, legacy),
            modern: surface(methods.CLIENT_REQUESTS, modern),
        },
        "client notifications": {
            legacy: surface(methods.CLIENT_NOTIFICATIONS, legacy),
            modern: surface(methods.CLIENT_NOTIFICATIONS, modern),
        },
        "server notifications": {
            legacy: surface(methods.SERVER_NOTIFICATIONS, legacy),
            modern: surface(methods.SERVER_NOTIFICATIONS, modern),
        },
        "server requests": {
            legacy: surface(methods.SERVER_REQUESTS, legacy),
            modern: surface(methods.SERVER_REQUESTS, modern),
        },
        "cacheable methods (ttlMs, cacheScope)": sorted(methods.CACHEABLE_METHODS),
        "a bare low-level Server answers": sorted(Server("bare")._request_handlers),  # pyright: ignore[reportPrivateUsage]
        "MCPServer importable from mcp.server": MCPServer.__module__,
    }
    out = RESULTS / "sdk-facts.json"
    out.write_text(json.dumps(facts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(facts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
