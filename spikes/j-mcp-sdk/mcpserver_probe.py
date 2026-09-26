"""Spike (j): what the high-level ``MCPServer`` of ``mcp`` 2.2.0 does with the cases design section 14 cares about.

A tool is declared the ``MCPServer`` way (a typed function; pydantic builds the schema and validates),
then called in-process through the SDK's ``Client`` in ``mode="legacy"``, which drives the real JSON-RPC
stream loop. It records: the published schemas, an unknown field (``instruct``), a wrong type, and an
unknown tool.

Output: ``results/mcpserver-checks.json``. Nothing here fails the run: it records behaviour.

Run: ``uv run python spikes/j-mcp-sdk/mcpserver_probe.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import anyio
from mcp import Client, MCPError
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, ConfigDict

RESULTS = Path(__file__).resolve().parent / "results"


class Voice(BaseModel):
    """The design's Voice fragment, as a pydantic model."""

    model_config = ConfigDict(extra="forbid")
    path: str
    sha256: str
    transcript: str


def build() -> MCPServer:
    server = MCPServer("mcpserver-probe")

    @server.tool()
    def check_line(voice: Voice, text: str, takes: int = 1) -> dict[str, Any]:
        """Return the spoken form of a line."""
        return {"spoken": " ".join(text.split()), "takes": takes}

    return server


async def run() -> dict[str, Any]:
    observed: dict[str, Any] = {}
    voice = {"path": "C:/voices/narrator.wav", "sha256": "5b" * 32, "transcript": "Good bread asks for patience."}
    async with Client(build(), mode="legacy") as client:
        tool = (await client.list_tools()).tools[0]
        schema_text = json.dumps(tool.input_schema)
        observed["published inputSchema"] = tool.input_schema
        observed["inputSchema contains $ref"] = '"$ref"' in schema_text
        observed["inputSchema root additionalProperties"] = tool.input_schema.get("additionalProperties", "absent")
        observed["published outputSchema"] = tool.output_schema

        extra = await client.call_tool("check_line", {"voice": voice, "text": "a line", "instruct": "calm"})
        observed["unknown top-level field instruct"] = {
            "is_error": extra.is_error,
            "structured_content": extra.structured_content,
        }

        wrong = await client.call_tool("check_line", {"voice": voice, "text": "a line", "takes": "three"})
        observed["wrong type"] = {
            "is_error": wrong.is_error,
            "structured_content": wrong.structured_content,
            "text": wrong.content[0].text if wrong.content and wrong.content[0].type == "text" else None,
        }

        try:
            unknown = await client.call_tool("no_such_tool", {})
            observed["unknown tool"] = {
                "json_rpc_error": False,
                "is_error": unknown.is_error,
                "text": unknown.content[0].text if unknown.content and unknown.content[0].type == "text" else None,
            }
        except MCPError as exc:
            observed["unknown tool"] = {"json_rpc_error": True, "code": exc.error.code}
    return observed


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    observed = anyio.run(run)
    out = RESULTS / "mcpserver-checks.json"
    out.write_text(json.dumps(observed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(observed, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
