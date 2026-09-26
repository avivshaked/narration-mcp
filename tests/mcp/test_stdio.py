"""The front-end as a stdio server process, driven by the SDK's own client (design sections 4 and 5).

The server is started as ``sys.executable -m tests.mcp.stdio_server``, never through an ``.exe`` launcher.
Each run is bounded by a timeout, and the SDK's stdio client closes stdin and then kills the process tree
when the block ends, on every exit path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import anyio
import pytest
from mcp import Client, StdioServerParameters, stdio_client

from narration.contracts import codes
from narration.contracts.names import TOOL_NAMES
from tests.mcp.fake_backend import JOB_ID, VALID_ARGUMENTS

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_TIMEOUT_S = 60


@pytest.mark.parametrize("mode", ["legacy", "auto"])
def test_the_front_end_serves_a_client_over_stdio_s5(tmp_path: Path, mode: Literal["legacy", "auto"]) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "tests.mcp.stdio_server", str(tmp_path / "store")],
        cwd=REPO_ROOT,
    )
    stderr_path = tmp_path / "server-stderr.log"

    async def main() -> None:
        with stderr_path.open("w", encoding="utf-8") as errlog, anyio.fail_after(RUN_TIMEOUT_S):
            async with Client(stdio_client(params, errlog=errlog), mode=mode) as client:
                tools = await client.list_tools()
                assert tuple(t.name for t in tools.tools) == TOOL_NAMES
                results = await client.call_tool("get_results", {"job_id": JOB_ID})
                assert results.is_error is False and results.structured_content is not None
                assert any(item.type == "resource_link" for item in results.content)
                refused = await client.call_tool("submit_job", {**VALID_ARGUMENTS["submit_job"], "instruct": "x"})
                assert refused.is_error is True and refused.structured_content is not None
                assert refused.structured_content["error"]["code"] == codes.INVALID_ARGUMENT

    try:
        anyio.run(main)
    except BaseException:
        print(stderr_path.read_text(encoding="utf-8"), file=sys.stderr)
        raise
