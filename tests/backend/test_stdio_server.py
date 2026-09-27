"""``narration-mcp`` as a real stdio server process, over its real backend and store, driven by the SDK's own
client (design sections 4, 5, 16). The store has no engine profile pinned, so nothing needs a model or a GPU
and no daemon is started: a submission is refused with ``BACKEND_NOT_INSTALLED`` before any job exists.

The server is started as ``sys.executable -m narration.mcp``, never through an ``.exe`` launcher, and each
run is bounded by a timeout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import anyio
import pytest
from mcp import Client, StdioServerParameters, stdio_client

from narration.contracts import codes
from narration.contracts.names import TOOL_NAMES

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_TIMEOUT_S = 60

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="the store runs on Windows only in v1 (plan.md 1.4)")


def test_narration_mcp_serves_its_real_backend_over_stdio_s5(tmp_path: Path) -> None:
    store = tmp_path / "store"
    config = tmp_path / "narration.toml"
    config.write_text(
        f"[server]\nstore_root = '{store.as_posix()}'\nmodels_root = '{(tmp_path / 'models').as_posix()}'\n",
        encoding="utf-8",
    )
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "narration.mcp", "--config", str(config)], cwd=REPO_ROOT
    )
    stderr_path = tmp_path / "server-stderr.log"

    async def main() -> None:
        with stderr_path.open("w", encoding="utf-8") as errlog, anyio.fail_after(RUN_TIMEOUT_S):
            async with Client(stdio_client(params, errlog=errlog), mode="auto") as client:
                tools = await client.list_tools()
                assert tuple(t.name for t in tools.tools) == TOOL_NAMES
                status = await client.call_tool("get_server_status", {})
                assert status.is_error is False and status.structured_content is not None
                assert status.structured_content["daemon"]["state"] == "stopped"
                assert status.structured_content["engine_profiles"] == []
                checked = await client.call_tool(
                    "check_text", {"segments": [{"segment_id": "p01", "text": "The ferry leaves at dawn."}]}
                )
                assert checked.is_error is False and checked.structured_content is not None
                assert checked.structured_content["segments"][0]["spoken_chars"] > 0
                voice = {"path": str(tmp_path / "clip.wav"), "sha256": "0" * 64, "transcript": "A quiet morning."}
                refused = await client.call_tool(
                    "submit_job", {"voice": voice, "segments": [{"segment_id": "p01", "text": "The ferry leaves."}]}
                )
                assert refused.is_error is True and refused.structured_content is not None
                assert refused.structured_content["error"]["code"] == codes.BACKEND_NOT_INSTALLED

    try:
        anyio.run(main)
    except BaseException:
        print(stderr_path.read_text(encoding="utf-8"), file=sys.stderr)
        raise
    log = store / "logs" / "narration-mcp.log"
    assert log.is_file() and "serving on stdio" in log.read_text(encoding="utf-8")
