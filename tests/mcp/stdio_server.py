"""The front-end over the fake backend, on stdio: the server process of ``test_stdio``.

Run as ``python -m tests.mcp.stdio_server <store_root>`` from the repository root. It is test code: the
product's entry point (``python -m narration.mcp``) gets its real backend in WP36.
"""

from __future__ import annotations

import sys
from pathlib import Path

import anyio

from narration.mcp import build_front_end
from tests.mcp.fake_backend import FakeBackend


def main(argv: list[str]) -> int:
    """Serve one client on stdio until stdin closes."""
    front = build_front_end(FakeBackend(Path(argv[0])))
    anyio.run(front.run_stdio)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
