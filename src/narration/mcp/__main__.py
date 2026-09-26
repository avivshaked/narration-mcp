"""Entry point of ``narration-mcp``, the stdio MCP server (design section 4)."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Run the MCP server on stdio. Not built yet (plan.md WP17)."""
    print("narration-mcp is not built yet; see plan.md (WP17, WP36).", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
