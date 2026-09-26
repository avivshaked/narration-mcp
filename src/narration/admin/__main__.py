"""Entry point of ``narration-admin``, the operator CLI (design section 7.1)."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Run an operator command. Not built yet (plan.md WP37)."""
    print("narration-admin is not built yet; see plan.md (WP37).", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
