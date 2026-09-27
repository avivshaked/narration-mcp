"""``narration-admin gc [--apply] [--json]``: remove what retention no longer keeps (design section 15).

**A dry run by default**: it lists what would go and removes nothing; ``--apply`` removes it. The store's
``gc`` does the work (``NarrationStore.gc``, WP12): renders, takes, analyses, voice profiles, designs and
finished jobs unused for ``[retention] retention_days``, measurements unused for
``measurement_retention_days``, and what a crash left behind. The provenance list, engine profiles and
their canaries, alignment benchmarks, and queued or running jobs are never collected. It is safe while the
daemon runs: the store collects in short transactions, and never touches a key that is published again
meanwhile.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from .cli import EXIT_FAILED, EXIT_OK, PROGRAM, Admin, Subparsers

KINDS = ("renders", "takes", "analyses", "profiles", "designs", "measurements", "jobs")


def register(subparsers: Subparsers) -> None:
    """Add ``gc`` (``narration.admin.cli``)."""
    parser = subparsers.add_parser(
        "gc",
        help="remove what retention no longer keeps; a dry run by default (section 15)",
        description=(
            "List what retention no longer keeps: cached renders, takes, analyses, profiles, designs, finished "
            "jobs, old measurements and crash leftovers. Nothing is removed without --apply."
        ),
    )
    parser.add_argument("--apply", action="store_true", help="remove what the dry run lists")
    parser.add_argument("--json", action="store_true", help="print the store's report as JSON")
    parser.set_defaults(handler=run_gc)


def run_gc(admin: Admin, args: argparse.Namespace) -> int:
    """``gc``: see the module docstring."""
    root = admin.config().server.store_root
    if not admin.store_exists():
        admin.say(f"No store at {root} yet; nothing to collect.")
        return EXIT_OK
    report: dict[str, Any] = admin.store().gc(dry_run=not args.apply)
    errors: list[dict[str, str]] = list(report.get("errors", []))
    if args.json:
        admin.say(json.dumps(report, ensure_ascii=False, indent=2))
        return EXIT_FAILED if errors else EXIT_OK
    items: dict[str, list[str]] = report.get("items", {})
    counts = [f"{len(items.get(kind, []))} {kind}" for kind in KINDS if items.get(kind)]
    loose = [
        f"{len(report.get(name, []))} {label}"
        for name, label in (
            ("leftovers", "crash leftovers"),
            ("orphans", "unindexed folders"),
            ("scratch", "scratch files"),
        )
        if report.get(name)
    ]
    what = ", ".join(counts + loose) or "nothing"
    size = _size(int(report.get("bytes", 0)))
    if args.apply:
        admin.say(f"Removed {what} ({size}) from {root}.")
    else:
        admin.say(f"A dry run: {what} ({size}) would be removed from {root}.")
        if counts or loose:
            admin.say(f"Nothing was removed. `{PROGRAM} gc --apply` removes them.")
    cutoffs = report.get("cutoffs", {})
    admin.say(
        f"Kept: what was used since {cutoffs.get('cache')} (measurements since {cutoffs.get('measurements')}), "
        "and always the provenance list, engine profiles and alignment benchmarks."
    )
    for error in errors:
        admin.warn(f"{PROGRAM}: could not remove {error.get('path')}: {error.get('error')}")
    if errors:
        admin.warn(f"{PROGRAM}: {len(errors)} item(s) were kept; they are tried again on the next run.")
        return EXIT_FAILED
    return EXIT_OK


def _size(count: int) -> str:
    for unit, scale in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if count >= scale:
            return f"{count / scale:.1f} {unit}"
    return f"{count} bytes"
