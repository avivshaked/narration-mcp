"""``narration-admin gc [--apply] [--json]``: remove what retention no longer keeps (design section 15).

**A dry run by default**: it lists what would go and removes nothing; ``--apply`` removes it. The store's
``gc`` does the work (``NarrationStore.gc``, WP12): renders, takes, analyses, voice profiles, designs and
finished jobs unused for ``[retention] retention_days``, measurements unused for
``measurement_retention_days``, and what a crash left behind. The provenance list, engine profiles and
their canaries, alignment benchmarks, and queued or running jobs are never collected. It is safe while the
daemon runs: the store collects in short transactions, and never touches a key that is published again
meanwhile.

**Failed and replaced takes** (plan.md WP48) are listed apart, in the dry run, the real run and ``--json``
(its ``failures`` key): how many the store holds, how old the oldest is, which of them this run takes out of
``narration-admin failures``' view, and which only lose their flags and metrics (their analysis goes, the take
stays): ``failures.retention_view``. So an audit can finish, or ``failures --export`` copy them, before they are
removed. What ``gc`` removes does not depend on it: that is retention's rule alone. The list is read before the
collection with the store's reads that never repair the index (``NarrationStore.peek_take`` and the like), so
reading it writes nothing: a dry run changes no row and no file, even where a take's file is missing. If the
list cannot be read, ``gc`` says so and goes on.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from typing import Any

from narration.contracts.errors import NarrationError

from .cli import EXIT_FAILED, EXIT_OK, PROGRAM, Admin, Subparsers
from .failures import collect, retention_view

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
    parser.add_argument(
        "--json", action="store_true", help="print the store's report as JSON, with the failed takes under 'failures'"
    )
    parser.set_defaults(handler=run_gc)


def run_gc(admin: Admin, args: argparse.Namespace) -> int:
    """``gc``: see the module docstring."""
    root = admin.config().server.store_root
    if not admin.store_exists():
        admin.say(f"No store at {root} yet; nothing to collect.")
        return EXIT_OK
    store = admin.store()
    try:
        failures = collect(store)  # before the collection: a real run removes what they name
    except (NarrationError, ValueError, OSError, sqlite3.Error) as exc:  # the view never stands in retention's way
        admin.warn(f"{PROGRAM}: the failed takes could not be listed ({type(exc).__name__}: {exc}); gc goes on.")
        failures = None
    report: dict[str, Any] = store.gc(dry_run=not args.apply)
    audit = retention_view(failures, report) if failures is not None else None
    errors: list[dict[str, str]] = list(report.get("errors", []))
    if args.json:
        admin.say(json.dumps({**report, "failures": audit}, ensure_ascii=False, indent=2))
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
    admin.say(_audit_line(audit, applied=bool(args.apply)))
    for error in errors:
        admin.warn(f"{PROGRAM}: could not remove {error.get('path')}: {error.get('error')}")
    if errors:
        admin.warn(f"{PROGRAM}: {len(errors)} item(s) were kept; they are tried again on the next run.")
        return EXIT_FAILED
    return EXIT_OK


def _audit_line(audit: dict[str, Any] | None, *, applied: bool) -> str:
    """The failed and replaced takes, apart (the module docstring)."""
    if audit is None:
        return f"Failed or replaced takes (`{PROGRAM} failures`): not listed (the warning above says why)."
    held, due = int(audit.get("takes", 0)), len(audit.get("due", []))
    if not held:
        return f"Failed or replaced takes (`{PROGRAM} failures`): none in the store."
    age = audit.get("oldest_age_days")
    oldest = f"; the oldest is from a job created {age} days ago" if age is not None else ""
    line = (
        f"Failed or replaced takes (`{PROGRAM} failures`): {held} {'were' if applied else 'are'} in the store{oldest}."
    )
    lose = len(audit.get("lose_analysis", []))
    if applied:
        line += f" This run removed {due} of them from the audit." if due else " This run removed none of them."
        return line + (f" {lose} lost their flags and metrics (their analysis was removed)." if lose else "")
    if lose:
        line += f" {lose} would lose their flags and metrics (their analysis would be removed; the take stays)."
    if not due:
        return line + " This run would remove none of them."
    return (
        line + f" This run would take {due} of them out of the audit (it removes the take, or every job that "
        f"lists it). `{PROGRAM} failures --export <dir>` copies them, with their reasons, first."
    )


def _size(count: int) -> str:
    for unit, scale in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if count >= scale:
            return f"{count / scale:.1f} {unit}"
    return f"{count} bytes"
