"""``narration-admin verify [--json]``: re-hash what must not change (design section 15).

- **The store** (``NarrationStore.verify``, WP12): every immutable file is hashed again against the index,
  and SQLite's own integrity check is run. A file missing or changed is a damaged cached result; one that
  lost its read-only mark but still matches is only reported.
- **The models** (``narration.admin.models``): every file the install recorded in
  ``<models_root>/manifest.json`` is hashed again.

It changes nothing. The exit code is 1 when anything is missing, changed or unreadable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .cli import EXIT_FAILED, EXIT_OK, PROGRAM, Admin, Subparsers
from .models import check_files, read_manifest


def register(subparsers: Subparsers) -> None:
    """Add ``verify`` (``narration.admin.cli``)."""
    parser = subparsers.add_parser(
        "verify",
        help="re-hash the store's immutable files and the installed models (section 15)",
        description=(
            "Hash every immutable file in the store again against its index, run the database's integrity "
            "check, and hash every installed model file against the install record. Changes nothing."
        ),
    )
    parser.add_argument("--json", action="store_true", help="print the reports as JSON")
    parser.set_defaults(handler=run_verify)


def run_verify(admin: Admin, args: argparse.Namespace) -> int:
    """``verify``: see the module docstring."""
    config = admin.config()
    store_report: dict[str, Any] | None = admin.store().verify() if admin.store_exists() else None
    models_report = _verify_models(config.server.models_root)
    problems = _store_problems(store_report) + len(models_report["problems"])
    if args.json:
        admin.say(
            json.dumps(
                {"ok": problems == 0, "store": store_report, "models": models_report}, ensure_ascii=False, indent=2
            )
        )
        return EXIT_FAILED if problems else EXIT_OK
    root = config.server.store_root
    if store_report is None:
        admin.say(f"store: no store at {root} yet; nothing to check.")
    else:
        matched = f"{store_report['ok']} of {store_report['checked']} files match"
        admin.say(f"store: {matched}; database: {store_report['database']}.")
        for rel in store_report["missing"]:
            admin.say(f"  missing: {rel}")
        for item in store_report["mismatched"]:
            admin.say(f"  changed: {item['path']}")
        for item in store_report["errors"]:
            admin.say(f"  cannot check: {item['path']} ({item['error']})")
        if store_report["writable"]:
            admin.say(
                f"  {len(store_report['writable'])} file(s) are no longer read-only but still match (something outside "
                "the service changed their attributes)."
            )
    admin.say(f"models: {models_report['files']} file(s) checked in {models_report['models']} model(s).")
    for problem in models_report["problems"]:
        admin.say(f"  {problem}")
    if problems:
        admin.say(
            f"\n{problems} problem(s). A changed or missing model file is repaired by `{PROGRAM} install`. A "
            "damaged cached result in the store is not repaired by the service yet: keep this report and tell "
            "the maintainers."
        )
        return EXIT_FAILED
    admin.say("\nEverything checked matches.")
    return EXIT_OK


def _store_problems(report: dict[str, Any] | None) -> int:
    if report is None:
        return 0
    return (
        len(report["missing"])
        + len(report["mismatched"])
        + len(report["errors"])
        + (0 if report["database"] == "ok" else 1)
    )


def _verify_models(models_root: Path) -> dict[str, Any]:
    try:
        records = read_manifest(models_root)
    except ValueError as exc:
        return {"models": 0, "files": 0, "problems": [str(exc)]}
    problems = [f"{model.key}: {problem}" for model in records.values() for problem in check_files(model)]
    files = sum(len(model.files_sha256) for model in records.values())
    return {"models": len(records), "files": files, "problems": problems}
