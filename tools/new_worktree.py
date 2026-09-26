"""Create a work package's git worktree (plan.md section 2.3).

Runs, from the main checkout::

    git worktree add worktrees/wp<NN>-<slug> -b wp/<NN>-<slug> <base>

then copies the gitignored ``AGENTS.local.md`` in, and syncs the server venv with ``uv sync`` (and, with
``--worker``, a worker project's venv), using the shared uv cache in ``<main checkout>/.dev/uv-cache`` so
every venv hardlinks the same wheels. The git hooks need nothing: ``core.hooksPath`` is shared by every
worktree of the repository.

Usage::

    python tools/new_worktree.py 10 text [--base main] [--worker qwen3tts] [--no-sync]

Standard library only.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main_checkout() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], check=True, capture_output=True, text=True
    )
    return Path(out.stdout.strip()).parent


def uv_env(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("UV_CACHE_DIR", str(root / ".dev" / "uv-cache"))
    env["UV_PYTHON_DOWNLOADS"] = "never"
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("number", help="the work package number, e.g. 10")
    parser.add_argument("slug", help="a short name, e.g. text")
    parser.add_argument("--base", default="main", help="the branch or commit to start from (default: main)")
    parser.add_argument(
        "--worker", action="append", default=[], choices=["qwen3tts", "qa"], help="also sync this worker"
    )
    parser.add_argument("--no-sync", action="store_true", help="skip uv sync")
    args = parser.parse_args(argv)

    root = main_checkout()
    number = f"{int(args.number):02d}"
    name = f"wp{number}-{args.slug}"
    branch = f"wp/{number}-{args.slug}"
    path = root / "worktrees" / name
    if path.exists():
        print(f"{path} exists already", file=sys.stderr)
        return 1
    subprocess.run(["git", "worktree", "add", str(path), "-b", branch, args.base], cwd=root, check=True)
    local = root / "AGENTS.local.md"
    if local.is_file():
        shutil.copyfile(local, path / "AGENTS.local.md")
    else:
        print("warning: AGENTS.local.md not found in the main checkout; ask the lead", file=sys.stderr)
    if not args.no_sync:
        env = uv_env(root)
        subprocess.run(["uv", "sync", "--python", "3.12"], cwd=path, env=env, check=True)
        for worker in args.worker:
            subprocess.run(["uv", "sync", "--python", "3.12"], cwd=path / "workers" / worker, env=env, check=True)
    print(f"worktree: {path}\nbranch:   {branch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
