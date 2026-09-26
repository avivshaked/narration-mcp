"""The developers' GPU lock (plan.md section 2.5; AGENTS.md section 5).

The GPU on a development machine is shared with other people's jobs, and one holder at a time may load a
model on it. The lock is a directory, ``<main checkout>/.dev/gpu.lock/`` (creating a directory is atomic),
with an ``owner.json`` inside: holder, start time, expected end, PID. Every worktree of the repository
shares the one lock, because it lives in the main checkout.

A lock past its expected end by more than 15 minutes, whose PID is gone, is **reported as stale, never
broken**: the lead clears it.

Usage::

    python tools/gpu_lock.py acquire --holder WP20 --minutes 30 [--need-mb 8000] [--wait-min 0]
    python tools/gpu_lock.py release --holder WP20
    python tools/gpu_lock.py status

Standard library only. This is a development tool; the service's own GPU scheduler is in the daemon
(design section 4).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MAX_MINUTES = 30
STALE_GRACE_MIN = 15
MARGIN_MB = 1024


def main_checkout() -> Path:
    """The main checkout's root, also when called from a linked worktree."""
    out = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], check=True, capture_output=True, text=True
    )
    return Path(out.stdout.strip()).parent


def lock_dir(root: Path | None = None) -> Path:
    return (root or main_checkout()) / ".dev" / "gpu.lock"


def now() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(microsecond=0)


def read_owner(lock: Path) -> dict[str, object] | None:
    try:
        return json.loads((lock / "owner.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"], capture_output=True, text=True, check=False
        )
        return f'"{pid}"' in out.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def is_stale(owner: dict[str, object]) -> bool:
    try:
        expected_end = dt.datetime.fromisoformat(str(owner["expected_end"]))
        pid = int(str(owner.get("pid", 0)))
    except (KeyError, ValueError):
        return True
    return now() > expected_end + dt.timedelta(minutes=STALE_GRACE_MIN) and not pid_alive(pid)


def free_vram_mb() -> int | None:
    """Free memory of GPU 0 by ``nvidia-smi``, or None when there is no NVIDIA GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return int(out.stdout.strip().splitlines()[0])


def acquire(
    holder: str, minutes: int, need_mb: int | None = None, wait_min: float = 0, root: Path | None = None
) -> int:
    """Take the lock. Returns 0 when taken, 1 when it is held by another, 2 when VRAM is short."""
    if minutes > MAX_MINUTES:
        print(f"runs longer than {MAX_MINUTES} min need the lead's OK first (AGENTS.md section 5)", file=sys.stderr)
        return 2
    lock = lock_dir(root)
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + wait_min * 60
    while True:
        if need_mb is not None:
            free = free_vram_mb()
            if free is None:
                print("no NVIDIA GPU found (nvidia-smi failed)", file=sys.stderr)
                return 2
            if free < need_mb + MARGIN_MB:
                print(f"free VRAM {free} MB < need {need_mb} MB + {MARGIN_MB} MB margin", file=sys.stderr)
                if time.monotonic() >= deadline:
                    return 2
                time.sleep(15)
                continue
        try:
            lock.mkdir()
        except FileExistsError:
            owner = read_owner(lock)
            if owner and owner.get("holder") == holder:
                print(f"already held by {holder}")
                return 0
            state = "STALE (report it to the lead; never break it)" if owner and is_stale(owner) else "held"
            print(f"GPU lock {state}: {json.dumps(owner, ensure_ascii=False)}", file=sys.stderr)
            if time.monotonic() >= deadline:
                return 1
            time.sleep(15)
            continue
        start = now()
        owner = {
            "holder": holder,
            "start": start.isoformat(),
            "expected_end": (start + dt.timedelta(minutes=minutes)).isoformat(),
            "pid": os.getppid(),
        }
        tmp = lock / "owner.json.partial"
        tmp.write_text(json.dumps(owner, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, lock / "owner.json")
        print(f"GPU lock taken by {holder} until {owner['expected_end']}")
        return 0


def release(holder: str, root: Path | None = None) -> int:
    """Release the lock if ``holder`` holds it. Returns 0 on success, 1 otherwise."""
    lock = lock_dir(root)
    owner = read_owner(lock)
    if not lock.exists():
        print("GPU lock is free")
        return 0
    if owner is not None and owner.get("holder") != holder:
        print(f"GPU lock is held by {owner.get('holder')}, not {holder}; not released", file=sys.stderr)
        return 1
    (lock / "owner.json").unlink(missing_ok=True)
    (lock / "owner.json.partial").unlink(missing_ok=True)
    lock.rmdir()
    print(f"GPU lock released by {holder}")
    return 0


def status(root: Path | None = None) -> int:
    lock = lock_dir(root)
    if not lock.exists():
        print("free")
        return 0
    owner = read_owner(lock)
    stale = owner is not None and is_stale(owner)
    print(json.dumps({"held": True, "stale": stale, "owner": owner}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("acquire", help="take the lock")
    a.add_argument("--holder", required=True, help="who holds it, e.g. WP20")
    a.add_argument("--minutes", type=int, required=True, help=f"expected duration (at most {MAX_MINUTES})")
    a.add_argument("--need-mb", type=int, default=None, help="refuse unless nvidia-smi shows this + 1 GB free")
    a.add_argument("--wait-min", type=float, default=0, help="keep trying for this many minutes")
    r = sub.add_parser("release", help="release the lock")
    r.add_argument("--holder", required=True)
    sub.add_parser("status", help="show who holds the lock")
    args = parser.parse_args(argv)
    if args.cmd == "acquire":
        return acquire(args.holder, args.minutes, args.need_mb, args.wait_min)
    if args.cmd == "release":
        return release(args.holder)
    return status()


if __name__ == "__main__":
    sys.exit(main())
