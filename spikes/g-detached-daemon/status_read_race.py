"""Spike (g), part 2: reading ``run/daemon.json`` while the daemon replaces it (design section 15).

The front-end, ``narration-admin`` and a starting daemon read ``run/daemon.json`` while a running daemon
may be renaming a new one over it. This script measures what such a reader sees on this machine:

1. ``os.path.realpath`` of a file that another thread keeps renaming away and recreating: how often the
   result is not under the folder the file is in (the store's path check compares ``realpath``s);
2. ``NarrationStore.get_daemon_status`` in this process while another process keeps writing the status
   with ``put_daemon_status``: which errors a read gets, and how often;
3. the same reads through ``narration.daemon.sweep.read_status``, which reads again after a short pause.

Everything is written under ``<repo>/.dev/spike-g/race/``; the counts go to ``results/status-read-race.json``.

    uv run python spikes/g-detached-daemon/status_read_race.py
"""

from __future__ import annotations

import collections
import json
import os
import platform as pyplatform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORK = REPO / ".dev" / "spike-g" / "race"
RESULTS = HERE / "results"
SECONDS = 8.0


def realpath_race() -> dict[str, Any]:
    folder = WORK / "realpath"
    folder.mkdir(parents=True)
    root = os.path.realpath(folder)
    tmp, target = os.path.join(root, ".tmp-x"), os.path.join(root, "daemon.json")
    stop = threading.Event()
    renames = collections.Counter()

    def writer() -> None:
        while not stop.is_set():
            with open(tmp, "wb") as f:
                f.write(b"{}")
            try:
                os.replace(tmp, target)
                renames["ok"] += 1
            except PermissionError:
                renames["PermissionError"] += 1

    thread = threading.Thread(target=writer)
    thread.start()
    calls = outside = prefixed = 0
    deadline = time.monotonic() + SECONDS
    try:
        while time.monotonic() < deadline:
            real = os.path.realpath(tmp)
            calls += 1
            if not real.startswith(root):
                outside += 1
                prefixed += real.startswith("\\\\?\\")
    finally:
        stop.set()
        thread.join(timeout=30)
    return {
        "realpath_calls": calls,
        "results_outside_the_folder": outside,
        "of_which_long_path_prefixed": prefixed,
        "writer_renames": dict(renames),
    }


def writer_process(store_root: Path, seconds: float) -> None:
    import dataclasses

    from narration.platform import get_platform
    from narration.store import NarrationStore

    with NarrationStore(store_root, get_platform()) as store:
        base = store.get_daemon_status()
        assert base is not None
        writes = 0
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            store.put_daemon_status(dataclasses.replace(base, est_drain_s=float(writes)))
            writes += 1
    print(json.dumps({"writes": writes}))


def store_race(use_read_status: bool) -> dict[str, Any]:
    from narration.daemon.status import StatusBoard
    from narration.daemon.sweep import read_status
    from narration.platform import get_platform
    from narration.store import NarrationStore

    store_root = WORK / ("store-retry" if use_read_status else "store-plain")
    errors: collections.Counter[str] = collections.Counter()
    reads = 0
    with NarrationStore(store_root, get_platform()) as store:
        StatusBoard(store, pid=os.getpid(), started_at="2026-01-01T00:00:00.000Z").set_state("idle")
        writer = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--writer", str(store_root), str(SECONDS + 2)],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            time.sleep(1.0)
            deadline = time.monotonic() + SECONDS
            while time.monotonic() < deadline:
                try:
                    read_status(store) if use_read_status else store.get_daemon_status()
                except Exception as exc:
                    errors[type(exc).__name__] += 1
                reads += 1
        finally:
            out, _ = writer.communicate(timeout=60)
    return {"reads": reads, "errors": dict(errors), "writer": json.loads(out.strip().splitlines()[-1])}


def main() -> None:
    if sys.argv[1:2] == ["--writer"]:
        writer_process(Path(sys.argv[2]), float(sys.argv[3]))
        return
    if sys.platform != "win32":
        raise SystemExit("this measures Windows file-sharing behaviour; run it on Windows")
    shutil.rmtree(WORK, ignore_errors=True)  # this script's own folder under .dev
    report = {
        "environment": {
            "date": time.strftime("%Y-%m-%d"),
            "python": pyplatform.python_version(),
            "windows": pyplatform.version(),
        },
        "seconds_each": SECONDS,
        "realpath_of_a_file_renamed_away": realpath_race(),
        "get_daemon_status_while_written": store_race(use_read_status=False),
        "read_status_while_written": store_race(use_read_status=True),
    }
    RESULTS.mkdir(exist_ok=True)
    tmp = RESULTS / ".status-read-race.json.tmp"
    tmp.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, RESULTS / "status-read-race.json")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
