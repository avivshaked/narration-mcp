"""Helpers for the Windows platform tests: start ``_child.py`` modes, wait for their files, clean up."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

WINDOWS_ONLY = "narration.platform is implemented for Windows only in v1 (plan.md Q2)"
CHILD = Path(__file__).with_name("_child.py")


def child_argv(*args: object) -> list[str]:
    """The command line of one ``_child.py`` mode, run by this interpreter."""
    return [sys.executable, str(CHILD), *(str(a) for a in args)]


def run_child(*args: object, timeout_s: float = 60.0) -> None:
    """Run a child mode to completion; fail with its stderr if it fails."""
    done = subprocess.run(child_argv(*args), stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout_s)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")


@contextmanager
def started(*args: object) -> Iterator[subprocess.Popen[bytes]]:
    """Start a child mode; on the way out, stop it if it still runs (it is this test's own child)."""
    process = subprocess.Popen(
        child_argv(*args), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=30)


def wait_for_file(path: Path, timeout_s: float = 30.0) -> bool:
    """Whether ``path`` appears within ``timeout_s``."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.02)
    return False


def heartbeat_stopped(marker: Path, timeout_s: float = 10.0) -> bool:
    """Whether the ``sleep`` child behind ``marker`` stops beating within ``timeout_s``.

    It beats every 50 ms, so a file that does not grow for 0.5 s belongs to a stopped interpreter.
    """
    beat = Path(str(marker) + ".beat")
    deadline = time.monotonic() + timeout_s
    size = beat.stat().st_size if beat.exists() else -1
    while time.monotonic() < deadline:
        time.sleep(0.5)
        now = beat.stat().st_size if beat.exists() else -1
        if now == size:
            return True
        size = now
    return False


def is_beating(marker: Path) -> bool:
    """Whether the ``sleep`` child behind ``marker`` beats within the next half second."""
    return not heartbeat_stopped(marker, timeout_s=0.6)


def read_json(path: Path) -> dict[str, Any]:
    """A child's JSON result."""
    return json.loads(path.read_text(encoding="utf-8"))
