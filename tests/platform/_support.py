"""Helpers for the Windows platform tests: start ``_child.py`` modes, wait for their files, clean up."""

from __future__ import annotations

import functools
import json
import subprocess
import sys
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

WINDOWS_ONLY = "narration.platform is implemented for Windows only in v1 (plan.md Q2)"
CHILD = Path(__file__).with_name("_child.py")

HOST_FORBIDS_BREAKAWAY = (
    "this host's own Job Objects forbid breakaway (KNOW: GitHub's hosted Windows runner does), so no child of "
    "this test can be in no job at all, and a detached start is refused here (left_in_job): the test asserts "
    "that refusal instead of the daemon's survival"
)


class HostForbidsBreakaway(UserWarning):
    """The warning a survival test gives when it took its refusal branch (``HOST_FORBIDS_BREAKAWAY``), so the
    run's warnings summary, and so a CI log, shows which branch each such test took."""


def note_the_refusal_branch() -> None:
    """Warn (``HostForbidsBreakaway``) that the calling test asserted the refusal, not the daemon's survival.
    On GitHub's hosted Windows runner every survival test takes this branch, so a regression that refused
    every detached start would stay green there; only a run on a host whose jobs allow breakaway (a Windows
    developer's) exercises the survival itself."""
    warnings.warn(HOST_FORBIDS_BREAKAWAY, HostForbidsBreakaway, stacklevel=2)


@functools.cache
def host_lets_a_child_leave_every_job() -> bool:
    """Whether a child this process starts with ``CREATE_BREAKAWAY_FROM_JOB`` ends up in no Job Object at all.

    False where a job around this process forbids breakaway (GitHub's hosted Windows runner is one, KNOW from
    PR #37's CI): then ``spawn_detached`` refuses with ``left_in_job`` (or ``breakaway_refused``, if that job is
    the innermost) rather than run a daemon, and the survival tests assert the refusal. Probed once per test
    process with a suspended child that never runs. Windows only.
    """
    from narration.platform import _windows  # pyright: ignore[reportPrivateUsage]

    try:
        probe = subprocess.Popen(
            [sys.executable, "-c", "pass"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_windows.CREATE_BREAKAWAY_FROM_JOB | _windows.CREATE_SUSPENDED,
        )
    except PermissionError:
        return False  # the innermost job forbids breakaway: CreateProcess itself refused
    try:
        return not _windows._in_job(probe.pid, None)  # pyright: ignore[reportPrivateUsage]
    finally:
        probe.kill()
        probe.wait(timeout=30)


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


def start_in_job(job: Any, argv: list[str]) -> subprocess.Popen[bytes]:
    """Start ``argv`` inside the Job Object ``job`` (a ``narration.platform._windows._JobObject``) from its
    first instruction: created suspended, assigned, then resumed. This is how a client that wants every
    descendant in its job should start a server; the MCP Python SDK and libuv assign after the start instead,
    so a child the server starts in the meantime lands outside. Windows only."""
    from narration.platform import _windows  # pyright: ignore[reportPrivateUsage]

    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=_windows.CREATE_SUSPENDED,
    )
    try:
        job.add(process.pid)
        _windows._resume(process.pid)  # pyright: ignore[reportPrivateUsage]
    except BaseException:
        process.kill()
        process.wait(timeout=30)
        raise
    return process


def run_in_job(job: Any, *args: object, timeout_s: float = 60.0) -> None:
    """Run a child mode to completion inside ``job`` (``start_in_job``); fail if it fails."""
    process = start_in_job(job, child_argv(*args))
    assert process.wait(timeout=timeout_s) == 0, f"the child in the job exited with {process.returncode}"


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
