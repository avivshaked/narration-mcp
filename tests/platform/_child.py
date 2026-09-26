"""Child processes for the Windows platform tests (run as a script: ``python _child.py <mode> <args>``).

Every mode exits by itself within seconds, even if the test that started it has died, and none touches a
process it did not start. Results are JSON files written to a temp name and renamed, so a test never reads
half a file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from narration.contracts.errors import NarrationError
from narration.platform import get_platform

if sys.platform != "win32":
    raise SystemExit("the platform test children run on Windows only")

MAX_LIFETIME_S = 15.0
"""The longest any child lives, whatever happens to the test."""


def write_json(path: str, data: dict[str, Any]) -> None:
    tmp = Path(path + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def wait_for(path: str, timeout_s: float = MAX_LIFETIME_S) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.02)
    return False


def sleep(marker: str, seconds: str) -> None:
    """A worker stand-in: say it runs, then append a byte to ``<marker>.beat`` every 50 ms until it exits.

    The heartbeat lets a test see that this interpreter stopped without looking at any process: under a venv,
    ``sys.executable`` is a launcher and this interpreter is its child, with a pid of its own.
    """
    write_json(marker, {"pid": os.getpid(), "ppid": os.getppid()})
    deadline = time.monotonic() + min(float(seconds), MAX_LIFETIME_S)
    while time.monotonic() < deadline:
        try:
            with open(marker + ".beat", "ab") as beat:
                beat.write(b".")
        except OSError:
            pass  # a test reading the file at this instant; the next beat will do
        time.sleep(0.05)


def singleton(store_root: str, out: str) -> None:
    """Try the singleton once and report whether it was acquired."""
    with get_platform().singleton(Path(store_root)) as acquired:
        write_json(out, {"acquired": acquired})


def hold(store_root: str, marker: str, go: str) -> None:
    """Hold the singleton until ``go`` exists, then die without releasing it (a crashed daemon)."""
    with get_platform().singleton(Path(store_root)) as acquired:
        write_json(marker, {"acquired": acquired})
        wait_for(go)
        os._exit(0)


def refused(out: str) -> None:
    """From inside a Job Object that forbids breakaway, try to start a daemon detached."""
    platform = get_platform()
    with platform.kill_on_close_group() as add:
        add(os.getpid())
        try:
            pid = platform.spawn_detached([sys.executable, "-c", "pass"], cwd=Path(out).parent, env=dict(os.environ))
        except NarrationError as exc:
            write_json(out, _fields(exc))
        else:
            write_json(out, {"spawned_pid": pid})
        os._exit(0)  # leaving the block would close the job, which holds this process too


def _fields(exc: NarrationError) -> dict[str, Any]:
    return {
        "code": exc.code,
        "retryable": exc.retryable,
        "retry_after_s": exc.retry_after_s,
        "details": exc.details,
    }


def client(daemon_marker: str, go: str, out: str, cwd: str, mark: str, how: str) -> None:
    """An MCP client stand-in: run inside a kill-on-close Job Object that allows breakaway, start a daemon,
    report its pid, and exit at once, which kills everything left in the job.

    ``how`` is ``platform`` (``spawn_detached``) or ``plain`` (a detached start without breakaway, the control).
    """
    from narration.platform._windows import _JobObject

    job = _JobObject(kill_on_close=True, allow_breakaway=True)
    job.add(os.getpid())
    argv = [sys.executable, __file__, "daemon", daemon_marker, go]
    env = dict(os.environ, NARRATION_TEST_MARK=mark)
    if how == "platform":
        try:
            pid = get_platform().spawn_detached(argv, cwd=Path(cwd), env=env)
        except NarrationError as exc:
            write_json(out, {"refused": _fields(exc)})
            os._exit(0)
    else:
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(argv, cwd=cwd, env=env, creationflags=flags, stdin=subprocess.DEVNULL)
        pid = proc.pid
    write_json(out, {"daemon_pid": pid})
    os._exit(0)


def daemon(marker: str, go: str) -> None:
    """A detached daemon stand-in: once ``go`` exists (its client is gone), report what it sees, then exit."""
    if wait_for(go):
        stdin = sys.stdin.read() if sys.stdin is not None else None
        write_json(
            marker,
            {
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "cwd": os.getcwd(),
                "mark": os.environ.get("NARRATION_TEST_MARK"),
                "stdin": stdin,
            },
        )


def supervisor(worker_pid: str, out: str, go: str) -> None:
    """A daemon stand-in: put a worker in a kill-on-close group, then die without closing it."""
    with get_platform().kill_on_close_group() as add:
        add(int(worker_pid))
        write_json(out, {"added": True})
        wait_for(go)
        os._exit(0)


MODES = {
    "sleep": sleep,
    "singleton": singleton,
    "hold": hold,
    "refused": refused,
    "client": client,
    "daemon": daemon,
    "supervisor": supervisor,
}

if __name__ == "__main__":
    MODES[sys.argv[1]](*sys.argv[2:])
