"""Starting the daemon, and telling whether one runs (design sections 4 and 4.1).

For the front-end's autostart (WP36) and ``narration-admin daemon start | status`` (WP37):

- ``start_detached`` starts ``python -m narration.daemon --store <store_root> --config <path>`` through
  ``Platform.spawn_detached``: ``CREATE_BREAKAWAY_FROM_JOB | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP``,
  the standard handles on ``NUL``, nothing inherited, the store root as the working directory. When the
  caller's Job Object forbids breakaway, it raises ``DAEMON_UNAVAILABLE`` (retryable, with the hint to run
  ``narration-admin daemon start`` in a terminal) and starts nothing: a daemon that is not detached would
  die with its client mid-job;
- ``running_daemon`` reads ``run/daemon.json`` and checks its pid (``sweep.daemon_alive``). Post a
  ``stop`` only when it says a daemon runs (``narration-admin daemon stop``, WP37): a daemon honours only
  the stops posted after it was launched (``service``, "Which stops a daemon honours"), so a stop posted
  with none running stops nothing, and the next daemon answers it ``stopped: false``;
- ``ensure_daemon`` starts one unless one runs, and can wait until it has written its status.

Starting a daemon when one already runs is harmless: the second one exits quietly (the singleton). One that
finds the running daemon ``stopping`` waits for it to go, then takes over.

**The interpreter.** The daemon runs as ``ProcessPlatform.python_for(python, console=False)``: on Windows
``pythonw.exe``, the venv's windowless Python, when it is there. A venv's ``python.exe`` is a launcher that
starts the interpreter as a console program, and that interpreter, started detached, gets a new console of
its own (KNOW, spike g: a ``conhost.exe`` child). Windows may show such a console as a window that a user
could close, killing the daemon (BELIEVE: spike g saw no window owned by the daemon's own processes, but did
not look at terminal hosts outside its tree). ``pythonw.exe`` gets no console at all (KNOW, spike g), and
the daemon starts its workers with ``CREATE_NO_WINDOW``.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from narration.contracts.interfaces import Store
from narration.contracts.models import DaemonStatus
from narration.platform import ProcessPlatform

from .settings import isolated, scrub_python_env
from .sweep import StatusUnreadable, daemon_alive, read_status

DAEMON_MODULE: Final = "narration.daemon"
RUNNING_STATES: Final = ("idle", "busy")
"""States of a daemon that serves the queue (``stopping`` is on its way out; ``stopped`` is gone)."""


def daemon_argv(store_root: Path, config_path: Path, *, python: Path, extra: Sequence[str] = ()) -> list[str]:
    """The daemon's command line, run by ``python`` with ``-P`` (``settings.SAFE_PATH_FLAG``: the store root,
    its working directory, is not put on ``sys.path``); ``-m narration.daemon --store <store_root>`` is its
    identity marker (section 4.1)."""
    return list(
        isolated([str(python), "-m", DAEMON_MODULE, "--store", str(store_root), "--config", str(config_path), *extra])
    )


def start_detached(
    store_root: Path,
    config_path: Path,
    *,
    platform: ProcessPlatform | None = None,
    env: Mapping[str, str] | None = None,
    python: Path | None = None,
    extra: Sequence[str] = (),
) -> int:
    """Start a daemon detached (section 4.1) and return the pid ``spawn_detached`` got back.

    That pid is the first process of the command line, a launcher's under a venv; the daemon records its
    own pid in ``run/daemon.json``. The store root is created first (the singleton's name hashes its
    ``realpath``). The interpreter is ``platform.python_for(python, console=False)``, where ``python`` is
    this interpreter by default. Its environment is ``env`` (this process's by default) without the
    ``PYTHON*`` variables that change imports (``settings.scrub_python_env``). The daemon is told when it was
    launched (``--launched-at``, this process's wall clock just before the spawn), so that a stop posted
    while it starts up is for it. Raises
    ``NarrationError(DAEMON_UNAVAILABLE)`` when breakaway is refused, and never falls back to a daemon that is
    not detached; on an OS v1 does not support, ``UnsupportedPlatform``.
    """
    if platform is None:
        from narration.platform import get_platform

        platform = get_platform()
    root = Path(os.path.abspath(store_root))
    root.mkdir(parents=True, exist_ok=True)
    interpreter = platform.python_for(Path(sys.executable) if python is None else python, console=False)
    launched = ("--launched-at", repr(time.time()))
    argv = daemon_argv(root, Path(os.path.abspath(config_path)), python=interpreter, extra=(*extra, *launched))
    return platform.spawn_detached(argv, cwd=root, env=scrub_python_env(os.environ if env is None else env))


def running_daemon(store: Store) -> DaemonStatus | None:
    """The status of the daemon that runs for this store, or None when none does (its ``run/daemon.json``
    is missing, cannot be read, says ``stopped``, or names a process that is gone)."""
    try:
        status = read_status(store)
    except StatusUnreadable:
        return None
    return status if daemon_alive(status) else None


def wait_for_daemon(store: Store, *, timeout_s: float, poll_s: float = 0.1) -> DaemonStatus | None:
    """Wait until a daemon runs and serves (``idle`` or ``busy``); its status, or None at the timeout."""
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        status = running_daemon(store)
        if status is not None and status.state in RUNNING_STATES:
            return status
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(poll_s, remaining))


@dataclass(frozen=True, slots=True, kw_only=True)
class EnsureResult:
    """What ``ensure_daemon`` did: whether it started a daemon (and the pid it got back), and the status
    of the daemon that runs, when known (None if it did not wait, or the wait timed out)."""

    started: bool
    spawned_pid: int | None
    status: DaemonStatus | None


def ensure_daemon(
    store: Store,
    config_path: Path,
    *,
    platform: ProcessPlatform | None = None,
    env: Mapping[str, str] | None = None,
    python: Path | None = None,
    wait_s: float = 0.0,
    extra: Sequence[str] = (),
) -> EnsureResult:
    """Start a daemon unless one serves this store already; with ``wait_s``, wait for it to serve.

    A daemon that is ``stopping`` does not count: a new one is started, and takes over once it has gone.
    Raises as ``start_detached`` does.

    **Call it after the job is committed to the store, never before.** The front-end commits, then reads
    the status here; a daemon about to exit writes ``stopping``, then looks for work once more
    (``JobRunner.has_work``). So either that look sees the job, or this read sees ``stopping`` and starts
    a daemon (``seam``, "The front-end's order"). Called before the commit, it can see ``idle`` from a
    daemon that then exits without the job.
    """
    status = running_daemon(store)
    if status is not None and status.state in RUNNING_STATES:
        return EnsureResult(started=False, spawned_pid=None, status=status)
    pid = start_detached(store.root, config_path, platform=platform, env=env, python=python, extra=extra)
    waited = wait_for_daemon(store, timeout_s=wait_s) if wait_s > 0 else None
    return EnsureResult(started=True, spawned_pid=pid, status=waited)
