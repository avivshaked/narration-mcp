"""Starting the daemon, and telling whether one runs (design sections 4 and 4.1).

For the front-end's autostart (WP36) and ``narration-admin daemon start | status`` (WP37):

- ``start_detached`` starts ``python -m narration.daemon --store <store_root> --config <path>`` through
  ``Platform.spawn_detached``: ``CREATE_BREAKAWAY_FROM_JOB | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP``,
  the standard handles on ``NUL``, nothing inherited, the store root as the working directory. The daemon
  runs only once the platform has confirmed it is in no Job Object. When the caller's Job Objects keep it
  in one (the innermost forbids breakaway, so Windows refuses; or the innermost allows it and an enclosing
  one does not, so Windows leaves the daemon in that one; KNOW, spike k), it raises ``DAEMON_UNAVAILABLE``
  (retryable, with the hint to run ``narration-admin daemon start`` in a terminal) and nothing runs: a
  daemon inside a client's kill-on-close job would die with the client mid-job;
- ``running_daemon`` reads ``run/daemon.json`` and checks its pid (``sweep.daemon_alive``). Post a
  ``stop`` only when it says a daemon runs (``narration-admin daemon stop``, WP37): a daemon honours only
  the stops posted after it was launched (``service``, "Which stops a daemon honours"), so a stop posted
  with none running stops nothing, and the next daemon answers it ``stopped: false``;
- ``ensure_daemon`` starts one unless one runs, and can wait until it has written its status;
- ``start_detached`` records each launch the platform let run in ``run/launch.json`` (the pid it got back and
  the launch time; a refused start records nothing), and ``check_launch`` reads it. A daemon launched less
  than ``START_WINDOW_S`` ago that has not written a status since is ``starting`` while its process runs, and
  ``failed`` once that process has gone. The front-end asks for no other daemon while one is starting, and
  reports a failed start rather than launching again at once (WP36: ``get_job`` and ``cancel_job`` on a job no
  daemon serves). The file is the service's own operational state, like ``run/daemon.json``, and records
  nothing of a caller (sections 0.2, 2). It is advice, read and written without a lock: two front-ends that
  poll in the same instant may both launch, and the singleton makes that harmless (the second daemon exits
  quietly, or waits and takes over).

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

import json
import logging
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from narration.contracts.interfaces import Store
from narration.contracts.models import DaemonStatus
from narration.platform import ProcessPlatform
from narration.store import files
from narration.store.layout import LAUNCH_JSON, RUN
from narration.store.store import parse_iso, utc_iso

from .settings import isolated, scrub_python_env
from .sweep import StatusUnreadable, daemon_alive, launch_alive, read_status

log = logging.getLogger(__name__)

DAEMON_MODULE: Final = "narration.daemon"
RUNNING_STATES: Final = ("idle", "busy")
"""States of a daemon that serves the queue (``stopping`` is on its way out; ``stopped`` is gone)."""
START_WINDOW_S: Final = 90.0
"""How long after a launch a daemon that has written no status yet counts as starting, or as failed once its
process has gone (``check_launch``).

A launched daemon writes its status only once it holds the store's singleton. One launched while another is
exiting first waits for the singleton, up to ``DaemonSettings.takeover_wait_s`` (60 s), and before that a new
interpreter starts and imports the service. So the window is that 60 s wait plus 30 s for the start and the
imports (BELIEVE: not measured; a follow-up measures launch-to-status time). Past the window, a daemon that has
still written nothing is taken to have failed or stuck, and one more may be launched: a stuck or failing
start costs at most one launch per window."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Launch:
    """A daemon launch as ``run/launch.json`` records it: the pid ``spawn_detached`` got back (a launcher's,
    under a venv) and the launch time in Unix seconds (the clock ``--launched-at`` passes)."""

    pid: int
    launched_at: float


@dataclass(frozen=True, slots=True, kw_only=True)
class LaunchCheck:
    """What became of the last launch while it is recent (``check_launch``): still ``starting``, or ``failed``."""

    launch: Launch
    state: Literal["starting", "failed"]


def launch_path(store_root: Path) -> Path:
    """``<store_root>/run/launch.json`` (``StoreLayout.launch_json_path``, built from the root alone)."""
    return Path(os.path.abspath(store_root)) / RUN / LAUNCH_JSON


def record_launch(store_root: Path, *, pid: int, launched_at: float) -> None:
    """Record a launch in ``run/launch.json`` (a temporary file, then a rename). It is advice to other
    launchers, never needed for a start: a file that cannot be written is logged, not raised."""
    path = launch_path(store_root)
    data = json.dumps({"pid": pid, "launched_at": utc_iso(launched_at)}).encode("utf-8")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        files.write_atomic(path, data, readonly=False, durable=False)
    except OSError:
        log.warning("could not record the daemon's launch in %s", path, exc_info=True)


def read_launch(store_root: Path) -> Launch | None:
    """The last launch ``run/launch.json`` records, or None when there is none or it cannot be read."""
    try:
        data = json.loads(files.read_retrying(launch_path(store_root)).decode("utf-8"))
        return Launch(pid=int(data["pid"]), launched_at=parse_iso(str(data["launched_at"])))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError, KeyError):
        log.warning("run/launch.json cannot be read; taken as no launch", exc_info=True)
        return None


def check_launch(store: Store, *, now: float | None = None) -> LaunchCheck | None:
    """What became of the last launch, while it was made less than ``START_WINDOW_S`` before ``now`` (a stamp up
    to the window in the future, from a clock stepped back, counts as now; one further out is not trusted):

    - None: no launch in the window, or a daemon status written since it (``run/daemon.json``'s ``started_at``
      is later): the daemon started, and whether it still runs is ``running_daemon``'s to say;
    - ``starting``: no status since, and the launched process still runs (``sweep.launch_alive``);
    - ``failed``: no status since, and the launched process has gone. It died before it held the singleton (an
      import or configuration error, say), or failed through its ``finally``, which writes ``stopped`` with no
      start time, or gave up waiting for a daemon that was exiting (``takeover_wait_s``). Its log has why.

    A ``stopped`` status newer than the launch does not decide it: it may be the exiting daemon's, written while
    the launched one waits for the singleton, so the launched process decides.
    """
    launch = read_launch(store.root)
    if launch is None:
        return None
    age = (time.time() if now is None else now) - launch.launched_at
    if not -START_WINDOW_S < age < START_WINDOW_S:
        return None
    try:
        status = read_status(store)
    except StatusUnreadable:
        status = None
    if status is not None and status.started_at is not None:
        try:
            if parse_iso(status.started_at) >= launch.launched_at:
                return None
        except ValueError:
            pass
    state: Literal["starting", "failed"] = "starting" if launch_alive(launch.pid, launch.launched_at) else "failed"
    return LaunchCheck(launch=launch, state=state)


def launch_in_progress(store: Store, *, now: float | None = None) -> Launch | None:
    """The last launch while that daemon is still starting (``check_launch`` says ``starting``), else None."""
    check = check_launch(store, now=now)
    return check.launch if check is not None and check.state == "starting" else None


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
    while it starts up is for it. Raises ``NarrationError(DAEMON_UNAVAILABLE)`` when the daemon cannot leave
    this process's Job Objects (breakaway refused, or the daemon left in an enclosing job and ended before it
    ran), and never falls back to a daemon that is not detached; on an OS v1 does not support,
    ``UnsupportedPlatform``.

    The launch is recorded in ``run/launch.json`` (``record_launch``) only once ``spawn_detached`` has returned,
    that is, for a daemon the platform let run (on Windows, resumed once it was found in no Job Object). A
    refused start records nothing, so it holds back no later launch (``launch_in_progress``).
    """
    if platform is None:
        from narration.platform import get_platform

        platform = get_platform()
    root = Path(os.path.abspath(store_root))
    root.mkdir(parents=True, exist_ok=True)
    interpreter = platform.python_for(Path(sys.executable) if python is None else python, console=False)
    launched_at = time.time()
    launched = ("--launched-at", repr(launched_at))
    argv = daemon_argv(root, Path(os.path.abspath(config_path)), python=interpreter, extra=(*extra, *launched))
    pid = platform.spawn_detached(argv, cwd=root, env=scrub_python_env(os.environ if env is None else env))
    record_launch(root, pid=pid, launched_at=launched_at)
    return pid


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
