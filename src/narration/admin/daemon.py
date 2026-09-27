"""``narration-admin daemon start | stop [--now] | status`` (design sections 4, 4.1 and 7.1).

- ``start`` starts the daemon detached (``narration.daemon.start.ensure_daemon``), unless one already
  serves this store, and waits until it serves. When Windows refuses to detach it from this terminal (the
  terminal's own job forbids breakaway), it says so and offers ``--foreground``, which runs the daemon in
  this terminal until it exits.
- ``stop`` asks the running daemon to stop: to finish the segment in flight, unload and exit (``stop``); or,
  with ``--now``, to end the work in flight at once and put it back on the queue (``stop_now``). **It posts
  the command only when ``running_daemon`` says a daemon runs.** A daemon honours only the stops posted
  after it was launched, so a stop posted with none running would stop nothing, and the next daemon would
  answer it ``stopped: false`` (``narration.daemon.service``, "Which stops a daemon honours"). Then it waits
  for the daemon's answer.
- ``status`` shows ``run/daemon.json`` and whether its daemon still runs; ``--json`` prints it as JSON.

Queued jobs survive a stop; the next start resumes them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Final

from narration.contracts.errors import NarrationError, UnsupportedPlatform
from narration.contracts.models import DaemonCommand, DaemonStatus
from narration.contracts.serial import to_json
from narration.daemon.__main__ import LOG_NAME
from narration.daemon.settings import scrub_python_env
from narration.daemon.start import daemon_argv, ensure_daemon, running_daemon
from narration.daemon.sweep import StatusUnreadable, daemon_alive, read_status
from narration.store import NarrationStore

from .cli import EXIT_FAILED, EXIT_OK, PROGRAM, Admin, AdminError, Subparsers

START_WAIT_S: Final = 30.0
"""How long ``daemon start`` waits, by default, for the daemon to say it serves."""
STOP_WAIT_S: Final = 300.0
"""How long ``daemon stop`` waits, by default: the segment in flight is finished first (a render and its QA)."""
STOP_NOW_WAIT_S: Final = 60.0
"""How long ``daemon stop --now`` waits, by default."""
POLL_S: Final = 0.2
FOREGROUND_EXIT_WAIT_S: Final = 30.0
"""After Ctrl+C, how long ``start --foreground`` waits for the daemon (which got the Ctrl+C too) to exit."""


def register(subparsers: Subparsers) -> None:
    """Add ``daemon start | stop | status`` (``narration.admin.cli``)."""
    parser = subparsers.add_parser(
        "daemon",
        help="start, stop or show the daemon (section 4.1)",
        description="Start, stop or show the daemon that runs this store's jobs (design section 4.1).",
    )
    commands = parser.add_subparsers(title="daemon commands", metavar="<daemon command>", required=True)

    start = commands.add_parser(
        "start",
        help="start the daemon detached, unless one serves already",
        description=(
            "Start the daemon detached, so it outlives this terminal, unless one serves this store already; "
            "then wait until it serves. Queued jobs resume."
        ),
    )
    start.add_argument(
        "--wait",
        type=float,
        default=START_WAIT_S,
        metavar="S",
        help=f"seconds to wait for it to serve (default {START_WAIT_S:g}; 0: do not wait)",
    )
    start.add_argument(
        "--foreground",
        action="store_true",
        help="run the daemon in this terminal until it exits, instead of detached (Ctrl+C stops it at once)",
    )
    start.set_defaults(handler=start_daemon)

    stop = commands.add_parser(
        "stop",
        help="stop the running daemon",
        description=(
            "Ask the running daemon to stop: it finishes the segment in flight, unloads and exits. With --now it "
            "ends the work in flight at once and puts it back on the queue. Nothing is posted when no daemon "
            "runs. Queued jobs resume on the next start."
        ),
    )
    stop.add_argument("--now", action="store_true", help="end the work in flight at once (it is re-queued)")
    stop.add_argument(
        "--wait",
        type=float,
        default=None,
        metavar="S",
        help=f"seconds to wait for it to stop (default {STOP_WAIT_S:g}, or {STOP_NOW_WAIT_S:g} with --now; 0: "
        "do not wait)",
    )
    stop.set_defaults(handler=stop_daemon)

    status = commands.add_parser(
        "status",
        help="show whether the daemon runs, and what it is doing",
        description="Show run/daemon.json and whether its daemon still runs. Exit code 0 if one runs, else 1.",
    )
    status.add_argument("--json", action="store_true", help="print the status as JSON")
    status.set_defaults(handler=show_status)


# ---------------------------------------------------------------- start
def start_daemon(admin: Admin, args: argparse.Namespace) -> int:
    """``daemon start``: see the module docstring."""
    config = admin.config()
    root = config.server.store_root
    store = admin.store()
    if args.foreground:
        return _run_in_foreground(admin, store)
    config_path = admin.config_path()
    wait_s = max(0.0, float(args.wait))
    try:
        result = ensure_daemon(store, config_path, platform=admin.platform(), wait_s=wait_s)
    except UnsupportedPlatform:
        raise
    except NarrationError as exc:
        if exc.code != "DAEMON_UNAVAILABLE":
            raise
        raise AdminError(
            f"Windows refused to start the daemon detached from this terminal ({exc.message}). This terminal "
            "runs inside a job that forbids breakaway, as some programs' built-in terminals do. Run "
            f"`{PROGRAM} daemon start` from a terminal outside that program, or run the daemon in this one with "
            f"`{PROGRAM} daemon start --foreground` and keep the terminal open while it works."
        ) from exc
    if not result.started:
        admin.say(f"A daemon already serves {root}: {_describe(result.status)}.")
        return EXIT_OK
    if wait_s == 0:
        admin.say(
            f"Started a daemon for {root} (launcher pid {result.spawned_pid}). It serves once it has written "
            f"its status; check with `{PROGRAM} daemon status`."
        )
        return EXIT_OK
    if result.status is None:
        admin.warn(
            f"{PROGRAM}: started a daemon for {root} (launcher pid {result.spawned_pid}), but it has not said it "
            f"serves within {wait_s:g} s. Its log is {_log_path(store)}; check with `{PROGRAM} daemon status`."
        )
        return EXIT_FAILED
    admin.say(f"Started the daemon for {root}: {_describe(result.status)}.")
    return EXIT_OK


def foreground_argv(store: NarrationStore, config_path: Path, *, python: Path, launched_at: float) -> list[str]:
    """The command line of a daemon run in this terminal: the detached daemon's (``-P -m narration.daemon
    --store <store_root> --config <path>``), launched now."""
    return daemon_argv(store.root, config_path, python=python, extra=("--launched-at", repr(launched_at)))


def _run_in_foreground(admin: Admin, store: NarrationStore) -> int:
    status = running_daemon(store)
    if status is not None and status.state in ("idle", "busy"):
        admin.say(f"A daemon already serves {store.root}: {_describe(status)}.")
        return EXIT_OK
    argv = foreground_argv(store, admin.config_path(), python=Path(sys.executable), launched_at=time.time())
    admin.say(
        f"Running the daemon for {store.root} in this terminal. It exits when idle ([daemon] idle_exit_min) or "
        f"on `{PROGRAM} daemon stop`; Ctrl+C stops it at once, and the job in flight goes back to the queue."
    )
    process = subprocess.Popen(argv, cwd=store.root, env=scrub_python_env(os.environ))
    try:
        return process.wait()
    except KeyboardInterrupt:
        # The daemon shares this console, so it got the Ctrl+C too and is exiting; this is our own child.
        try:
            process.wait(timeout=FOREGROUND_EXIT_WAIT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise


# ---------------------------------------------------------------- stop
def stop_daemon(admin: Admin, args: argparse.Namespace) -> int:
    """``daemon stop``: see the module docstring."""
    root = admin.config().server.store_root
    if not admin.store_exists():
        admin.say(f"No daemon runs for {root} (no store there yet); nothing to stop.")
        return EXIT_OK
    store = admin.store()
    status = running_daemon(store)
    if status is None:
        admin.say(f"No daemon runs for {root}; nothing to stop, so no stop was posted.")
        return EXIT_OK
    kind = "stop_now" if args.now else "stop"
    command = store.post_command(kind)
    how = (
        "now: the work in flight is ended and goes back to the queue"
        if args.now
        else "after the segment in flight, then unload and exit"
    )
    admin.say(f"Asked the daemon ({_describe(status)}) to stop {how}.")
    wait_s = (STOP_NOW_WAIT_S if args.now else STOP_WAIT_S) if args.wait is None else max(0.0, float(args.wait))
    if wait_s == 0:
        return EXIT_OK
    done, gone = _wait_for_stop(store, command, wait_s)
    if done is not None:
        result: dict[str, Any] = dict(done.result or {})
        if result.get("stopped") is True:
            requeued = result.get("requeued") or []
            again = (
                f" {len(requeued)} job(s) went back to the queue; they resume on the next start." if requeued else ""
            )
            admin.say(f"The daemon stopped.{again}")
            return EXIT_OK
        admin.warn(f"{PROGRAM}: the daemon did not take the stop as its own: {result.get('reason', result)}.")
        return EXIT_FAILED
    if gone:
        admin.say("The daemon exited before it read the stop (it was already on its way out). None runs now.")
        return EXIT_OK
    now = running_daemon(store)
    admin.warn(
        f"{PROGRAM}: the daemon has not stopped within {wait_s:g} s ({_describe(now)}). "
        + (
            f"It finishes the segment in flight first. Wait and check with `{PROGRAM} daemon status`, or stop it at "
            f"once with `{PROGRAM} daemon stop --now`."
            if not args.now
            else f"Check its log, {_log_path(store)}, and `{PROGRAM} daemon status`."
        )
    )
    return EXIT_FAILED


def _wait_for_stop(store: NarrationStore, command: DaemonCommand, wait_s: float) -> tuple[DaemonCommand | None, bool]:
    """The completed command, or (None, True) when the daemon went without completing it, or (None, False)
    at the timeout."""
    deadline = time.monotonic() + wait_s
    while True:
        done = store.wait_for_command(command.command_id, timeout_s=min(POLL_S, max(0.0, deadline - time.monotonic())))
        if done is not None:
            return done, False
        if running_daemon(store) is None:
            # It may have completed the command as it exited: look once more before saying it did not.
            done = store.wait_for_command(command.command_id, timeout_s=0)
            return (done, False) if done is not None else (None, True)
        if time.monotonic() >= deadline:
            return None, False


# ---------------------------------------------------------------- status
def show_status(admin: Admin, args: argparse.Namespace) -> int:
    """``daemon status``: see the module docstring."""
    root = admin.config().server.store_root
    status: DaemonStatus | None = None
    unreadable: str | None = None
    if admin.store_exists():
        try:
            status = read_status(admin.store())
        except StatusUnreadable as exc:
            unreadable = str(exc)
    running = daemon_alive(status)
    if args.json:
        payload: dict[str, Any] = {
            "store_root": str(root),
            "running": running,
            "status": to_json(status) if status is not None else None,
            "unreadable": unreadable,
        }
        admin.say(json.dumps(payload, ensure_ascii=False, indent=2))
        return EXIT_OK if running else EXIT_FAILED
    if running and status is not None:
        admin.say(f"The daemon for {root} runs: {_describe(status)}.")
        for line in _details(status):
            admin.say(f"  {line}")
        return EXIT_OK
    if unreadable is not None:
        admin.say(f"No daemon is known to run for {root}: {unreadable}. `{PROGRAM} daemon start` starts one.")
    elif status is None:
        admin.say(f"No daemon runs for {root} (none has written a status). `{PROGRAM} daemon start` starts one.")
    else:
        admin.say(
            f"No daemon runs for {root}. The last one said {status.state!r} at {status.updated_at}"
            + ("" if status.state == "stopped" else " and then went without saying it stopped (see its log)")
            + f". `{PROGRAM} daemon start` starts one."
        )
    return EXIT_FAILED


def _describe(status: DaemonStatus | None) -> str:
    if status is None:
        return "its status is not known"
    return f"pid {status.pid}, {status.state} since {status.updated_at}, started {status.started_at}"


def _details(status: DaemonStatus) -> list[str]:
    lines: list[str] = []
    job = status.current_job
    if job is not None:
        label = f" ({job.label})" if job.label else ""
        lines.append(f"job: {job.job_id}{label}, {job.kind}, phase {job.phase}, since {job.started_at}")
    for worker in status.workers:
        lines.append(f"worker: {worker.role}, pid {worker.pid}")
    gpu = status.gpu
    held = f"loaded: {gpu.holder}" if gpu.in_use else "no model loaded"
    unload = f", unloads in {gpu.unload_in_s:.0f} s" if gpu.in_use and gpu.unload_in_s is not None else ""
    free = f", {gpu.free_mb} of {gpu.total_mb} MB free" if gpu.free_mb is not None and gpu.total_mb is not None else ""
    lines.append(f"gpu: {gpu.name or 'unknown'}{free}; {held}{unload}")
    if status.est_drain_s is not None:
        lines.append(f"queue: about {status.est_drain_s:.0f} s of work")
    return lines


def _log_path(store: NarrationStore) -> Path:
    return store.layout.logs_dir() / LOG_NAME
