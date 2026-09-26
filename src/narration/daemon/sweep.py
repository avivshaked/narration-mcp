"""What a daemon does on start about the daemon before it (design section 4.1: process identity, stop).

**Why it is safe.** The sweep runs only while this daemon holds its store's singleton, so no other daemon
of the store runs: every job left ``running`` was left by a daemon that is gone. Such a job goes back to
the queue, and one left ``cancelling`` is finished as ``cancelled``, both by compare-and-set
(``seam.return_job``). The leases that daemon held need nothing: they lapse at their TTL, and this daemon,
claiming under the same holder name (``seam.DAEMON_HOLDER``), may take those keys again at once. Its
workers need nothing either: they were in its kill-on-close group, which Windows closed when it died.

**The pid.** The singleton gives no signal that its last holder died without releasing it (WP19), so the
previous ``run/daemon.json`` tells a clean stop (``stopped``) from a crash (any other state, and its pid
gone). ``daemon_alive`` reads only whether that pid exists and when that process started, so a pid Windows
has since given to another program is not taken for the daemon. No other process is ever inspected,
signalled or killed.

**Stale commands.** A ``stop`` or ``stop_now`` posted before this daemon started was meant for a daemon
that is gone; it is completed without stopping this one. ``release_gpu`` is always answered normally.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

import psutil

from narration.contracts.interfaces import Store
from narration.contracts.models import DaemonStatus
from narration.store.layout import StorePathError
from narration.store.store import parse_iso

from .seam import return_job

log = logging.getLogger(__name__)

START_TOLERANCE_S: Final = 2.0
"""How much later than the recorded ``started_at`` a process may have been created and still be the daemon
(the daemon records ``started_at`` after its process started, so a real one is never later)."""
EXIT_WAIT_S: Final = 10.0
"""How long the sweep waits for a previous daemon that is still exiting."""

READ_ATTEMPTS: Final = 10
READ_PAUSE_S: Final = 0.02


def read_status(
    store: Store,
    *,
    attempts: int = READ_ATTEMPTS,
    pause_s: float = READ_PAUSE_S,
    sleep: Callable[[float], None] = time.sleep,
) -> DaemonStatus | None:
    """``run/daemon.json`` as ``Store.get_daemon_status`` reads it, read again after a short pause when the
    read fails the way it can while the daemon renames a new file over it.

    KNOW (WP30, Windows): a reader that opens the file during that rename gets ``PermissionError`` (about
    one read in twenty in a tight loop), and ``os.path.realpath``, which the store's path check uses, can
    return a ``\\\\?\\``-prefixed path for a file replaced during the call, which the check takes for a path
    outside the store (``StorePathError``). Both pass within milliseconds. Any other error, or either one
    ``attempts`` times running, is raised.
    """
    for attempt in range(1, attempts + 1):
        try:
            return store.get_daemon_status()
        except (PermissionError, StorePathError):
            if attempt >= attempts:
                raise
            sleep(pause_s)
    raise ValueError(f"attempts must be at least 1, not {attempts}")


PreviousDaemon = Literal["none", "clean", "died", "exiting"]
"""What the previous ``run/daemon.json`` says: none was written, it stopped cleanly, it died, or its process
still runs (it released the singleton and is exiting)."""


def daemon_alive(status: DaemonStatus | None) -> bool:
    """Whether the daemon ``status`` describes still runs: its pid exists, and that process was created no
    later than the status's ``started_at`` (so it is not a later process that got the same pid).

    A process whose creation time cannot be read counts as not confirmed, so False: a caller that then
    starts a daemon starts one that exits quietly if one runs after all.
    """
    if status is None or status.state == "stopped" or status.pid is None or status.started_at is None:
        return False
    try:
        created = psutil.Process(status.pid).create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False
    return created <= parse_iso(status.started_at) + START_TOLERANCE_S


@dataclass(frozen=True, slots=True, kw_only=True)
class SweepReport:
    """What the sweep found and did."""

    previous: PreviousDaemon
    previous_pid: int | None
    requeued: tuple[str, ...]
    cancelled: tuple[str, ...]
    stale_commands: tuple[str, ...]


def sweep(
    store: Store,
    previous: DaemonStatus | None,
    *,
    started_at: str,
    alive: Callable[[DaemonStatus | None], bool] = daemon_alive,
    exit_wait_s: float = EXIT_WAIT_S,
    sleep: Callable[[float], None] = time.sleep,
) -> SweepReport:
    """Clean up after the previous daemon (see the module docstring). Call it holding the singleton, with
    the ``run/daemon.json`` read before this daemon wrote its own, and this daemon's ``started_at``."""
    kind: PreviousDaemon
    if previous is None:
        kind = "none"
    elif previous.state == "stopped":
        kind = "clean"
    elif alive(previous):
        kind = "exiting"
        deadline = time.monotonic() + exit_wait_s
        while alive(previous) and time.monotonic() < deadline:
            sleep(0.1)
        if alive(previous):
            log.warning(
                "the previous daemon (pid %s) still runs but no longer holds the singleton; sweeping anyway",
                previous.pid,
            )
    else:
        kind = "died"
    reason = (
        f"the daemon that ran it (pid {previous.pid}) stopped without finishing it"
        if previous is not None
        else "no daemon was running it"
    )
    requeued: list[str] = []
    cancelled: list[str] = []
    for job in store.queued_jobs():
        if job.status not in ("running", "cancelling"):
            continue
        changed = return_job(store, job.job_id, reason=reason)
        if changed is not None:
            (requeued if changed.status == "queued" else cancelled).append(job.job_id)
    stale: list[str] = []
    since = parse_iso(started_at)
    for command in store.pending_commands():
        if command.kind in ("stop", "stop_now") and parse_iso(command.requested_at) < since:
            store.complete_command(
                command.command_id,
                {"stopped": False, "reason": "it was posted before this daemon started, for a daemon that is gone"},
            )
            stale.append(command.command_id)
    report = SweepReport(
        previous=kind,
        previous_pid=previous.pid if previous is not None else None,
        requeued=tuple(requeued),
        cancelled=tuple(cancelled),
        stale_commands=tuple(stale),
    )
    if kind == "died" or requeued or cancelled or stale:
        log.warning(
            "previous daemon: %s (pid %s); jobs queued again: %s; cancels finished: %s; stale commands: %s",
            kind,
            report.previous_pid,
            ", ".join(requeued) or "none",
            ", ".join(cancelled) or "none",
            ", ".join(stale) or "none",
        )
    return report
