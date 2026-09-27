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

**An unreadable status.** ``run/daemon.json`` is written through a temp file and a rename, but not flushed
to disk first, so a crash or a power loss can leave it empty or cut short. ``read_status`` then raises
``StatusUnreadable``, and every reader treats that as "no daemon to speak of": the sweep as a daemon that
died (``sweep(..., previous_unreadable=True)``), a daemon that finds the singleton held as a holder that
has not said it serves (it waits, as for one that is exiting), and ``start.running_daemon`` as no daemon.

**Commands.** The sweep leaves every pending command alone: the daemon answers each one, and judges a
pending ``stop`` by when it was posted (``service``, "Which stops a daemon honours").
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
from narration.store.store import parse_iso

from .seam import return_job

log = logging.getLogger(__name__)

START_TOLERANCE_S: Final = 2.0
"""How much later than the recorded ``started_at`` a process may have been created and still be the daemon
(the daemon records ``started_at`` after its process started, so a real one is never later)."""
EXIT_WAIT_S: Final = 10.0
"""How long the sweep waits for a previous daemon that is still exiting."""


class StatusUnreadable(Exception):
    """``run/daemon.json`` exists but is not a status: empty, cut short, not JSON, or not a status's shape."""


def read_status(store: Store) -> DaemonStatus | None:
    """``run/daemon.json`` as ``Store.get_daemon_status`` reads it; ``StatusUnreadable`` for a torn file.

    The store copes with the daemon renaming a new file over it (WP12's follow-ups, from spike g): it reads
    again while Windows refuses to open the file during the rename (``files.read_retrying``), and its path
    check strips the ``\\\\?\\`` prefix ``realpath`` can leave on such a file (``platform.real_path``). What is
    left here is a file that is there but is not a status: empty, cut short, not JSON, not UTF-8, or not a
    status's shape (see the module docstring). Any other error is raised as it is.
    """
    try:
        return store.get_daemon_status()
    except (ValueError, TypeError) as exc:  # JSON, UTF-8 and contract errors are ValueErrors
        raise StatusUnreadable(f"run/daemon.json cannot be read as a status: {exc}") from exc


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


def sweep(
    store: Store,
    previous: DaemonStatus | None,
    *,
    started_at: str,
    previous_unreadable: bool = False,
    alive: Callable[[DaemonStatus | None], bool] = daemon_alive,
    exit_wait_s: float = EXIT_WAIT_S,
    sleep: Callable[[float], None] = time.sleep,
) -> SweepReport:
    """Clean up after the previous daemon (see the module docstring). Call it holding the singleton, with
    the ``run/daemon.json`` read before this daemon wrote its own (``previous_unreadable`` when it was there
    but could not be read), and this daemon's ``started_at``."""
    kind: PreviousDaemon
    if previous_unreadable:
        kind = "died"
    elif previous is None:
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
    if previous is not None:
        reason = f"the daemon that ran it (pid {previous.pid}) stopped without finishing it"
    elif previous_unreadable:
        reason = "the daemon that ran it stopped without finishing it (its run/daemon.json could not be read)"
    else:
        reason = "no daemon was running it"
    requeued: list[str] = []
    cancelled: list[str] = []
    for job in store.queued_jobs():
        if job.status not in ("running", "cancelling"):
            continue
        changed = return_job(store, job.job_id, reason=reason)
        if changed is not None:
            (requeued if changed.status == "queued" else cancelled).append(job.job_id)
    report = SweepReport(
        previous=kind,
        previous_pid=previous.pid if previous is not None else None,
        requeued=tuple(requeued),
        cancelled=tuple(cancelled),
    )
    if kind == "died" or requeued or cancelled:
        log.warning(
            "previous daemon: %s (pid %s); jobs queued again: %s; cancels finished: %s",
            kind,
            report.previous_pid,
            ", ".join(requeued) or "none",
            ", ".join(cancelled) or "none",
        )
    return report
