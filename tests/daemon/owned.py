"""The daemon processes a test (or spike g) may end: only ones it can prove it started (AGENTS.md hard rule 7).

A pid alone proves nothing: once a process exits, Windows may give its pid to any other program. So a test
never kills by pid. It captures ``psutil.Process`` objects at a moment it can prove who they are, and later
acts only through them: psutil keeps each one's creation time and refuses (``NoSuchProcess``) to signal a
pid that another process has taken since.

The proof for a daemon (``capture_daemon``):

- the process now at the status's ``pid`` was created no later than the status's ``started_at``. The daemon
  wrote that status while it ran, holding that pid; a process that holds the pid now and existed then must
  be that daemon, since two live processes never share a pid. A later process given the same pid was
  created after ``started_at`` and is refused. ``started_at`` is written to the millisecond, cut short, so
  "no later than" allows the millisecond the cut removed (``STARTED_AT_PRECISION_S``), and no more. A real
  daemon reads that clock long after its creation. A test that writes a status must also give a time the
  clock reached after the creation (``test_owned.a_time_after_creation``): on Windows ``time.time`` can lag
  the exact ``create_time`` by a timer tick;
- its parent is the launcher pid the test's session got back from ``start_detached`` (or it is that pid
  itself). psutil's ``parent()`` refuses a parent younger than the child, so a reused parent pid is refused.

``OwnedDaemon.identity`` and ``reattach`` carry that proof across processes as (pid, creation time) pairs:
a later ``reattach`` accepts a pid only if its creation time is exactly the one recorded.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Mapping
from dataclasses import dataclass

import psutil

from narration.contracts.models import DaemonStatus
from narration.store import NarrationStore
from narration.store.store import parse_iso

KILL_WAIT_S = 15.0

STARTED_AT_PRECISION_S = 0.001
"""``started_at`` is written with milliseconds, cut short (``utc_iso``): the moment it names may be up to this
much earlier than the moment the daemon read its clock."""


@dataclass(frozen=True)
class OwnedDaemon:
    """A daemon and its launcher, proven ours. ``launcher`` is None when the daemon is the process that was
    started; after ``reattach`` either may be None, when that process has gone."""

    daemon: psutil.Process | None
    launcher: psutil.Process | None

    def roots(self) -> list[psutil.Process]:
        return [p for p in (self.daemon, self.launcher) if p is not None]

    def running(self) -> bool:
        """Whether the daemon or its launcher still runs (False for a pid that another process has taken)."""
        return any(p.is_running() for p in self.roots())

    def identity(self) -> dict[str, float | int | None]:
        """The pids and exact creation times, for ``reattach`` in another process."""
        return {
            "daemon_pid": self.daemon.pid if self.daemon is not None else None,
            "daemon_created": self.daemon.create_time() if self.daemon is not None else None,
            "launcher_pid": self.launcher.pid if self.launcher is not None else None,
            "launcher_created": self.launcher.create_time() if self.launcher is not None else None,
        }

    def kill(self) -> None:
        """Kill the daemon, its launcher and their descendants, through these objects only."""
        members: list[psutil.Process] = []
        for root in self.roots():
            try:
                members += [root, *root.children(recursive=True)]  # refuses a root whose pid was reused
            except psutil.NoSuchProcess:
                continue
        for member in members:
            with contextlib.suppress(psutil.NoSuchProcess):
                member.kill()
        psutil.wait_procs(members, timeout=KILL_WAIT_S)


def capture_daemon(status: DaemonStatus, launcher_pid: int) -> OwnedDaemon | None:
    """The daemon ``status`` describes, if it is provably the one started as ``launcher_pid`` (see the module
    docstring); None otherwise, and then nothing may be killed."""
    if status.pid is None or status.started_at is None:
        return None
    try:
        daemon = psutil.Process(status.pid)
        if daemon.create_time() >= parse_iso(status.started_at) + STARTED_AT_PRECISION_S:
            return None  # a later process that was given the daemon's pid
        if status.pid == launcher_pid:
            return OwnedDaemon(daemon=daemon, launcher=None)
        parent = daemon.parent()
    except psutil.Error:
        return None
    if parent is None or parent.pid != launcher_pid:
        return None
    return OwnedDaemon(daemon=daemon, launcher=parent)


def _exactly(pid: int, created: float) -> psutil.Process | None:
    try:
        process = psutil.Process(pid)
        return process if process.create_time() == created else None
    except psutil.Error:
        return None


def reattach(identity: Mapping[str, float | int | None]) -> OwnedDaemon | None:
    """The processes ``OwnedDaemon.identity`` recorded, each only if its pid still names a process created at
    exactly the recorded time; None when both are gone."""
    daemon_pid, daemon_created = identity.get("daemon_pid"), identity.get("daemon_created")
    daemon = (
        _exactly(int(daemon_pid), float(daemon_created))
        if daemon_pid is not None and daemon_created is not None
        else None
    )
    launcher_pid, launcher_created = identity.get("launcher_pid"), identity.get("launcher_created")
    launcher = (
        _exactly(int(launcher_pid), float(launcher_created))
        if launcher_pid is not None and launcher_created is not None
        else None
    )
    if daemon is None and launcher is None:
        return None
    return OwnedDaemon(daemon=daemon, launcher=launcher)


def stop_detached(store: NarrationStore, owned: OwnedDaemon | None, *, wait_s: float = 30.0) -> None:
    """Stop a detached daemon a test started: ``stop`` through the store, then, if it still runs after
    ``wait_s``, a kill through ``owned``. With ``owned`` None (it could not be proven ours) nothing is killed."""
    posted = store.post_command("stop")
    store.wait_for_command(posted.command_id, timeout_s=wait_s)
    if owned is None:
        return
    deadline = time.monotonic() + wait_s
    while owned.running() and time.monotonic() < deadline:
        time.sleep(0.1)
    if owned.running():
        owned.kill()
