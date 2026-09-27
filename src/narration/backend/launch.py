"""Starting the daemon from the front-end, and asking whether one runs (design sections 4, 4.1, 16).

The front-end never runs a job itself: it commits the job to the store, then makes sure a daemon serves the
store (``[daemon] autostart``). **The order matters** (WP30's seam, "The front-end's order"): the job is
committed first, and the daemon's status read after, so either a daemon about to exit sees the job, or this
read sees it ``stopping`` and starts another.

``DetachedLauncher`` uses WP30's ``narration.daemon.start``: ``ensure_daemon`` starts ``python -m
narration.daemon`` detached unless a daemon already serves the store, and ``running_daemon`` reads
``run/daemon.json`` and checks its pid. A start the platform refuses (breakaway forbidden by the client's Job
Object) is ``DAEMON_UNAVAILABLE``, retryable, with the hint to run ``narration-admin daemon start`` in a
terminal; a daemon that is not detached is never started, since it would die with its client mid-job.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Final, Protocol

from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.interfaces import Store
from narration.contracts.models import DaemonStatus

log = logging.getLogger(__name__)

DAEMON_RETRY_S: Final = 30.0
"""``DAEMON_UNAVAILABLE``'s ``retry_after_s`` (DC-2): time for an operator to start the daemon by hand."""
START_MODULE: Final = "narration.daemon.start"
SERVING_STATES: Final = ("idle", "busy")
"""The daemon states that serve the queue (``stopping`` is on its way out)."""
START_HINT: Final = (
    "Run 'narration-admin daemon start' in a terminal, then send the identical request again; it is "
    "deduplicated, and the job already queued is kept."
)


class DaemonLauncher(Protocol):
    """How the front-end reads the daemon's status and starts it (see the module docstring)."""

    def running(self, store: Store) -> DaemonStatus | None:
        """The status of the daemon that serves this store, or None when none runs."""
        ...

    def ensure(self, store: Store) -> None:
        """Start a daemon unless one serves the store. Call it only after the job is committed. Raises
        ``NarrationError(DAEMON_UNAVAILABLE)`` (retryable, with ``retry_after_s``) when it cannot."""
        ...


def _start_module() -> ModuleType | None:
    """WP30's ``narration.daemon.start``, or None in a build without the daemon."""
    try:
        return importlib.import_module(START_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name is not None and START_MODULE.startswith(exc.name):
            return None
        raise


def unavailable(message: str, *, details: dict[str, Any] | None = None) -> NarrationError:
    """``DAEMON_UNAVAILABLE`` with the hint to start the daemon by hand and DC-2's ``retry_after_s``."""
    return NarrationError(
        codes.DAEMON_UNAVAILABLE,
        message,
        hint=START_HINT,
        details=details,
        retryable=True,
        retry_after_s=DAEMON_RETRY_S,
    )


class DetachedLauncher:
    """Starts the daemon detached through WP30's ``narration.daemon.start`` (see the module docstring).

    ``config_path`` is the configuration file the daemon is started with (the front-end's own). With
    ``autostart`` False (``[daemon] autostart``), ``ensure`` starts nothing: an operator runs the daemon.
    ``extra`` is appended to the daemon's command line (for tests: ``--fake-workers``, ``--runner``).
    """

    def __init__(self, config_path: Path | None, *, autostart: bool = True, extra: Sequence[str] = ()) -> None:
        self._config_path = config_path
        self._autostart = autostart
        self._extra = tuple(extra)

    def running(self, store: Store) -> DaemonStatus | None:
        """The daemon that serves this store (``running_daemon``: its status file and a live pid), or None."""
        start = _start_module()
        if start is None:
            return None
        status: DaemonStatus | None = start.running_daemon(store)
        return status

    def ensure(self, store: Store) -> None:
        """Start a daemon unless one serves the store (``ensure_daemon``); never waits for it to serve."""
        if not self._autostart:
            return
        if self._config_path is None:
            raise unavailable("the front-end was started without a configuration file, so it cannot start the daemon")
        start = _start_module()
        if start is None:
            raise unavailable("this build of the service has no daemon to start")
        try:
            result = start.ensure_daemon(store, self._config_path, extra=self._extra)
        except NarrationError as exc:
            if exc.code != codes.DAEMON_UNAVAILABLE or not exc.retryable:
                raise  # e.g. UnsupportedPlatform: no retry helps, and its hint says why
            raise unavailable(exc.message, details=exc.details) from exc
        except OSError as exc:
            log.exception("the daemon could not be started")
            raise unavailable(f"the daemon could not be started ({exc.strerror or type(exc).__name__})") from exc
        if getattr(result, "started", False):
            log.info("started the daemon (pid %s)", getattr(result, "spawned_pid", None))


def serving(status: DaemonStatus | None) -> bool:
    """Whether a daemon status says it serves the queue."""
    return status is not None and status.state in SERVING_STATES


__all__ = ["DAEMON_RETRY_S", "START_HINT", "DaemonLauncher", "DetachedLauncher", "serving", "unavailable"]
