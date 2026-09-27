"""Leases kept alive while their work runs (design section 4 item 6).

A key's lease tells every other holder that the key is being made, so they wait for it rather than make it
twice. The work under a lease can take minutes (a long render, a cold load, a slow scoring). A
``LeaseKeeper`` renews the lease while the work runs, so the lease's TTL only bounds how long it outlives a
daemon that died, and never cuts short work that is still going.

One keeper serves one engine, and renews every lease the engine holds on **one** thread, for the engine's
life. The store opens a database connection for each thread that uses it and keeps it until the store
closes, so a thread per piece of work would leave a connection behind for every piece.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final

from narration.contracts.interfaces import Lease
from narration.store import LeaseLostError

log = logging.getLogger(__name__)

END_WAIT_S: Final = 30.0
"""How long a block's end waits for a renewal of its lease already under way (a store write) to finish."""
THREAD_NAME: Final = "narration lease keeper"


@dataclass(eq=False, slots=True)
class _Kept:
    lease: Lease
    ttl_s: float
    every_s: float
    due: float
    lost: bool = field(default=False)


class LeaseKeeper:
    """Renews the leases of one engine's work while it runs, all on one thread (see the module docstring).

    ``kept`` registers a lease for the length of a ``with`` block. The thread starts at the first lease and
    then waits, idle, for the next; ``close`` stops it, and a later ``kept`` starts it again.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._live: list[_Kept] = []
        self._renewing: _Kept | None = None
        self._thread: threading.Thread | None = None
        self._stop: threading.Event | None = None

    @contextmanager
    def kept(self, lease: Lease, *, ttl_s: float, every_s: float) -> Iterator[None]:
        """Renew ``lease`` to ``ttl_s`` every ``every_s`` seconds until the block ends.

        A lease another holder has taken meanwhile (``LeaseLostError``: it lapsed first) is not renewed again,
        and the work goes on: the store publishes whichever result comes first. Any other failure to renew is
        logged and tried again at the next interval. The lease is never renewed after the block ends: the end
        waits for a renewal already under way.
        """
        entry = _Kept(lease, ttl_s, every_s, time.monotonic() + every_s)
        with self._cond:
            self._live.append(entry)
            self._start()
            self._cond.notify_all()
        try:
            yield
        finally:
            with self._cond:
                self._live.remove(entry)
                deadline = time.monotonic() + END_WAIT_S
                while self._renewing is entry and time.monotonic() < deadline:
                    self._cond.wait(timeout=max(0.0, deadline - time.monotonic()))
                if self._renewing is entry:
                    log.warning("a renewal of the lease on %s was still under way as its work ended", lease.key)

    def close(self, *, timeout_s: float = 10.0) -> None:
        """Stop the thread (after a renewal under way). Leases still kept are no longer renewed."""
        with self._cond:
            thread, stop = self._thread, self._stop
            self._thread = self._stop = None
            if stop is not None:
                stop.set()
            self._cond.notify_all()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout_s)

    @property
    def running(self) -> bool:
        """Whether the keeper's thread is running."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    # ------------------------------------------------------------------ the thread
    def _start(self) -> None:
        """Start the thread if it is not running. Called with the lock held."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(self._stop,), name=THREAD_NAME, daemon=True)
        self._thread.start()

    def _run(self, stop: threading.Event) -> None:
        while True:
            with self._cond:
                entry = self._next_due(stop)
                if entry is None:
                    return
                entry.due = time.monotonic() + entry.every_s
                self._renewing = entry
            try:
                entry.lease.renew(entry.ttl_s)
            except LeaseLostError as exc:
                entry.lost = True
                log.warning("the lease on %s was lost while its work ran: %s", entry.lease.key, exc)
            except Exception:  # a store hiccup: the next interval tries again
                log.warning("the lease on %s could not be renewed", entry.lease.key, exc_info=True)
            finally:
                with self._cond:
                    self._renewing = None
                    self._cond.notify_all()

    def _next_due(self, stop: threading.Event) -> _Kept | None:
        """Wait for the next lease due for renewal, or None once stopped. Called with the lock held."""
        while not stop.is_set():
            live = [e for e in self._live if not e.lost]
            now = time.monotonic()
            due = min(live, key=lambda e: e.due, default=None)
            if due is not None and due.due <= now:
                return due
            self._cond.wait(timeout=None if due is None else due.due - now)
        return None


__all__ = ["END_WAIT_S", "THREAD_NAME", "LeaseKeeper"]
