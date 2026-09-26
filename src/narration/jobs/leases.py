"""A lease kept alive while its work runs (design section 4 item 6).

A key's lease tells every other holder that the key is being made, so they wait for it rather than make it
twice. The work under a lease can take minutes (a long render, a cold load, a slow scoring). ``kept`` renews
the lease on a thread of its own while the work runs, so the lease's TTL only bounds how long it outlives a
daemon that died, and never cuts short work that is still going.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from narration.contracts.interfaces import Lease
from narration.store import LeaseLostError

log = logging.getLogger(__name__)


@contextmanager
def kept(lease: Lease, *, ttl_s: float, every_s: float) -> Iterator[None]:
    """Renew ``lease`` to ``ttl_s`` every ``every_s`` seconds until the block ends.

    A lease another holder has taken meanwhile (``LeaseLostError``: it lapsed first) is not renewed again,
    and the work goes on: the store publishes whichever result comes first. Any other failure to renew is
    logged and tried again at the next interval.
    """
    stop = threading.Event()

    def renew() -> None:
        while not stop.wait(every_s):
            try:
                lease.renew(ttl_s)
            except LeaseLostError as exc:
                log.warning("the lease on %s was lost while its work ran: %s", lease.key, exc)
                return
            except Exception:  # a store hiccup: the next interval tries again
                log.warning("the lease on %s could not be renewed", lease.key, exc_info=True)

    keeper = threading.Thread(target=renew, name=f"lease {lease.key[:24]}", daemon=True)
    keeper.start()
    try:
        yield
    finally:
        stop.set()
        keeper.join(timeout=every_s + 5.0)


__all__ = ["kept"]
