"""Stand-ins for ``db.write_txn`` that make a transaction fail where the tests need it to.

A test swaps one in with ``monkeypatch.setattr(store_db, "write_txn", …)``: the store looks the function up
on the module each time it opens a write transaction.
"""

from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Callable, Iterator


@contextlib.contextmanager
def commit_fails(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """A write transaction whose body runs, and whose COMMIT then fails (a disk error, say)."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("ROLLBACK")
    raise sqlite3.OperationalError("disk I/O error")


def observing_rollback(
    look: Callable[[], object], seen: list[object]
) -> Callable[[sqlite3.Connection], contextlib.AbstractContextManager[sqlite3.Connection]]:
    """A write transaction like ``db.write_txn`` that, when its body fails, records ``look()`` in ``seen``
    just before the ROLLBACK: while the write lock is still held."""

    @contextlib.contextmanager
    def write_txn(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            seen.append(look())
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    return write_txn
