"""A stand-in for the store's database connection that makes write transactions fail where the tests need
them to, while the store's own ``db.write_txn`` runs unchanged.

A test swaps it in with ``scripted(monkeypatch, store, ...)``: the store asks ``_conn()`` for its connection
each time it runs a statement, so every statement of this thread then passes through it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

import pytest

from narration.store import NarrationStore


class ScriptedConnection:
    """Passes every call to a real connection, except as scripted.

    * ``fail_commit``: the COMMIT of every write transaction (``BEGIN IMMEDIATE``) fails with a disk error.
      With ``sqlite_ends_it``, SQLite has ended the transaction by then, as it does after some I/O errors:
      the write lock is released before the store sees the error. ``meanwhile`` then runs, as another
      writer would in that moment.
    * ``look``: its result is recorded in ``seen`` just before each ROLLBACK, while the write lock is held.

    ``statements`` lists the first words of every statement, and ``COMMIT failed`` for a failed COMMIT.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        fail_commit: bool,
        sqlite_ends_it: bool,
        meanwhile: Callable[[], None] | None,
        look: Callable[[], object] | None,
    ) -> None:
        self._real = conn
        self._fail_commit, self._sqlite_ends_it, self._meanwhile, self._look = (
            fail_commit,
            sqlite_ends_it,
            meanwhile,
            look,
        )
        self._immediate = False
        self.seen: list[object] = []
        self.statements: list[str] = []

    def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
        verb = " ".join(sql.split()[:2]).upper()
        if verb.startswith("BEGIN"):
            self._immediate = verb == "BEGIN IMMEDIATE"
        if verb == "COMMIT" and self._immediate and self._fail_commit:
            self.statements.append("COMMIT failed")
            if self._sqlite_ends_it:
                self._real.execute("ROLLBACK")
                if self._meanwhile is not None:
                    self._meanwhile()
            raise sqlite3.OperationalError("disk I/O error")
        if verb == "ROLLBACK" and self._look is not None:
            self.seen.append(self._look())
        self.statements.append(verb.split()[0] if not verb.startswith("BEGIN") else verb)
        return self._real.execute(sql, parameters)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def scripted(
    monkeypatch: pytest.MonkeyPatch,
    store: NarrationStore,
    *,
    fail_commit: bool = False,
    sqlite_ends_it: bool = False,
    meanwhile: Callable[[], None] | None = None,
    look: Callable[[], object] | None = None,
) -> ScriptedConnection:
    """Make ``store`` run its statements through a ``ScriptedConnection``; ``monkeypatch.undo()`` ends it."""
    connection = ScriptedConnection(
        store._conn(), fail_commit=fail_commit, sqlite_ends_it=sqlite_ends_it, meanwhile=meanwhile, look=look
    )
    monkeypatch.setattr(store, "_conn", lambda: connection)
    return connection
