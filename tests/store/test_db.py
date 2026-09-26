"""The store's database: WAL mode, the schema version and its migrations (design section 15)."""

from __future__ import annotations

from pathlib import Path

import pytest

from narration.store import db


def test_the_database_is_in_wal_mode_s15(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "narration.sqlite")
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == db.BUSY_TIMEOUT_MS
    finally:
        conn.close()


def test_migrations_are_recorded_and_idempotent(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "narration.sqlite")
    try:
        assert db.migrate(conn) == db.SCHEMA_VERSION
        assert db.migrate(conn) == db.SCHEMA_VERSION
        rows = conn.execute("SELECT version, name FROM schema_migrations ORDER BY version").fetchall()
        assert [tuple(r) for r in rows] == [(m.version, m.name) for m in db.MIGRATIONS]
        assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}
        assert {"jobs", "queue", "renders", "takes", "analyses", "measurements", "profiles", "candidates"} <= tables
        assert {"provenance", "files", "leases", "engine_profiles", "alignment_benchmarks", "commands"} <= tables
    finally:
        conn.close()


def test_a_store_from_a_newer_version_is_refused(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "narration.sqlite")
    try:
        db.migrate(conn)
        conn.execute(
            "INSERT INTO schema_migrations VALUES (?, 'from the future', '2030-01-01')", (db.SCHEMA_VERSION + 1,)
        )
        with pytest.raises(db.StoreSchemaError, match="newer version"):
            db.migrate(conn)
    finally:
        conn.close()


def test_a_failed_write_transaction_rolls_back(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "narration.sqlite")
    try:
        db.migrate(conn)
        with pytest.raises(RuntimeError), db.write_txn(conn):
            conn.execute("INSERT INTO settings VALUES ('a', 'b')")
            raise RuntimeError("abort")
        assert conn.execute("SELECT COUNT(*) FROM settings").fetchone()[0] == 0
    finally:
        conn.close()
