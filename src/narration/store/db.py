"""The store's SQLite database: connection settings, the schema and its migrations (design section 15).

One file, ``narration.sqlite``, in WAL mode, shared by every front-end and the daemon. Writes take the
database's write lock at once (``BEGIN IMMEDIATE``), so a check and the write that depends on it are one
atomic step across processes. ``schema_migrations`` records each migration applied; ``PRAGMA
user_version`` mirrors the latest. A store made by a newer version of the service is refused, never
downgraded.

Records are kept as JSON (``record`` columns) beside the columns that are searched on. Paths inside them
are relative to the store root.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final


class StoreSchemaError(RuntimeError):
    """The database was made by a newer version of the service, or could not be put in WAL mode."""


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    sql: str


_V1: Final = """
CREATE TABLE jobs (
    seq             INTEGER PRIMARY KEY AUTOINCREMENT,  -- FIFO order within a priority
    job_id          TEXT NOT NULL UNIQUE,
    kind            TEXT NOT NULL,
    status          TEXT NOT NULL,
    priority_rank   INTEGER NOT NULL,                   -- 0 interactive, 1 batch
    request_sha256  TEXT NOT NULL,
    idempotency_key TEXT,
    claimed_by      TEXT,
    inserted_at     REAL NOT NULL,                      -- Unix seconds, for the submit rate
    last_used_at    REAL NOT NULL,                      -- Unix seconds, for retention
    updated_at      TEXT NOT NULL,
    record          TEXT NOT NULL
);
CREATE INDEX jobs_by_queue ON jobs (status, priority_rank, seq);
CREATE INDEX jobs_by_request ON jobs (kind, request_sha256, status);
CREATE INDEX jobs_by_idempotency_key ON jobs (kind, idempotency_key, status);
CREATE INDEX jobs_by_inserted_at ON jobs (inserted_at);

-- The queue is the active jobs, by priority then FIFO (section 4, GPU scheduler item 3).
CREATE VIEW queue AS
    SELECT job_id, kind, status, priority_rank, seq FROM jobs
    WHERE status IN ('queued', 'running', 'cancelling')
    ORDER BY priority_rank, seq;

-- The three cache layers (section 10.2).
CREATE TABLE renders (
    render_key   TEXT PRIMARY KEY,
    render_id    TEXT NOT NULL UNIQUE,
    rel_dir      TEXT NOT NULL,
    record       TEXT NOT NULL,
    created_at   REAL NOT NULL,
    last_used_at REAL NOT NULL
);
CREATE TABLE takes (
    delivery_key TEXT PRIMARY KEY,
    take_id      TEXT NOT NULL UNIQUE,
    render_id    TEXT NOT NULL,
    rel_dir      TEXT NOT NULL,
    record       TEXT NOT NULL,
    created_at   REAL NOT NULL,
    last_used_at REAL NOT NULL
);
CREATE TABLE analyses (
    analysis_key TEXT PRIMARY KEY,
    analysis_id  TEXT NOT NULL UNIQUE,
    take_id      TEXT NOT NULL,
    rel_path     TEXT NOT NULL,
    record       TEXT NOT NULL,
    created_at   REAL NOT NULL,
    last_used_at REAL NOT NULL
);
CREATE INDEX analyses_by_take ON analyses (take_id);

-- Measurements: one per (voice, engine profile); kept longer (measurement_retention_days).
CREATE TABLE measurements (
    voice_hash        TEXT NOT NULL,
    engine_profile_id TEXT NOT NULL,
    measurement_key   TEXT NOT NULL UNIQUE,
    rel_dir           TEXT NOT NULL,
    record            TEXT NOT NULL,
    created_at        REAL NOT NULL,
    last_used_at      REAL NOT NULL,
    PRIMARY KEY (voice_hash, engine_profile_id)
);

-- Voice profiles: one per audio file (the latest profile version replaces an older one).
CREATE TABLE profiles (
    audio_sha256    TEXT PRIMARY KEY,
    profile_version TEXT NOT NULL,
    rel_dir         TEXT NOT NULL,
    record          TEXT NOT NULL,
    created_at      REAL NOT NULL,
    last_used_at    REAL NOT NULL
);

-- Designed candidates (section 3.1).
CREATE TABLE candidates (
    design_id    TEXT NOT NULL,
    idx          INTEGER NOT NULL,
    clip_sha256  TEXT NOT NULL,
    rel_dir      TEXT NOT NULL,
    record       TEXT NOT NULL,
    created_at   REAL NOT NULL,
    last_used_at REAL NOT NULL,
    PRIMARY KEY (design_id, idx)
);

-- The index of provenance.jsonl (section 17.4): append-only, never pruned.
CREATE TABLE provenance (
    clip_sha256 TEXT NOT NULL,
    design_id   TEXT NOT NULL,
    date        TEXT NOT NULL,
    PRIMARY KEY (clip_sha256, design_id)
);

-- Every immutable file the store published, with its hash, for verify (section 15).
CREATE TABLE files (
    rel_path   TEXT PRIMARY KEY,
    sha256     TEXT NOT NULL,
    size       INTEGER NOT NULL,
    owner_kind TEXT NOT NULL,
    owner_id   TEXT NOT NULL
);
CREATE INDEX files_by_owner ON files (owner_kind, owner_id);

-- One producer per key (section 4 item 6).
CREATE TABLE leases (
    key         TEXT PRIMARY KEY,
    holder      TEXT NOT NULL,
    token       TEXT NOT NULL,
    acquired_at REAL NOT NULL,
    expires_at  REAL NOT NULL
);

-- The service's own records: engine profiles and alignment benchmarks, never collected.
CREATE TABLE engine_profiles (
    engine_profile_id TEXT PRIMARY KEY,
    hash              TEXT NOT NULL,
    rel_path          TEXT NOT NULL,
    record            TEXT NOT NULL,
    updated_at        REAL NOT NULL
);
CREATE TABLE alignment_benchmarks (
    method_id  TEXT PRIMARY KEY,
    rel_path   TEXT NOT NULL,
    record     TEXT NOT NULL,
    updated_at REAL NOT NULL
);

-- Small named values: which engine profile and benchmark are current.
CREATE TABLE settings (
    name  TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Operator and front-end requests to the daemon (release_gpu, stop, stop --now).
CREATE TABLE commands (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    command_id   TEXT NOT NULL UNIQUE,
    kind         TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    done_at      TEXT,
    result       TEXT
);
"""

MIGRATIONS: Final = (Migration(1, "initial schema", _V1),)
SCHEMA_VERSION: Final = MIGRATIONS[-1].version
BUSY_TIMEOUT_MS: Final = 30_000


def connect(path: Path) -> sqlite3.Connection:
    """Open the database with the store's settings: WAL, full sync, a 30 s busy timeout, manual
    transactions (``isolation_level=None``), rows by name."""
    conn = sqlite3.connect(str(path), timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None, check_same_thread=False)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        if str(mode).lower() != "wal":
            raise StoreSchemaError(f"the store's database could not be put in WAL mode (it is in {mode!r} mode)")
        conn.execute("PRAGMA synchronous = FULL")
        conn.execute("PRAGMA foreign_keys = ON")
    except BaseException:
        conn.close()
        raise
    return conn


@contextmanager
def write_txn(conn: sqlite3.Connection, *, undo: Callable[[], None] | None = None) -> Iterator[sqlite3.Connection]:
    """A write transaction that holds the database's write lock from its first statement.

    If the transaction does not commit (its body fails, or the COMMIT itself does), it is rolled back
    before the error is raised, so the write lock is never left held. ``undo`` puts back what the caller
    changed outside the database (``store._Renames``), and runs while the write lock is held, just before
    that ROLLBACK: no other writer ever sees the rows rolled back while the files are not yet put back. If
    SQLite has already ended the transaction itself (it does after some I/O errors), the lock is taken
    again for ``undo``; another writer may have run in between, so ``undo`` must put back only what is
    still its own. If the lock cannot be taken again, ``undo`` runs anyway.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        _abandon(conn, undo)
        raise
    try:
        conn.execute("COMMIT")
    except BaseException:
        _abandon(conn, undo)
        raise


def _abandon(conn: sqlite3.Connection, undo: Callable[[], None] | None) -> None:
    """Undo and roll back a write transaction that will not commit, holding the write lock throughout."""
    try:
        if undo is not None:
            if not conn.in_transaction:
                with contextlib.suppress(sqlite3.Error):
                    conn.execute("BEGIN IMMEDIATE")
            undo()
    finally:
        if conn.in_transaction:
            with contextlib.suppress(sqlite3.Error):
                conn.execute("ROLLBACK")


@contextmanager
def read_txn(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """A read transaction: one consistent snapshot for several statements."""
    conn.execute("BEGIN")
    try:
        yield conn
    finally:
        conn.execute("COMMIT")


def migrate(conn: sqlite3.Connection) -> int:
    """Apply every migration the database lacks; returns the schema version now in force."""
    with write_txn(conn):
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)"
        )
        current = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()[0]
        if current > SCHEMA_VERSION:
            raise StoreSchemaError(
                f"this store was made by a newer version of the service (schema {current}; this version knows "
                f"up to {SCHEMA_VERSION}); use that version, or point store_root at another folder"
            )
        for migration in MIGRATIONS:
            if migration.version <= current:
                continue
            for statement in _statements(migration.sql):
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                (migration.version, migration.name, dt.datetime.now(dt.UTC).isoformat(timespec="seconds")),
            )
            current = migration.version
        conn.execute(f"PRAGMA user_version = {int(current)}")
    return current


def _statements(sql: str) -> list[str]:
    """Split a migration into statements (``executescript`` would commit the open transaction)."""
    out: list[str] = []
    buffer = ""
    for line in sql.splitlines(keepends=True):
        buffer += line.split("--", 1)[0] + ("\n" if "--" in line else "")
        if sqlite3.complete_statement(buffer):
            if buffer.strip():
                out.append(buffer.strip())
            buffer = ""
    if buffer.strip():
        raise ValueError(f"incomplete SQL statement in a migration: {buffer.strip()[:80]}")
    return out
