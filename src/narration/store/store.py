"""The store (design sections 4, 6, 15 and 17.2; plan.md WP12): SQLite rows plus content-addressed files.

**Stateless (sections 0.2 and 2).** The store holds caches of work done (renders, takes, analyses,
profiles, designed candidates), the job queue, and the service's own records (engine profiles, the canary,
alignment benchmarks, the provenance list, measurements of voices). It never holds a caller's script as a
record of theirs, a choice of take, an approval, a pronunciation list or a voice: a job keeps its request
by value only for the retention period, and a clip copied into ``scratch/`` is a working copy.

**Nothing is published half written.** A published folder is assembled under a ``.staging-`` name and
renamed into place whole; a published file is written under a ``.tmp-`` name and renamed. The rename and
the index row that makes it visible happen in one database write transaction, which takes the write lock at
once, so every process sees a key as either absent or complete. Immutable files are read-only.

**One producer per key (section 4 item 6).** ``claim`` gives a lease on a key to one holder; others get
``in_flight`` and can ``wait_for`` it; a lease past its time can be claimed again; and the claim checks the
cache first, in the same transaction.

**Retention (section 15).** Every item has a last-used time (``touch``); ``gc`` removes what is older than
the retention period, and is a dry run unless told otherwise. It never removes provenance, engine profiles,
the canary, alignment benchmarks, or an active job. A live item keeps what it names alive with it:

- A measurement keeps every take named in its calibration or ladder, with the take's render and analyses,
  for ``measurement_retention_days``.
- A design keeps its candidates' voice profiles, and using a design (``touch("design", …)``) uses them
  too.

**Where measurement audio lives (a deviation from section 15's folder layout).** Section 15 draws
``measurements/<voice_hash>/<engine_profile_id>/`` with ``calibration/`` and ``ladder/`` folders. Here a
measurement's calibration and ladder takes stay where every take lives, ``takes/<ab>/tk_…/``, and the
measurement's folder holds ``measurement.json`` only. The measurement names its takes by id, and ``gc``
keeps those takes as long as the measurement (above), so section 15's promise holds: measurement audio is
kept for ``measurement_retention_days``. Copying each take into the measurement's folder would store the
same audio twice under two retention rules.

**Concurrent collection.** ``gc`` works in short write transactions, a bounded batch of items each. In
each one it renames every victim to a ``.trash-`` sibling and deletes its rows, and only after the commit
does it remove those trash names, outside the lock. A key published again meanwhile goes to a fresh folder
that ``gc`` never touches, and no writer waits on ``gc`` for more than a few seconds.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path, PurePath
from typing import Any, Final, Literal, TypeVar, get_args

from narration import keys
from narration.config import Config, RetentionConfig
from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.interfaces import ClaimResult, Platform
from narration.contracts.models import (
    AlignmentBenchmark,
    AnalysisRecord,
    AudioRef,
    Candidate,
    DaemonCommand,
    DaemonStatus,
    EngineProfile,
    JobRecord,
    MeasurementRecord,
    ProfilePictures,
    ProfileRecord,
    ProvenanceEntry,
    RenderRecord,
    TakeRecord,
)
from narration.contracts.names import DaemonCommandKind, EngineKind, JobStatus
from narration.contracts.serial import ContractError, from_json, to_json
from narration.keys.ulid import UlidGenerator
from narration.platform import real_path

from . import db, files
from .layout import (
    ANALYSES,
    CANDIDATE_JSON,
    CLIP_WAV,
    DELIVERY_WAV,
    DESIGN_ID_PATTERN,
    DESIGNS,
    JOB_JSON,
    JOBS,
    MEASUREMENT_JSON,
    MEASUREMENTS,
    PITCH_PNG,
    PROFILE_JSON,
    PROFILES,
    RAW_WAV,
    RENDER_JSON,
    RENDERS,
    SCRATCH,
    SPECTROGRAM_PNG,
    TAKE_JSON,
    TAKES,
    TEMP_PREFIXES,
    InvalidIdError,
    StoreLayout,
    StorePathError,
    method_file_stem,
)
from .records import (
    map_candidate,
    map_engine_profile,
    map_profile,
    map_render,
    map_take,
    row_json,
    sidecar_bytes,
)

RetentionKind = Literal["render", "take", "analysis", "measurement", "profile", "design", "job"]
"""What ``touch`` accepts, with the id each takes: ``render_id``, ``take_id``, ``analysis_id``, the
``measurement_key``, the audio's sha256, ``design_id``, ``job_id``."""

_Decision = Literal["keep", "new", "replace"]
_R = TypeVar("_R")
_T = TypeVar("_T")


class _Reread:
    """What a reader's row check returns when the row changed while it looked (``NarrationStore._drop``)."""


_REREAD: Final = _Reread()
_READ_ATTEMPTS: Final = 5
"""How many times a reader reads a row again that concurrent publishes keep replacing (``_read_one``)."""

_PRIORITY_RANK: Final = {"interactive": 0, "batch": 1}
_ACTIVE: Final = ("queued", "running", "cancelling")
_FIXED_JOB_FIELDS: Final = frozenset(
    {"schema", "job_id", "kind", "request", "request_sha256", "idempotency_key", "created_at"}
)
_GC_KINDS: Final = {
    "renders": "render",
    "takes": "take",
    "analyses": "analysis",
    "profiles": "profile",
    "designs": "design",
    "measurements": "measurement",
    "jobs": "job",
}
_STORE_TIME: Final = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z")
"""A time as the store writes it (``utc_iso``): fixed-width UTC to the millisecond, so text order is time
order."""
_DROP_ROWS: Final = {
    "render": ("renders", "render_id"),
    "take": ("takes", "take_id"),
    "analysis": ("analyses", "analysis_id"),
    "measurement": ("measurements", "measurement_key"),
    "profile": ("profiles", "audio_sha256"),
}
"""For ``_drop``: the table and id column of each kind whose row is dropped when its file is gone."""
_ENGINE_UNHASHED: Final = frozenset({"hash", "snapshot_dir", "observed", "tier", "canary"})
"""The ``EngineProfile`` fields its ``hash`` leaves out; every other field is pinned with the id."""
DC6_HINT: Final = "Use a new key for a different request."
_log = logging.getLogger(__name__)
_DAY: Final = 86_400.0
DEFAULT_GRACE_S: Final = _DAY
"""How old a temporary, staging or unindexed entry must be before ``gc`` treats it as left over by a crash
(a publish takes seconds, so a day never races one)."""
GC_BATCH_ITEMS = 200
"""The most items one ``gc`` write transaction examines (collected or not)."""
GC_BATCH_SECONDS = 2.0
"""About how long one ``gc`` write transaction may run before it commits and lets other writers in: far
below the busy timeout (``db.BUSY_TIMEOUT_MS``) every other writer waits for."""


class StoreError(RuntimeError):
    """A store invariant does not hold: a bug in the caller, or damage to the store."""


class StoreIntegrityError(StoreError):
    """A file's bytes do not match the hash its record states, or a pinned record would change."""


class LeaseLostError(StoreError):
    """The lease was taken over by another holder after it lapsed; the work must not be published as ours."""


class NotFoundError(NarrationError):
    """A job, command or engine profile the store does not have (``NOT_FOUND``)."""

    def __init__(self, what: str, ident: str) -> None:
        super().__init__(codes.NOT_FOUND, f"no {what} {ident!r} in the store")


def utc_iso(seconds: float) -> str:
    """ISO 8601 UTC with milliseconds and a ``Z``, as every record's times are written."""
    return dt.datetime.fromtimestamp(seconds, dt.UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso(text: str) -> float:
    """Unix seconds of an ISO 8601 time; a time without a zone is taken as UTC."""
    value = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.UTC)
    return value.timestamp()


class SqliteLease:
    """A lease on one key (``interfaces.Lease``). Renewing or releasing acts only while this lease's token
    is the one on record, so a holder that lost its lease can never cut short the next holder's."""

    def __init__(self, store: NarrationStore, key: str, holder: str, token: str) -> None:
        self._store = store
        self._key = key
        self.holder = holder
        self.token = token

    @property
    def key(self) -> str:
        return self._key

    def renew(self, ttl_s: float) -> None:
        """Extend the lease to ``ttl_s`` from now; raises ``LeaseLostError`` if another holder has it."""
        _check_ttl(ttl_s)
        with self._store._write() as conn:
            cur = conn.execute(
                "UPDATE leases SET expires_at = ? WHERE key = ? AND token = ?",
                (self._store._clock() + ttl_s, self._key, self.token),
            )
            if cur.rowcount == 0:
                raise LeaseLostError(f"the lease on {self._key} is no longer held by {self.holder!r}")

    def release(self) -> None:
        """Give the key up (idempotent)."""
        with self._store._write() as conn:
            conn.execute("DELETE FROM leases WHERE key = ? AND token = ?", (self._key, self.token))

    def __enter__(self) -> SqliteLease:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def _check_ttl(ttl_s: float) -> None:
    if isinstance(ttl_s, bool) or not isinstance(ttl_s, (int, float)) or not math.isfinite(ttl_s) or ttl_s <= 0:
        raise ValueError(f"ttl_s must be a positive number of seconds, got {ttl_s!r}")


class NarrationStore:
    """The service's ``Store`` (``narration.contracts.interfaces``): see the module docstring.

    ``platform`` supplies the OS-specific path checks (``narration.platform``). ``retention`` sets what
    ``gc`` keeps. ``alignment_method_id`` is the method id of the configured aligner (model, pinned
    revision and snap parameters, section 11.2): ``current_alignment_benchmark`` returns its benchmark, and
    None when it is not given. ``clock`` (Unix seconds) can be replaced for tests. One instance may be used
    from many threads; each thread gets its own database connection. Call ``close`` when done.
    """

    def __init__(
        self,
        root: Path,
        platform: Platform,
        *,
        retention: RetentionConfig | None = None,
        alignment_method_id: str | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if alignment_method_id is not None:
            method_file_stem(alignment_method_id)  # checks it names a method
        self._layout = StoreLayout(root, platform)
        self._retention = retention or RetentionConfig()
        self._alignment_method_id = alignment_method_id
        self._clock: Callable[[], float] = clock or time.time
        self._ulids = UlidGenerator()
        self._local = threading.local()
        self._conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        self._closed = False
        db.migrate(self._conn())
        self._reconcile_provenance()

    @classmethod
    def from_config(
        cls, config: Config, platform: Platform, *, alignment_method_id: str | None = None
    ) -> NarrationStore:
        """The store at ``[server] store_root`` with ``[retention]``.

        ``alignment_method_id`` is the configured aligner's method id. The configuration names the aligner
        model (``[alignment] model``) but not its pinned revision or snap parameters, which the method id
        also covers, so the caller that pinned the aligner passes it here.
        """
        return cls(
            config.server.store_root,
            platform,
            retention=config.retention,
            alignment_method_id=alignment_method_id,
        )

    # ================================================================ plumbing
    def close(self) -> None:
        """Close every connection this instance opened."""
        with self._conns_lock:
            self._closed = True
            conns, self._conns = self._conns, []
        for conn in conns:
            conn.close()

    def __enter__(self) -> NarrationStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            with self._conns_lock:
                if self._closed:
                    raise StoreError("the store is closed")
                conn = db.connect(self._layout.db_path)
                self._conns.append(conn)
            self._local.conn = conn
        return conn

    def _write(
        self, *, undo: Callable[[], None] | None = None
    ) -> contextlib.AbstractContextManager[sqlite3.Connection]:
        return db.write_txn(self._conn(), undo=undo)

    def _read(self) -> contextlib.AbstractContextManager[sqlite3.Connection]:
        return db.read_txn(self._conn())

    def _now_iso(self) -> str:
        return utc_iso(self._clock())

    def _to_rel(self, path: str) -> str:
        if not path:
            return ""
        p = Path(path)
        if not p.is_absolute():
            raise StorePathError(f"expected an absolute path inside the store, got {path!r}")
        self._layout.confine(p)
        return self._layout.rel(p)

    def _to_abs(self, rel: str) -> str:
        return str(self._layout.abs(rel)) if rel else ""

    def _present(self, rel: str) -> bool:
        return os.path.isfile(self._layout.abs(rel))

    def _source(self, path: Path, what: str) -> Path:
        """A file the caller hands the store to move into place. It must be a regular file under
        ``scratch/``: moving it away is a write, and a published file must never be consumed or moved."""
        p = Path(os.path.abspath(path))
        if os.path.islink(p) or os.path.isjunction(p) or not p.is_file():
            raise StorePathError(f"{what} {p} is not a regular file")
        self._layout.confine(p)
        scratch = Path(real_path(self._layout.tree(SCRATCH)))
        if scratch not in Path(real_path(p)).parents:
            raise StorePathError(
                f"{what} {p} is not under the store's {SCRATCH} folder; the store moves in only files written "
                "there for it (scratch_path)"
            )
        return p

    def _consume(self, path: Path | None) -> None:
        """Remove a source file the store did not need (its key was already published). Only a file under
        ``scratch/`` is ever removed."""
        if path is not None:
            with contextlib.suppress(StorePathError):
                files.discard(self._source(path, "source"))

    @contextlib.contextmanager
    def _staging(self, final: Path) -> Iterator[_Staging]:
        """A ``.staging-`` folder beside ``final``, removed on the way out. If the publish fails, every
        file the caller handed in is moved back to where it came from first: a failed publish never
        destroys a render or a design (GPU work)."""
        final.parent.mkdir(parents=True, exist_ok=True)
        staging = _Staging(self._layout.confine(final.with_name(f".staging-{files.token()}-{final.name}")))
        staging.path.mkdir()
        all_returned = True
        try:
            yield staging
        except BaseException:
            all_returned = staging.return_sources()
            raise
        finally:
            if all_returned:
                files.remove_tree(staging.path)
            else:
                _log.error(
                    "kept %s: it holds files that could not be moved back to where the caller had them",
                    staging.path,
                )

    def _publish_dir(
        self,
        staging: Path,
        final: Path,
        decide: Callable[[sqlite3.Connection], _Decision],
        commit: Callable[[sqlite3.Connection], None],
    ) -> bool:
        """Rename a staged folder into place and index it, in one write transaction. Anything already at
        ``final`` (an older profile version, or a crash's leftover without an index row) is renamed aside
        first, and removed after the commit. If the transaction does not commit (a rename, the index write
        or the COMMIT itself fails), both renames are undone: the staged folder goes back to ``staging``
        and what was at ``final`` comes back, so a failed replace never loses the published folder."""
        renames = _Renames()
        trash: Path | None = None
        with self._write_undoing(renames) as conn:
            if decide(conn) == "keep":
                return False
            if os.path.lexists(final):
                trash = files.trash_name(final, self._clock())
                renames.rename(final, trash)
            renames.rename(staging, final)
            commit(conn)
        files.fsync_dir(final.parent)
        if trash is not None:
            _discard_trash(trash, final)
        return True

    def _publish_file(
        self,
        tmp: Path,
        final: Path,
        decide: Callable[[sqlite3.Connection], _Decision],
        commit: Callable[[sqlite3.Connection], None],
        *,
        source: Path | None = None,
    ) -> bool:
        """Rename a finished temporary file into place and index it, in one write transaction. A file
        already at ``final`` is renamed aside first, and removed after the commit. If the transaction does
        not commit, both renames are undone, so the old file stays published. ``final``'s folder is made
        under the lock, so ``tmp`` may wait elsewhere on the same volume (``put_measurement``). A reader that
        finds ``final`` missing meanwhile reads again once this commits (``_drop``).

        ``source`` is where the caller had ``tmp`` (a clip the store moved in). If the publish fails, ``tmp``
        goes back there, writable, instead of being discarded. If the key is already published (``keep``),
        ``tmp`` is a duplicate and is discarded.
        """
        renames = _Renames()
        trash: Path | None = None
        try:
            with self._write_undoing(renames) as conn:
                if decide(conn) == "keep":
                    files.discard(tmp)
                    return False
                if os.path.lexists(final):
                    trash = files.trash_name(final, self._clock())
                    renames.rename(final, trash)
                _make_folder(final.parent)  # under the lock: gc may have taken it
                renames.rename(tmp, final)
                commit(conn)
        except BaseException:
            _give_back(tmp, source)
            raise
        files.fsync_dir(final.parent)
        if trash is not None:
            _discard_trash(trash, final)
        return True

    def _write_undoing(self, renames: _Renames) -> contextlib.AbstractContextManager[sqlite3.Connection]:
        """A write transaction whose file renames are undone if it does not commit, whether its body or its
        COMMIT fails. They are undone under the write lock, before the ROLLBACK releases it
        (``db.write_txn``), so no other process ever sees a restored row whose file or folder is still
        under a ``.trash-`` name, or a rolled-back row next to the file that replaced its own."""
        return self._write(undo=renames.undo)

    def _index_files(
        self, conn: sqlite3.Connection, owner_kind: str, owner_id: str, entries: list[tuple[str, str, int]]
    ) -> None:
        conn.execute("DELETE FROM files WHERE owner_kind = ? AND owner_id = ?", (owner_kind, owner_id))
        conn.executemany(
            "INSERT OR REPLACE INTO files (rel_path, sha256, size, owner_kind, owner_id) VALUES (?, ?, ?, ?, ?)",
            [(rel, sha, size, owner_kind, owner_id) for rel, sha, size in entries],
        )

    @staticmethod
    def _check_hash(stated: str, actual: str, what: str) -> None:
        if stated and stated != actual:
            raise StoreIntegrityError(f"{what}: the record says sha256 {stated}, the file is {actual}")

    # ================================================================ layout (section 15)
    @property
    def root(self) -> Path:
        return self._layout.root

    @property
    def layout(self) -> StoreLayout:
        return self._layout

    def render_dir(self, render_id: str) -> Path:
        return self._layout.render_dir(render_id)

    def take_dir(self, take_id: str) -> Path:
        return self._layout.take_dir(take_id)

    def analysis_path(self, take_id: str, analysis_id: str) -> Path:
        return self._layout.analysis_path(take_id, analysis_id)

    def measurement_dir(self, voice_hash: str, engine_profile_id: str) -> Path:
        return self._layout.measurement_dir(voice_hash, engine_profile_id)

    def design_dir(self, design_id: str, index: int) -> Path:
        return self._layout.design_dir(design_id, index)

    def profile_dir(self, audio_sha256: str) -> Path:
        return self._layout.profile_dir(audio_sha256)

    def job_dir(self, job_id: str) -> Path:
        return self._layout.job_dir(job_id)

    def scratch_path(self, *parts: str) -> Path:
        """A path under ``scratch/`` for worker I/O; its parent folder exists."""
        return self._layout.scratch_path(*parts)

    # ================================================================ renders (the render layer)
    def get_render(self, render_key: str) -> RenderRecord | None:
        return self._read_one(self._fetch("SELECT record FROM renders WHERE render_key = ?", render_key), self._render)

    def get_render_by_id(self, render_id: str) -> RenderRecord | None:
        return self._read_one(self._fetch("SELECT record FROM renders WHERE render_id = ?", render_id), self._render)

    def _render(self, row: sqlite3.Row) -> RenderRecord | _Reread | None:
        record = from_json(RenderRecord, json.loads(row["record"]))
        if not self._present(record.raw.path):
            return None if self._drop("render", record.render_id, row["record"], record.raw.path) else _REREAD
        return map_render(record, self._to_abs)

    def put_render(self, record: RenderRecord, raw_audio: Path) -> RenderRecord:
        """Publish ``raw_audio`` (moved into place, read-only) and ``render.json``; returns the record with its
        path and the file's sha256. If the key is already published, that render is returned and
        ``raw_audio`` is discarded: the first published render of a key is the one kept."""
        if record.render_id != keys.render_id(record.render_key):
            raise StoreIntegrityError(f"render_id {record.render_id} does not name render_key {record.render_key}")
        existing = self.get_render(record.render_key)
        if existing is not None:
            self._consume(raw_audio)
            self.touch("render", record.render_id)
            return existing
        src = self._source(raw_audio, "raw_audio")
        sha, size = files.sha256_file(src)
        self._check_hash(record.raw.sha256, sha, f"render {record.render_id}")
        final = self._layout.render_dir(record.render_id)
        rel_dir = self._layout.rel(final)
        stored = dataclasses.replace(
            record, raw=dataclasses.replace(record.raw, path=f"{rel_dir}/{RAW_WAV}", sha256=sha)
        )
        now = self._clock()
        with self._staging(final) as staging:
            staging.move_in(src, RAW_WAV)
            jsha, jsize = files.write_atomic(staging.path / RENDER_JSON, sidecar_bytes(stored), readonly=True)

            def decide(conn: sqlite3.Connection) -> _Decision:
                found = conn.execute("SELECT 1 FROM renders WHERE render_key = ?", (record.render_key,)).fetchone()
                return "keep" if found else "new"

            def commit(conn: sqlite3.Connection) -> None:
                conn.execute(
                    "INSERT INTO renders (render_key, render_id, rel_dir, record, created_at, last_used_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (record.render_key, record.render_id, rel_dir, row_json(stored), now, now),
                )
                self._index_files(
                    conn,
                    "render",
                    record.render_id,
                    [(f"{rel_dir}/{RAW_WAV}", sha, size), (f"{rel_dir}/{RENDER_JSON}", jsha, jsize)],
                )

            self._publish_dir(staging.path, final, decide, commit)
        return self._must(self.get_render(record.render_key), "render", record.render_key)

    # ================================================================ takes (the delivery layer)
    def get_take(self, delivery_key: str) -> TakeRecord | None:
        return self._read_one(self._fetch("SELECT record FROM takes WHERE delivery_key = ?", delivery_key), self._take)

    def get_take_by_id(self, take_id: str) -> TakeRecord | None:
        return self._read_one(self._fetch("SELECT record FROM takes WHERE take_id = ?", take_id), self._take)

    def _take(self, row: sqlite3.Row) -> TakeRecord | _Reread | None:
        record = from_json(TakeRecord, json.loads(row["record"]))
        if not self._present(record.delivery.path):
            return None if self._drop("take", record.take_id, row["record"], record.delivery.path) else _REREAD
        return map_take(record, self._to_abs)

    def put_take(self, record: TakeRecord, delivery_audio: Path) -> TakeRecord:
        """Publish ``delivery_audio`` (moved into place, read-only) and ``take.json``; returns the record with
        its path and sha256. The first published take of a key is the one kept."""
        if record.take_id != keys.take_id(record.delivery_key):
            raise StoreIntegrityError(f"take_id {record.take_id} does not name delivery_key {record.delivery_key}")
        existing = self.get_take(record.delivery_key)
        if existing is not None:
            self._consume(delivery_audio)
            self.touch("take", record.take_id)
            return existing
        src = self._source(delivery_audio, "delivery_audio")
        sha, size = files.sha256_file(src)
        self._check_hash(record.delivery.sha256, sha, f"take {record.take_id}")
        final = self._layout.take_dir(record.take_id)
        rel_dir = self._layout.rel(final)
        stored = dataclasses.replace(
            record, delivery=dataclasses.replace(record.delivery, path=f"{rel_dir}/{DELIVERY_WAV}", sha256=sha)
        )
        now = self._clock()
        with self._staging(final) as staging:
            staging.move_in(src, DELIVERY_WAV)
            jsha, jsize = files.write_atomic(staging.path / TAKE_JSON, sidecar_bytes(stored), readonly=True)

            def decide(conn: sqlite3.Connection) -> _Decision:
                found = conn.execute("SELECT 1 FROM takes WHERE delivery_key = ?", (record.delivery_key,)).fetchone()
                return "keep" if found else "new"

            def commit(conn: sqlite3.Connection) -> None:
                conn.execute(
                    "INSERT INTO takes (delivery_key, take_id, render_id, rel_dir, record, created_at, last_used_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (record.delivery_key, record.take_id, record.render_id, rel_dir, row_json(stored), now, now),
                )
                self._index_files(
                    conn,
                    "take",
                    record.take_id,
                    [(f"{rel_dir}/{DELIVERY_WAV}", sha, size), (f"{rel_dir}/{TAKE_JSON}", jsha, jsize)],
                )

            self._publish_dir(staging.path, final, decide, commit)
        return self._must(self.get_take(record.delivery_key), "take", record.delivery_key)

    # ================================================================ analyses (the analysis layer)
    def get_analysis(self, analysis_key: str) -> AnalysisRecord | None:
        sql = "SELECT record, rel_path FROM analyses WHERE analysis_key = ?"
        return self._read_one(self._fetch(sql, analysis_key), self._analysis)

    def get_analysis_by_id(self, analysis_id: str) -> AnalysisRecord | None:
        sql = "SELECT record, rel_path FROM analyses WHERE analysis_id = ?"
        return self._read_one(self._fetch(sql, analysis_id), self._analysis)

    def analyses_of(self, take_id: str) -> tuple[AnalysisRecord, ...]:
        """Every analysis of a take, oldest first (``narration://takes/{take_id}``)."""
        sql = "SELECT record, rel_path FROM analyses WHERE take_id = ? ORDER BY created_at, analysis_id"
        return self._read_all(self._fetch_all(sql, take_id), self._analysis)

    def _analysis(self, row: sqlite3.Row) -> AnalysisRecord | _Reread | None:
        record = from_json(AnalysisRecord, json.loads(row["record"]))
        if not self._present(row["rel_path"]):
            return None if self._drop("analysis", record.analysis_id, row["record"], row["rel_path"]) else _REREAD
        return record

    def put_analysis(self, record: AnalysisRecord) -> AnalysisRecord:
        """Publish ``analyses/an_<16hex>.json`` inside its take's folder. The take must be published first."""
        if record.analysis_id != keys.analysis_id(record.analysis_key):
            raise StoreIntegrityError(
                f"analysis_id {record.analysis_id} does not name analysis_key {record.analysis_key}"
            )
        existing = self.get_analysis(record.analysis_key)
        if existing is not None:
            self.touch("analysis", record.analysis_id)
            return existing
        if self.get_take_by_id(record.take_id) is None:
            raise StoreError(f"take {record.take_id} is not in the store; publish the take before its analysis")
        final = self._layout.analysis_path(record.take_id, record.analysis_id)
        final.parent.mkdir(exist_ok=True)
        rel = self._layout.rel(final)
        now = self._clock()
        tmp, sha, size = files.write_temp(final, sidecar_bytes(record), readonly=True)

        def decide(conn: sqlite3.Connection) -> _Decision:
            # Checked again under the write lock: gc may have collected the take since the check above.
            if conn.execute("SELECT 1 FROM takes WHERE take_id = ?", (record.take_id,)).fetchone() is None:
                raise StoreError(f"take {record.take_id} is no longer in the store; publish the take again first")
            found = conn.execute("SELECT 1 FROM analyses WHERE analysis_key = ?", (record.analysis_key,)).fetchone()
            return "keep" if found else "new"

        def commit(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO analyses (analysis_key, analysis_id, take_id, rel_path, record, created_at, last_used_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (record.analysis_key, record.analysis_id, record.take_id, rel, row_json(record), now, now),
            )
            conn.execute(
                "UPDATE takes SET last_used_at = MAX(last_used_at, ?) WHERE take_id = ?", (now, record.take_id)
            )
            self._index_files(conn, "analysis", record.analysis_id, [(rel, sha, size)])

        self._publish_file(tmp, final, decide, commit)
        return self._must(self.get_analysis(record.analysis_key), "analysis", record.analysis_key)

    # ================================================================ measurements (section 3.2)
    def get_measurement(self, voice_hash: str, engine_profile_id: str) -> MeasurementRecord | None:
        sql = "SELECT record, rel_dir FROM measurements WHERE voice_hash = ? AND engine_profile_id = ?"
        return self._read_one(self._fetch(sql, voice_hash, engine_profile_id), self._measurement)

    def measurements_of(self, voice_hash: str) -> tuple[MeasurementRecord, ...]:
        """A voice's measurements, one per engine profile (``narration://measurements/{voice_hash}``)."""
        sql = "SELECT record, rel_dir FROM measurements WHERE voice_hash = ? ORDER BY engine_profile_id"
        return self._read_all(self._fetch_all(sql, voice_hash), self._measurement)

    def _measurement(self, row: sqlite3.Row) -> MeasurementRecord | _Reread | None:
        record = from_json(MeasurementRecord, json.loads(row["record"]))
        rel = f"{row['rel_dir']}/{MEASUREMENT_JSON}"
        if not self._present(rel):
            return None if self._drop("measurement", record.measurement_key, row["record"], rel) else _REREAD
        return record

    def put_measurement(self, record: MeasurementRecord) -> MeasurementRecord:
        """Publish ``measurement.json`` for (voice, engine profile). The same measurement key again is a
        no-op; a different one (a new corpus or ladder settings) replaces the old measurement."""
        if not keys.KEY_PATTERN.fullmatch(record.measurement_key):
            raise InvalidIdError("measurement_key", record.measurement_key)
        folder = self._layout.measurement_dir(record.voice_hash, record.engine_profile.id)
        final = folder / MEASUREMENT_JSON
        rel_dir = self._layout.rel(folder)
        now = self._clock()
        # Until this publish holds the write lock, gc may rename the folder away with the measurement it
        # replaces, so the temporary file waits at the top of the measurements tree, which gc never renames.
        # The folder is made (again, if gc took it) under the lock, by _publish_file.
        top = self._layout.tree(MEASUREMENTS)
        top.mkdir(parents=True, exist_ok=True)
        tmp, sha, size = files.write_temp(top / MEASUREMENT_JSON, sidecar_bytes(record), readonly=True)
        old_key: list[str] = []

        def decide(conn: sqlite3.Connection) -> _Decision:
            row = conn.execute(
                "SELECT measurement_key FROM measurements WHERE voice_hash = ? AND engine_profile_id = ?",
                (record.voice_hash, record.engine_profile.id),
            ).fetchone()
            if row is None:
                return "new"
            if row["measurement_key"] == record.measurement_key and os.path.isfile(final):
                return "keep"
            old_key.append(row["measurement_key"])
            return "replace"

        def commit(conn: sqlite3.Connection) -> None:
            for key in old_key:
                conn.execute("DELETE FROM files WHERE owner_kind = 'measurement' AND owner_id = ?", (key,))
            conn.execute(
                "INSERT OR REPLACE INTO measurements"
                " (voice_hash, engine_profile_id, measurement_key, rel_dir, record, created_at, last_used_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.voice_hash,
                    record.engine_profile.id,
                    record.measurement_key,
                    rel_dir,
                    row_json(record),
                    now,
                    now,
                ),
            )
            self._index_files(
                conn, "measurement", record.measurement_key, [(f"{rel_dir}/{MEASUREMENT_JSON}", sha, size)]
            )

        if not self._publish_file(tmp, final, decide, commit):
            self.touch("measurement", record.measurement_key)
        return self._must(
            self.get_measurement(record.voice_hash, record.engine_profile.id), "measurement", record.measurement_key
        )

    # ================================================================ voice profiles (section 3.6)
    def get_profile(self, audio_sha256: str, profile_version: str) -> ProfileRecord | None:
        sql = "SELECT record, rel_dir FROM profiles WHERE audio_sha256 = ? AND profile_version = ?"
        return self._read_one(self._fetch(sql, audio_sha256, profile_version), self._profile)

    def _profile(self, row: sqlite3.Row) -> ProfileRecord | _Reread | None:
        record = from_json(ProfileRecord, json.loads(row["record"]))
        rel = f"{row['rel_dir']}/{PROFILE_JSON}"
        if not self._present(rel):
            return None if self._drop("profile", record.audio_sha256, row["record"], rel) else _REREAD
        return map_profile(record, self._to_abs)

    def put_profile(self, record: ProfileRecord, pictures_dir: Path | None) -> ProfileRecord:
        """Publish ``profiles/<ab>/<sha256>/`` with ``profile.json`` and the pictures moved from
        ``pictures_dir`` (the file names the record's ``pictures`` give; with ``pictures_dir`` None, the
        record must name no pictures). A newer ``profile_version`` replaces an older profile of the audio."""
        existing = self.get_profile(record.audio_sha256, record.profile_version)
        sources = self._picture_sources(record.pictures, pictures_dir)
        if existing is not None:
            for src in sources.values():
                self._consume(src)
            self.touch("profile", record.audio_sha256)
            return existing
        final = self._layout.profile_dir(record.audio_sha256)
        rel_dir = self._layout.rel(final)
        pictures = ProfilePictures(
            spectrogram=f"{rel_dir}/{SPECTROGRAM_PNG}" if "spectrogram" in sources else "",
            pitch=f"{rel_dir}/{PITCH_PNG}" if "pitch" in sources else "",
        )
        stored = dataclasses.replace(record, pictures=pictures)
        now = self._clock()
        with self._staging(final) as staging:
            entries: list[tuple[str, str, int]] = []
            for field, name in (("spectrogram", SPECTROGRAM_PNG), ("pitch", PITCH_PNG)):
                if field in sources:
                    psha, psize = files.sha256_file(sources[field])
                    staging.move_in(sources[field], name)
                    entries.append((f"{rel_dir}/{name}", psha, psize))
            jsha, jsize = files.write_atomic(staging.path / PROFILE_JSON, sidecar_bytes(stored), readonly=True)
            entries.append((f"{rel_dir}/{PROFILE_JSON}", jsha, jsize))

            def decide(conn: sqlite3.Connection) -> _Decision:
                row = conn.execute(
                    "SELECT profile_version FROM profiles WHERE audio_sha256 = ?", (record.audio_sha256,)
                ).fetchone()
                if row is None:
                    return "new"
                return "keep" if row["profile_version"] == record.profile_version else "replace"

            def commit(conn: sqlite3.Connection) -> None:
                conn.execute(
                    "INSERT OR REPLACE INTO profiles"
                    " (audio_sha256, profile_version, rel_dir, record, created_at, last_used_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (record.audio_sha256, record.profile_version, rel_dir, row_json(stored), now, now),
                )
                self._index_files(conn, "profile", record.audio_sha256, entries)

            self._publish_dir(staging.path, final, decide, commit)
        return self._must(self.get_profile(record.audio_sha256, record.profile_version), "profile", record.audio_sha256)

    def _picture_sources(self, pictures: ProfilePictures, pictures_dir: Path | None) -> dict[str, Path]:
        named = {f: v for f, v in (("spectrogram", pictures.spectrogram), ("pitch", pictures.pitch)) if v}
        if pictures_dir is None:
            if named:
                raise StoreError("the profile names pictures but no pictures_dir was given")
            return {}
        return {f: self._source(pictures_dir / PurePath(v).name, f"the {f} picture") for f, v in named.items()}

    # ================================================================ designed candidates (section 3.1)
    def put_candidate(self, candidate: Candidate, clip_audio: Path) -> Candidate:
        """Move the designed clip into ``designs/<design_id>/<cand>/clip.wav`` (read-only) and publish
        ``candidate.json``; returns the candidate with ``clip.path`` and ``clip.sha256`` filled in. The
        candidate's profile, if any, keeps the picture paths ``put_profile`` gave it."""
        if not DESIGN_ID_PATTERN.fullmatch(candidate.design_id):
            raise InvalidIdError("design_id", candidate.design_id)
        existing = self._get_candidate(candidate.design_id, candidate.index)
        if existing is not None:
            self._consume(clip_audio)
            self.touch("design", candidate.design_id)
            return existing
        src = self._source(clip_audio, "clip_audio")
        sha, size = files.sha256_file(src)
        self._check_hash(candidate.clip.sha256, sha, f"candidate {candidate.design_id}/{candidate.index}")
        final = self._layout.design_dir(candidate.design_id, candidate.index)
        rel_dir = self._layout.rel(final)
        profile = map_profile(candidate.profile, self._to_rel) if candidate.profile is not None else None
        stored = dataclasses.replace(
            candidate, clip=AudioRef(path=f"{rel_dir}/{CLIP_WAV}", sha256=sha), profile=profile
        )
        now = self._clock()
        owner = f"{candidate.design_id}/{candidate.index}"
        with self._staging(final) as staging:
            staging.move_in(src, CLIP_WAV)
            jsha, jsize = files.write_atomic(staging.path / CANDIDATE_JSON, sidecar_bytes(stored), readonly=True)

            def decide(conn: sqlite3.Connection) -> _Decision:
                found = conn.execute(
                    "SELECT 1 FROM candidates WHERE design_id = ? AND idx = ?", (candidate.design_id, candidate.index)
                ).fetchone()
                return "keep" if found else "new"

            def commit(conn: sqlite3.Connection) -> None:
                conn.execute(
                    "INSERT INTO candidates (design_id, idx, clip_sha256, rel_dir, record, created_at, last_used_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (candidate.design_id, candidate.index, sha, rel_dir, row_json(stored), now, now),
                )
                self._index_files(
                    conn,
                    "candidate",
                    owner,
                    [(f"{rel_dir}/{CLIP_WAV}", sha, size), (f"{rel_dir}/{CANDIDATE_JSON}", jsha, jsize)],
                )

            self._publish_dir(staging.path, final, decide, commit)
        return self._must(self._get_candidate(candidate.design_id, candidate.index), "candidate", owner)

    def _get_candidate(self, design_id: str, index: int) -> Candidate | None:
        row = (
            self._conn()
            .execute("SELECT record FROM candidates WHERE design_id = ? AND idx = ?", (design_id, index))
            .fetchone()
        )
        return self._candidate_from_row(row)

    def _candidate_from_row(self, row: sqlite3.Row | None) -> Candidate | None:
        if row is None:
            return None
        record = from_json(Candidate, json.loads(row["record"]))
        if not self._present(record.clip.path):
            return None
        return map_candidate(record, self._to_abs)

    def get_design(self, design_id: str) -> tuple[Candidate, ...]:
        """A design's candidates, by index; empty if the design is unknown or collected."""
        rows = (
            self._conn()
            .execute("SELECT record FROM candidates WHERE design_id = ? ORDER BY idx", (design_id,))
            .fetchall()
        )
        return tuple(c for c in (self._candidate_from_row(r) for r in rows) if c is not None)

    # ================================================================ provenance (section 17.4)
    def add_provenance(self, entry: ProvenanceEntry) -> None:
        """Append to ``provenance.jsonl`` (append-only, never pruned) and index it. Adding the same entry
        again changes nothing."""
        if not keys.HEX64_PATTERN.fullmatch(entry.clip_sha256):
            raise InvalidIdError("clip_sha256", entry.clip_sha256)
        if not DESIGN_ID_PATTERN.fullmatch(entry.design_id):
            raise InvalidIdError("design_id", entry.design_id)
        line = (row_json(entry) + "\n").encode("utf-8")
        path = self._layout.provenance_path
        with self._write() as conn:
            found = conn.execute(
                "SELECT 1 FROM provenance WHERE clip_sha256 = ? AND design_id = ?", (entry.clip_sha256, entry.design_id)
            ).fetchone()
            if found:
                return
            with open(path, "a+b") as f:
                f.seek(0, os.SEEK_END)
                if f.tell() > 0:
                    f.seek(-1, os.SEEK_END)
                    if f.read(1) != b"\n":
                        f.write(b"\n")  # a line cut short by a crash stays on its own line
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            conn.execute(
                "INSERT INTO provenance (clip_sha256, design_id, date) VALUES (?, ?, ?)",
                (entry.clip_sha256, entry.design_id, entry.date),
            )

    def is_provenance(self, clip_sha256: str) -> bool:
        """Whether the service designed this clip (section 17.4)."""
        return (
            self._conn().execute("SELECT 1 FROM provenance WHERE clip_sha256 = ?", (clip_sha256,)).fetchone()
            is not None
        )

    def _reconcile_provenance(self) -> None:
        """Index any line of ``provenance.jsonl`` the database lacks (a crash between the append and the
        commit). A line cut short by a crash is skipped."""
        path = self._layout.provenance_path
        if not path.is_file():
            return
        entries: list[ProvenanceEntry] = []
        for raw in path.read_bytes().splitlines():
            try:
                entries.append(from_json(ProvenanceEntry, json.loads(raw.decode("utf-8"))))
            except (UnicodeDecodeError, json.JSONDecodeError, ContractError):
                continue
        with self._write() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO provenance (clip_sha256, design_id, date) VALUES (?, ?, ?)",
                [(e.clip_sha256, e.design_id, e.date) for e in entries],
            )

    # ================================================================ engine profiles and the canary
    def get_engine_profile(self, engine_profile_id: str) -> EngineProfile | None:
        row = (
            self._conn()
            .execute("SELECT record FROM engine_profiles WHERE engine_profile_id = ?", (engine_profile_id,))
            .fetchone()
        )
        return map_engine_profile(from_json(EngineProfile, json.loads(row["record"])), self._to_abs) if row else None

    def put_engine_profile(self, profile: EngineProfile) -> EngineProfile:
        """Write ``engines/<engine_profile_id>.json``. A pinned profile's hashed fields never change: the same
        id with another ``hash``, or with any other field the hash covers changed, is refused with
        ``StoreIntegrityError`` (a new pin is a new id). The fields the hash leaves out (``snapshot_dir``,
        ``observed``, ``tier``, the canary) may be updated."""
        path = self._layout.engine_path(profile.engine_profile_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        stored = map_engine_profile(profile, self._to_rel)
        text = row_json(stored)
        rel = self._layout.rel(path)
        now = self._clock()
        tmp, sha, size = files.write_temp(path, sidecar_bytes(stored), readonly=True)

        def decide(conn: sqlite3.Connection) -> _Decision:
            row = conn.execute(
                "SELECT hash, record FROM engine_profiles WHERE engine_profile_id = ?", (profile.engine_profile_id,)
            ).fetchone()
            if row is None:
                return "new"
            pinned, offered = _hashed_fields(json.loads(row["record"])), _hashed_fields(json.loads(text))
            changed = sorted(k for k in pinned.keys() | offered.keys() if pinned.get(k) != offered.get(k))
            if row["hash"] != profile.hash or changed:
                what = f"hash {profile.hash}" if row["hash"] != profile.hash else f"a different {', '.join(changed)}"
                raise StoreIntegrityError(
                    f"engine profile {profile.engine_profile_id} is pinned; this one has {what}. A changed "
                    "profile is a new pin with a new id (design section 10.1)"
                )
            return "keep" if row["record"] == text and os.path.isfile(path) else "replace"

        def commit(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT OR REPLACE INTO engine_profiles (engine_profile_id, hash, rel_path, record, updated_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (profile.engine_profile_id, profile.hash, rel, text, now),
            )
            self._index_files(conn, "engine", profile.engine_profile_id, [(rel, sha, size)])

        self._publish_file(tmp, path, decide, commit)
        return self._must(
            self.get_engine_profile(profile.engine_profile_id), "engine profile", profile.engine_profile_id
        )

    def list_engine_profiles(self) -> tuple[EngineProfile, ...]:
        rows = self._conn().execute("SELECT record FROM engine_profiles ORDER BY engine_profile_id").fetchall()
        return tuple(map_engine_profile(from_json(EngineProfile, json.loads(r["record"])), self._to_abs) for r in rows)

    def current_engine_profile(self, kind: EngineKind) -> EngineProfile | None:
        """The profile in use for ``base`` or ``design`` work, or None before ``engine pin``."""
        ident = self._setting(f"current_engine_profile.{_engine_kind(kind)}")
        return self.get_engine_profile(ident) if ident else None

    def set_current_engine_profile(self, kind: EngineKind, engine_profile_id: str) -> None:
        name = f"current_engine_profile.{_engine_kind(kind)}"
        with self._write() as conn:
            found = conn.execute(
                "SELECT 1 FROM engine_profiles WHERE engine_profile_id = ?", (engine_profile_id,)
            ).fetchone()
            if not found:
                raise NotFoundError("engine profile", engine_profile_id)
            conn.execute("INSERT OR REPLACE INTO settings (name, value) VALUES (?, ?)", (name, engine_profile_id))

    def put_canary_clip(self, engine_profile_id: str, audio: Path) -> AudioRef:
        """Move the designed canary clip to ``engines/<engine_profile_id>/canary.wav`` (read-only, never
        collected, re-hashed by ``verify``) and return its location and sha256 (DC-3). Once a pinned
        profile's canary names the clip, a different clip for that profile is refused."""
        final = self._layout.canary_clip_path(engine_profile_id)
        final.parent.mkdir(parents=True, exist_ok=True)
        rel = self._layout.rel(final)
        src = self._source(audio, "audio")
        sha, size = files.sha256_file(src)
        pinned = self.get_engine_profile(engine_profile_id)  # before the move: nothing after it may strand the clip
        tmp = self._layout.confine(files.temp_name(final))
        files.move_into(src, tmp, readonly=True)

        def decide(conn: sqlite3.Connection) -> _Decision:
            row = conn.execute("SELECT sha256 FROM files WHERE rel_path = ?", (rel,)).fetchone()
            if row is None:
                return "new"
            if row["sha256"] == sha and os.path.isfile(final):
                return "keep"
            if pinned is not None and pinned.canary is not None and pinned.canary.clip.sha256 == row["sha256"]:
                raise StoreIntegrityError(
                    f"engine profile {engine_profile_id} is pinned with canary {row['sha256']}; a new canary is a "
                    "new pin with a new id"
                )
            return "replace"

        def commit(conn: sqlite3.Connection) -> None:
            self._index_files(conn, "canary", engine_profile_id, [(rel, sha, size)])

        self._publish_file(tmp, final, decide, commit, source=src)  # a failure gives the clip back
        return AudioRef(path=str(final), sha256=sha)

    # ================================================================ the alignment benchmark (section 11.2)
    def get_alignment_benchmark(self, method_id: str) -> AlignmentBenchmark | None:
        row = (
            self._conn().execute("SELECT record FROM alignment_benchmarks WHERE method_id = ?", (method_id,)).fetchone()
        )
        return from_json(AlignmentBenchmark, json.loads(row["record"])) if row else None

    def put_alignment_benchmark(self, record: AlignmentBenchmark) -> AlignmentBenchmark:
        """Write ``alignment/<method_id>.json``. A later benchmark of the same method (e.g. the full-size one
        after the Phase 0 one) replaces it. Publishing a benchmark does not make its method the one in use:
        that is the configured aligner's (``current_alignment_benchmark``)."""
        path = self._layout.alignment_path(record.method_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        rel = self._layout.rel(path)
        text = row_json(record)
        now = self._clock()
        tmp, sha, size = files.write_temp(path, sidecar_bytes(record), readonly=True)

        def decide(conn: sqlite3.Connection) -> _Decision:
            return "new"

        def commit(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT OR REPLACE INTO alignment_benchmarks (method_id, rel_path, record, updated_at)"
                " VALUES (?, ?, ?, ?)",
                (record.method_id, rel, text, now),
            )
            self._index_files(conn, "alignment", record.method_id, [(rel, sha, size)])

        self._publish_file(tmp, path, decide, commit)
        return self._must(self.get_alignment_benchmark(record.method_id), "alignment benchmark", record.method_id)

    def current_alignment_benchmark(self) -> AlignmentBenchmark | None:
        """The benchmark of the configured aligner method (``alignment_method_id``), whose measured error
        every take reports; None if that method has no published benchmark, or no method is configured.
        Never simply the last benchmark published: that may be another aligner's."""
        if self._alignment_method_id is None:
            return None
        return self.get_alignment_benchmark(self._alignment_method_id)

    def _setting(self, name: str) -> str | None:
        row = self._conn().execute("SELECT value FROM settings WHERE name = ?", (name,)).fetchone()
        return str(row["value"]) if row else None

    # ================================================================ claims (section 4 item 6)
    def claim(self, key: str, holder: str, *, ttl_s: float) -> tuple[ClaimResult, SqliteLease | None]:
        """``exists`` if the key's result is published (checked now, in the same transaction);
        ``in_flight`` if another holder has a live lease; otherwise ``claimed`` with a lease. A lease past
        its time is free to claim. Claiming again under the same holder renews that holder's claim."""
        if not keys.KEY_PATTERN.fullmatch(key):
            raise InvalidIdError("key", key)
        if not isinstance(holder, str) or not holder or len(holder) > 200:
            raise ValueError("holder must be a non-empty name of at most 200 characters")
        _check_ttl(ttl_s)
        with self._write() as conn:
            if self._published(conn, key):
                return "exists", None
            now = self._clock()
            row = conn.execute("SELECT holder, expires_at FROM leases WHERE key = ?", (key,)).fetchone()
            if row is not None and row["expires_at"] > now and row["holder"] != holder:
                return "in_flight", None
            token = files.token() + files.token()
            conn.execute(
                "INSERT OR REPLACE INTO leases (key, holder, token, acquired_at, expires_at) VALUES (?, ?, ?, ?, ?)",
                (key, holder, token, now, now + ttl_s),
            )
        return "claimed", SqliteLease(self, key, holder, token)

    def wait_for(self, key: str, *, timeout_s: float) -> bool:
        """Wait until the key's result exists (True), or its lease lapses or is released without a result,
        or the timeout passes (False)."""
        deadline = time.monotonic() + max(0.0, timeout_s)
        delay = 0.02
        while True:
            with self._read() as conn:
                if self._published(conn, key):
                    return True
                row = conn.execute("SELECT expires_at FROM leases WHERE key = ?", (key,)).fetchone()
                live = row is not None and row["expires_at"] > self._clock()
            if not live:
                with self._read() as conn:
                    return self._published(conn, key)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(delay, remaining))
            delay = min(delay * 1.5, 0.5)

    def _published(self, conn: sqlite3.Connection, key: str) -> bool:
        """Whether a result for ``key`` is indexed and its file is on disk (any layer)."""
        for sql in (
            "SELECT rel_dir || '/' || 'raw.wav' FROM renders WHERE render_key = ?",
            "SELECT rel_dir || '/' || 'delivery.wav' FROM takes WHERE delivery_key = ?",
            "SELECT rel_path FROM analyses WHERE analysis_key = ?",
            "SELECT rel_dir || '/' || 'measurement.json' FROM measurements WHERE measurement_key = ?",
        ):
            row = conn.execute(sql, (key,)).fetchone()
            if row is not None and self._present(row[0]):
                return True
        return False

    # ================================================================ jobs and the queue (sections 6, 7.3, 8)
    def create_job(self, record: JobRecord) -> tuple[JobRecord, bool]:
        """Insert a job, or return the active job (queued, running or cancelling) of the same kind with the
        same ``request_sha256``. The bool says whether it is new. A request whose earlier job has finished
        makes a new job.

        ``idempotency_key`` deduplicates retries, and a retry is the same request: an active job of the same
        kind with the same key and the same request is returned. With the same key and a different request
        it is refused, ``NarrationError(INVALID_ARGUMENT, field="idempotency_key")`` (plan.md DC-6): returning
        the old job would tell the caller that its different request had been queued.
        """
        self._layout.job_dir(record.job_id)  # checks the id
        record = self._validated_job(record)
        now = self._clock()
        with self._write() as conn:
            row = None
            if record.idempotency_key:
                row = conn.execute(
                    "SELECT record, request_sha256 FROM jobs WHERE kind = ? AND idempotency_key = ?"
                    " AND status IN (?, ?, ?) ORDER BY seq LIMIT 1",
                    (record.kind, record.idempotency_key, *_ACTIVE),
                ).fetchone()
                if row is not None and row["request_sha256"] != record.request_sha256:
                    raise NarrationError(
                        codes.INVALID_ARGUMENT,
                        f"idempotency_key {record.idempotency_key!r} belongs to an active {record.kind} job "
                        "with a different request",
                        field="idempotency_key",
                        hint=DC6_HINT,
                    )
            if row is None:
                row = conn.execute(
                    "SELECT record FROM jobs WHERE kind = ? AND request_sha256 = ? AND status IN (?, ?, ?)"
                    " ORDER BY seq LIMIT 1",
                    (record.kind, record.request_sha256, *_ACTIVE),
                ).fetchone()
            if row is not None:
                return from_json(JobRecord, json.loads(row["record"])), False
            conn.execute(
                "INSERT INTO jobs (job_id, kind, status, priority_rank, request_sha256, idempotency_key,"
                " claimed_by, inserted_at, last_used_at, updated_at, record)"
                " VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)",
                (
                    record.job_id,
                    record.kind,
                    record.status,
                    _PRIORITY_RANK[record.priority],
                    record.request_sha256,
                    record.idempotency_key,
                    now,
                    now,
                    record.updated_at,
                    row_json(record),
                ),
            )
            self._write_job_json(record)
        return record, True

    def get_job(self, job_id: str) -> JobRecord | None:
        row = self._conn().execute("SELECT record FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return from_json(JobRecord, json.loads(row["record"])) if row else None

    def update_job(self, job_id: str, *, expect_status: JobStatus | None = None, **changes: Any) -> JobRecord | None:
        """Change ``JobRecord`` fields and bump ``updated_at`` (unless ``changes`` sets it). With
        ``expect_status``, the change is made only while the job is in that status (compare-and-set);
        otherwise nothing changes and the result is None. The job's identity (id, kind, request, its hash,
        the idempotency key, the creation time) never changes. Raises ``NotFoundError`` for an unknown job."""
        fixed = _FIXED_JOB_FIELDS & changes.keys()
        if fixed:
            raise ValueError(f"a job's {', '.join(sorted(fixed))} never change")
        with self._write() as conn:
            row = conn.execute("SELECT record FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise NotFoundError("job", job_id)
            current = from_json(JobRecord, json.loads(row["record"]))
            if expect_status is not None and current.status != expect_status:
                return None
            changes.setdefault("updated_at", self._now_iso())
            updated = self._validated_job(dataclasses.replace(current, **changes))
            self._store_job(conn, updated)
        return updated

    def queued_jobs(self) -> tuple[JobRecord, ...]:
        """Queued, running and cancelling jobs, by priority (``interactive`` first) then FIFO."""
        rows = (
            self._conn()
            .execute("SELECT record FROM jobs WHERE status IN (?, ?, ?) ORDER BY priority_rank, seq", _ACTIVE)
            .fetchall()
        )
        return tuple(from_json(JobRecord, json.loads(r["record"])) for r in rows)

    def claim_job(self, job_id: str, holder: str) -> JobRecord | None:
        """Atomically move this job from ``queued`` to ``running`` for ``holder``; None if it is no longer
        queued (or unknown)."""
        with self._write() as conn:
            row = conn.execute("SELECT record FROM jobs WHERE job_id = ? AND status = 'queued'", (job_id,)).fetchone()
            return self._start_job(conn, row, holder)

    def claim_next_job(self, holder: str) -> JobRecord | None:
        """Atomically move the next queued job (priority, then FIFO) to ``running`` for ``holder``."""
        with self._write() as conn:
            row = conn.execute(
                "SELECT record FROM jobs WHERE status = 'queued' ORDER BY priority_rank, seq LIMIT 1"
            ).fetchone()
            return self._start_job(conn, row, holder)

    def _start_job(self, conn: sqlite3.Connection, row: sqlite3.Row | None, holder: str) -> JobRecord | None:
        if row is None:
            return None
        current = from_json(JobRecord, json.loads(row["record"]))
        updated = dataclasses.replace(current, status="running", updated_at=self._now_iso())
        self._store_job(conn, updated)
        conn.execute("UPDATE jobs SET claimed_by = ? WHERE job_id = ?", (holder, updated.job_id))
        return updated

    def jobs_created_since(self, iso_time: str) -> int:
        """How many jobs were created at or after ``iso_time`` (whatever their status now), across every
        process: the submit rate cap (``RATE_LIMITED``)."""
        since = parse_iso(iso_time)
        return int(self._conn().execute("SELECT COUNT(*) FROM jobs WHERE inserted_at >= ?", (since,)).fetchone()[0])

    def _validated_job(self, record: JobRecord) -> JobRecord:
        """A job record checked against its contract (types, literals), as the store will read it back."""
        try:
            return from_json(JobRecord, to_json(record))
        except ContractError as exc:
            raise ValueError(f"not a valid job record: {exc}") from exc

    def _store_job(self, conn: sqlite3.Connection, record: JobRecord) -> None:
        conn.execute(
            "UPDATE jobs SET status = ?, priority_rank = ?, updated_at = ?, last_used_at = ?, record = ?"
            " WHERE job_id = ?",
            (
                record.status,
                _PRIORITY_RANK[record.priority],
                record.updated_at,
                self._clock(),
                row_json(record),
                record.job_id,
            ),
        )
        self._write_job_json(record)

    def _write_job_json(self, record: JobRecord) -> None:
        """``jobs/<job_id>/job.json``, a copy of the row for people and tools. Best effort: the row is the
        record, so a copy that cannot be written (the file held open, a full disk) is logged and never
        fails the job's change."""
        folder = self._layout.job_dir(record.job_id)
        try:
            folder.mkdir(parents=True, exist_ok=True)
            files.write_atomic(folder / JOB_JSON, sidecar_bytes(record), readonly=False, durable=False)
        except OSError as exc:
            _log.warning(
                "could not write %s for job %s (the database row is up to date): %s", JOB_JSON, record.job_id, exc
            )

    # ================================================================ the daemon's status and commands
    def get_daemon_status(self) -> DaemonStatus | None:
        """``run/daemon.json``, or None when no daemon has written one. The daemon renames a new file over it
        while others read it, so a read Windows refuses during that rename is tried again
        (``files.read_retrying``)."""
        path = self._layout.daemon_json_path()
        try:
            data = json.loads(files.read_retrying(path).decode("utf-8"))
        except FileNotFoundError:
            return None
        return from_json(DaemonStatus, data)

    def put_daemon_status(self, status: DaemonStatus) -> None:
        path = self._layout.daemon_json_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        files.write_atomic(path, sidecar_bytes(status), readonly=False, durable=False)

    def post_command(self, kind: DaemonCommandKind) -> DaemonCommand:
        if kind not in get_args(DaemonCommandKind):
            raise ValueError(f"unknown daemon command {kind!r}")
        command = DaemonCommand(
            command_id=self._ulids.new(), kind=kind, requested_at=self._now_iso(), done_at=None, result=None
        )
        with self._write() as conn:
            conn.execute(
                "INSERT INTO commands (command_id, kind, requested_at) VALUES (?, ?, ?)",
                (command.command_id, command.kind, command.requested_at),
            )
        return command

    def pending_commands(self) -> tuple[DaemonCommand, ...]:
        """Commands not yet completed, oldest first."""
        rows = self._conn().execute("SELECT * FROM commands WHERE done_at IS NULL ORDER BY seq").fetchall()
        return tuple(_command(r) for r in rows)

    def commands_since(self, requested_at: str) -> tuple[DaemonCommand, ...]:
        """Every command posted at or after ``requested_at`` (a store time, as ``utc_iso`` writes it), pending
        or done, in the order they were posted (``seq``). Store times are fixed-width, so the comparison is
        on the text. Raises ``ValueError`` for anything that is not a store time. Commands are never pruned."""
        if not isinstance(requested_at, str) or not _STORE_TIME.fullmatch(requested_at):
            raise ValueError(f"not a store time (YYYY-MM-DDTHH:MM:SS.mmmZ): {requested_at!r}")
        rows = (
            self._conn()
            .execute("SELECT * FROM commands WHERE requested_at >= ? ORDER BY seq", (requested_at,))
            .fetchall()
        )
        return tuple(_command(r) for r in rows)

    def complete_command(self, command_id: str, result: Mapping[str, Any] | None) -> DaemonCommand:
        """Record a command's outcome. Completing it again keeps the first outcome."""
        text = json.dumps(dict(result), ensure_ascii=False, allow_nan=False) if result is not None else None
        with self._write() as conn:
            conn.execute(
                "UPDATE commands SET done_at = ?, result = ? WHERE command_id = ? AND done_at IS NULL",
                (self._now_iso(), text, command_id),
            )
            row = conn.execute("SELECT * FROM commands WHERE command_id = ?", (command_id,)).fetchone()
        if row is None:
            raise NotFoundError("daemon command", command_id)
        return _command(row)

    def wait_for_command(self, command_id: str, *, timeout_s: float) -> DaemonCommand | None:
        """The command once the daemon has completed it, or None at the timeout."""
        deadline = time.monotonic() + max(0.0, timeout_s)
        delay = 0.02
        while True:
            row = self._conn().execute("SELECT * FROM commands WHERE command_id = ?", (command_id,)).fetchone()
            if row is None:
                raise NotFoundError("daemon command", command_id)
            if row["done_at"] is not None:
                return _command(row)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(delay, remaining))
            delay = min(delay * 1.5, 0.5)

    # ================================================================ retention (section 15)
    def touch(self, kind: str, ident: str) -> None:
        """Record a use of an item, which restarts its retention period. ``kind`` is one of ``RetentionKind``
        (anything else is a ``ValueError``). Using an analysis also uses its take; using a design also uses
        its candidates' voice profiles. An unknown item is ignored (it may just have been collected)."""
        now = self._clock()
        statements = {
            "render": ["UPDATE renders SET last_used_at = ? WHERE render_id = ?"],
            "take": ["UPDATE takes SET last_used_at = ? WHERE take_id = ?"],
            "analysis": [
                "UPDATE analyses SET last_used_at = ? WHERE analysis_id = ?",
                "UPDATE takes SET last_used_at = ?"
                " WHERE take_id = (SELECT take_id FROM analyses WHERE analysis_id = ?)",
            ],
            "measurement": ["UPDATE measurements SET last_used_at = ? WHERE measurement_key = ?"],
            "profile": ["UPDATE profiles SET last_used_at = ? WHERE audio_sha256 = ?"],
            "design": ["UPDATE candidates SET last_used_at = ? WHERE design_id = ?"],
            "job": ["UPDATE jobs SET last_used_at = ? WHERE job_id = ?"],
        }.get(kind)
        if statements is None:
            raise ValueError(f"unknown retention kind {kind!r}")
        with self._write() as conn:
            for sql in statements:
                conn.execute(sql, (now, ident))
            if kind == "design":
                rows = conn.execute("SELECT clip_sha256, record FROM candidates WHERE design_id = ?", (ident,))
                conn.executemany(
                    "UPDATE profiles SET last_used_at = ? WHERE audio_sha256 = ?",
                    [(now, sha) for sha in sorted(_candidate_profiles(rows.fetchall()))],
                )

    def _drop(self, kind: str, ident: str, record_text: str, rel: str) -> bool:
        """Forget an index entry whose file is gone (removed by hand, or damaged). Checked again under the
        write lock: the row is dropped only if it is still the row that was read (``record_text``) and its
        file (``rel``) is still missing, so a row a concurrent publish has just replaced is never dropped.

        Returns True if it dropped the row. False means the row was replaced or deleted, or its file is back:
        the reader saw a publish in progress, which renames the old file aside before it renames the new one
        in. That publish has finished by the time this returns (this waited for its lock), so the reader
        reads again (``_read_one``)."""
        table, column = _DROP_ROWS[kind]  # fixed names from this module, never from a caller
        with self._write() as conn:
            row = conn.execute(f"SELECT record FROM {table} WHERE {column} = ?", (ident,)).fetchone()
            if row is None or row["record"] != record_text or self._present(rel):
                return False
            self._delete_rows(conn, kind, ident)
            return True

    def _fetch(self, sql: str, *args: object) -> Callable[[], sqlite3.Row | None]:
        """A query for one row, to run (and run again) later: ``_read_one``'s ``fetch``."""
        return lambda: self._conn().execute(sql, args).fetchone()

    def _fetch_all(self, sql: str, *args: object) -> Callable[[], list[sqlite3.Row]]:
        return lambda: self._conn().execute(sql, args).fetchall()

    def _read_one(
        self, fetch: Callable[[], sqlite3.Row | None], convert: Callable[[sqlite3.Row], _T | _Reread | None]
    ) -> _T | None:
        """``convert(fetch())``, read again while ``convert`` says the row changed under it (``_drop``). A
        row replaced over and over during the read is given up on after ``_READ_ATTEMPTS`` reads."""
        for _ in range(_READ_ATTEMPTS):
            row = fetch()
            if row is None:
                return None
            result = convert(row)
            if not isinstance(result, _Reread):
                return result
        return None

    def _read_all(
        self, fetch: Callable[[], list[sqlite3.Row]], convert: Callable[[sqlite3.Row], _T | _Reread | None]
    ) -> tuple[_T, ...]:
        """``_read_one`` for a query of several rows: all of them are read again if any one changed."""
        results: list[_T | _Reread | None] = []
        for _ in range(_READ_ATTEMPTS):
            results = [convert(row) for row in fetch()]
            if not any(isinstance(r, _Reread) for r in results):
                break
        return tuple(r for r in results if r is not None and not isinstance(r, _Reread))

    def _delete_rows(self, conn: sqlite3.Connection, kind: str, ident: str) -> None:
        if kind == "render":
            conn.execute("DELETE FROM renders WHERE render_id = ?", (ident,))
            conn.execute("DELETE FROM files WHERE owner_kind = 'render' AND owner_id = ?", (ident,))
        elif kind == "take":
            for (analysis_id,) in conn.execute(
                "SELECT analysis_id FROM analyses WHERE take_id = ?", (ident,)
            ).fetchall():
                self._delete_rows(conn, "analysis", analysis_id)
            conn.execute("DELETE FROM takes WHERE take_id = ?", (ident,))
            conn.execute("DELETE FROM files WHERE owner_kind = 'take' AND owner_id = ?", (ident,))
        elif kind == "analysis":
            conn.execute("DELETE FROM analyses WHERE analysis_id = ?", (ident,))
            conn.execute("DELETE FROM files WHERE owner_kind = 'analysis' AND owner_id = ?", (ident,))
        elif kind == "measurement":
            conn.execute("DELETE FROM measurements WHERE measurement_key = ?", (ident,))
            conn.execute("DELETE FROM files WHERE owner_kind = 'measurement' AND owner_id = ?", (ident,))
        elif kind == "profile":
            conn.execute("DELETE FROM profiles WHERE audio_sha256 = ?", (ident,))
            conn.execute("DELETE FROM files WHERE owner_kind = 'profile' AND owner_id = ?", (ident,))
        elif kind == "design":
            conn.execute("DELETE FROM candidates WHERE design_id = ?", (ident,))
            conn.execute("DELETE FROM files WHERE owner_kind = 'candidate' AND owner_id LIKE ?", (f"{ident}/%",))
        elif kind == "job":
            conn.execute("DELETE FROM jobs WHERE job_id = ?", (ident,))
        else:
            raise ValueError(kind)

    def gc(self, *, dry_run: bool = True, now: float | None = None, grace_s: float = DEFAULT_GRACE_S) -> dict[str, Any]:
        """Collect what retention allows (section 15). **A dry run by default**: it only reports.

        Collected: renders, takes (with their analyses), analyses, voice profiles, designs and finished jobs
        not used for ``retention_days``; measurements not used for ``measurement_retention_days``; expired
        leases; ``.tmp-``/``.staging-``/``.trash-`` leftovers and unindexed folders older than ``grace_s``
        (what a crash leaves; a trash name's age is the time written in it, see ``files.trash_name``);
        scratch files not touched for ``retention_days``. Never collected: the provenance list, engine
        profiles and their canaries, alignment benchmarks, queued or running jobs, and what a live
        measurement or design keeps alive (the module docstring).

        A real run collects in bounded write transactions (``GC_BATCH_ITEMS`` items, about
        ``GC_BATCH_SECONDS``), so other writers never wait on it for long. In each, it renames each item to a
        ``.trash-`` sibling and deletes its rows, and after the commit it removes those trash names, so a
        key published again meanwhile is never touched. A folder in use is tried once, then listed under
        ``errors`` and kept for the next run. Leftovers and orphans are found before any row is deleted, so a
        dry run and a real run report the same things, each once. A path that cannot be removed is listed
        under ``errors`` and the run goes on. If a transaction fails, its renames are undone and the error
        is raised; what earlier transactions collected stays collected.
        """
        now = self._clock() if now is None else now
        cache_cutoff = now - self._retention.retention_days * _DAY
        measurement_cutoff = now - self._retention.measurement_retention_days * _DAY
        leftover_cutoff = now - grace_s
        errors: list[dict[str, str]] = []
        # What a crash left, found against the index as it stands, before any row is deleted.
        leftovers, orphans = self._leftovers(leftover_cutoff)
        scratch = self._old_entries(self._layout.tree(SCRATCH), cache_cutoff)
        if dry_run:
            with self._read() as conn:
                items = self._gc_plan(conn, cache_cutoff, measurement_cutoff)
                victims = self._victims(conn, items, errors)
                expired_leases = int(
                    conn.execute("SELECT COUNT(*) FROM leases WHERE expires_at <= ?", (now,)).fetchone()[0]
                )
            # As a real run would: nothing inside a victim, and no leftover inside an orphan, is listed twice.
            orphans, scratch = _outside(orphans, victims), _outside(scratch, victims)
            leftovers = _outside(leftovers, [*victims, *orphans])
            total = sum(files.tree_size(p) for p in [*victims, *leftovers, *orphans, *scratch])
        else:
            items, orphans, expired_leases, total, victims = self._collect(
                now, cache_cutoff, measurement_cutoff, leftover_cutoff, orphans, errors
            )
            leftovers, scratch = _outside(leftovers, victims), _outside(scratch, victims)
            removed: list[Path] = []
            for path in [*leftovers, *scratch]:
                if path in scratch and _newest_mtime(path) >= cache_cutoff:
                    continue  # used again since it was listed
                size = files.tree_size(path)
                if self._remove(path, path, errors):
                    removed.append(path)
                    total += size
            leftovers = [p for p in leftovers if p in removed]
            scratch = [p for p in scratch if p in removed]
        with self._read() as conn:
            kept = {
                "provenance": int(conn.execute("SELECT COUNT(*) FROM provenance").fetchone()[0]),
                "engine_profiles": int(conn.execute("SELECT COUNT(*) FROM engine_profiles").fetchone()[0]),
                "alignment_benchmarks": int(conn.execute("SELECT COUNT(*) FROM alignment_benchmarks").fetchone()[0]),
            }
        return {
            "dry_run": dry_run,
            "now": utc_iso(now),
            "cutoffs": {
                "cache": utc_iso(cache_cutoff),
                "measurements": utc_iso(measurement_cutoff),
                "leftovers": utc_iso(leftover_cutoff),
            },
            "items": items,
            "expired_leases": expired_leases,
            "leftovers": [self._layout.rel(p) for p in leftovers],
            "orphans": [self._layout.rel(p) for p in orphans],
            "scratch": [self._layout.rel(p) for p in scratch],
            "bytes": total,
            "kept": kept,
            "errors": errors,
        }

    def _collect(
        self,
        now: float,
        cache_cutoff: float,
        measurement_cutoff: float,
        leftover_cutoff: float,
        orphans: list[Path],
        errors: list[dict[str, str]],
    ) -> tuple[dict[str, list[str]], list[Path], int, int, list[Path]]:
        """The real half of ``gc``, in bounded write transactions. Each one checks its items against the plan
        made under its own lock (an item used since, or newly kept alive, is left), renames each to a
        ``.trash-`` sibling and deletes its rows; after its commit its trash is removed. Orphans follow, each
        checked again for an index row. Returns the items collected, the orphans taken, the expired leases
        deleted, the bytes removed and the original paths of everything collected."""
        with self._read() as conn:
            planned = self._gc_plan(conn, cache_cutoff, measurement_cutoff)
        queue = [(kind, ident) for kind, idents in planned.items() for ident in idents]
        items: dict[str, list[str]] = {kind: [] for kind in planned}
        victims: list[Path] = []
        total = 0
        at = 0
        while at < len(queue):
            renames = _Renames()
            batch: list[tuple[str, str]] = []
            with self._write_undoing(renames) as conn:
                still = {k: set(v) for k, v in self._gc_plan(conn, cache_cutoff, measurement_cutoff).items()}
                started = time.monotonic()
                examined = 0
                while at < len(queue) and _batch_open(examined, started):
                    kind, ident = queue[at]
                    at += 1
                    examined += 1
                    if ident not in still[kind]:
                        continue  # used again, or newly kept alive, since the plan
                    try:
                        path = self._item_path(conn, kind, ident)
                    except NarrationError as exc:
                        errors.append({"path": f"{kind}/{ident}", "error": exc.message})
                        continue
                    if self._to_trash(path, renames, errors):
                        self._delete_rows(conn, _GC_KINDS[kind], ident)
                        batch.append((kind, ident))
            for kind, ident in batch:
                items[kind].append(ident)
            total += self._empty_trash(renames, errors, victims)
        taken: list[Path] = []
        at = 0
        while at < len(orphans):
            renames = _Renames()
            chosen: list[Path] = []
            with self._write_undoing(renames) as conn:
                indexed = self._indexed(conn)
                started = time.monotonic()
                examined = 0
                while at < len(orphans) and _batch_open(examined, started):
                    path = orphans[at]
                    at += 1
                    examined += 1
                    if self._layout.rel(path) in indexed or _mtime(path) >= leftover_cutoff:
                        continue  # published since it was listed
                    if self._to_trash(path, renames, errors):
                        chosen.append(path)
            taken.extend(chosen)
            total += self._empty_trash(renames, errors, victims)
        with self._write() as conn:
            expired = conn.execute("DELETE FROM leases WHERE expires_at <= ?", (now,)).rowcount
        return items, taken, expired, total, victims

    def _to_trash(self, path: Path, renames: _Renames, errors: list[dict[str, str]]) -> bool:
        """Rename ``path`` to a ``.trash-`` sibling (nothing to do if it is already gone), trying once: this
        runs under the write lock, so a folder in use is not waited for. False, with the error listed, if it
        cannot be renamed: its rows are then kept for the next run."""
        if not os.path.lexists(path):
            return True
        trash = files.trash_name(path, self._clock())
        try:
            self._layout.confine(trash)
            renames.rename(path, trash, attempts=1)
        except (OSError, NarrationError) as exc:
            errors.append({"path": self._layout.rel(path), "error": _describe(exc)})
            return False
        return True

    def _empty_trash(self, renames: _Renames, errors: list[dict[str, str]], victims: list[Path]) -> int:
        """After a ``gc`` transaction commits (outside the lock): measure and remove its trash, and record the
        original paths in ``victims``. Returns the bytes removed."""
        total = 0
        for original, trash in renames.done:
            victims.append(original)
            size = files.tree_size(trash)
            if self._remove(trash, original, errors):
                total += size
            if original.parent.parent.name == MEASUREMENTS:
                with contextlib.suppress(OSError):
                    original.parent.rmdir()  # the voice's folder, once its last measurement is gone
        return total

    def _remove(self, path: Path, shown_as: Path, errors: list[dict[str, str]]) -> bool:
        """Remove a tree; on failure list the error (under ``shown_as``) and carry on."""
        try:
            files.remove_tree(path)
        except OSError as exc:
            errors.append({"path": self._layout.rel(shown_as), "error": _describe(exc)})
            return False
        return True

    def _gc_plan(
        self, conn: sqlite3.Connection, cache_cutoff: float, measurement_cutoff: float
    ) -> dict[str, list[str]]:
        """The ids retention allows ``gc`` to collect, by kind, less what a live item keeps alive: the takes
        a live measurement names (with their renders and analyses), and a live design's voice profiles."""

        def ids(sql: str, *args: Any) -> list[str]:
            return sorted(str(r[0]) for r in conn.execute(sql, args).fetchall())

        plan = {
            "renders": ids("SELECT render_id FROM renders WHERE last_used_at < ?", cache_cutoff),
            "takes": ids(
                "SELECT take_id FROM takes t WHERE last_used_at < ? AND NOT EXISTS"
                " (SELECT 1 FROM analyses a WHERE a.take_id = t.take_id AND a.last_used_at >= ?)",
                cache_cutoff,
                cache_cutoff,
            ),
            "analyses": ids(
                "SELECT a.analysis_id FROM analyses a JOIN takes t ON t.take_id = a.take_id"
                " WHERE a.last_used_at < ? AND (t.last_used_at >= ? OR EXISTS (SELECT 1 FROM analyses b"
                " WHERE b.take_id = a.take_id AND b.last_used_at >= ?))",
                cache_cutoff,
                cache_cutoff,
                cache_cutoff,
            ),
            "profiles": ids("SELECT audio_sha256 FROM profiles WHERE last_used_at < ?", cache_cutoff),
            "designs": ids(
                "SELECT design_id FROM candidates GROUP BY design_id HAVING MAX(last_used_at) < ?", cache_cutoff
            ),
            "measurements": ids("SELECT measurement_key FROM measurements WHERE last_used_at < ?", measurement_cutoff),
            "jobs": ids(
                "SELECT job_id FROM jobs WHERE last_used_at < ? AND status NOT IN (?, ?, ?)",
                cache_cutoff,
                *_ACTIVE,
            ),
        }
        # A live measurement keeps the takes it names, with their renders and analyses.
        pinned_takes: set[str] = set()
        for (text,) in conn.execute("SELECT record FROM measurements WHERE last_used_at >= ?", (measurement_cutoff,)):
            pinned_takes |= _measurement_takes(json.loads(text))
        pinned_renders = {
            str(r["render_id"])
            for r in conn.execute("SELECT take_id, render_id FROM takes").fetchall()
            if r["take_id"] in pinned_takes
        }
        pinned_analyses = {
            str(r["analysis_id"])
            for r in conn.execute("SELECT analysis_id, take_id FROM analyses").fetchall()
            if r["take_id"] in pinned_takes
        }
        # A live design keeps its candidates' voice profiles.
        live_candidates = conn.execute(
            "SELECT clip_sha256, record FROM candidates WHERE design_id IN"
            " (SELECT design_id FROM candidates GROUP BY design_id HAVING MAX(last_used_at) >= ?)",
            (cache_cutoff,),
        ).fetchall()
        pinned_profiles = _candidate_profiles(live_candidates)
        plan["takes"] = [t for t in plan["takes"] if t not in pinned_takes]
        plan["renders"] = [r for r in plan["renders"] if r not in pinned_renders]
        plan["analyses"] = [a for a in plan["analyses"] if a not in pinned_analyses]
        plan["profiles"] = [p for p in plan["profiles"] if p not in pinned_profiles]
        return plan

    def _victims(
        self, conn: sqlite3.Connection, items: Mapping[str, list[str]], errors: list[dict[str, str]]
    ) -> list[Path]:
        """The paths of the planned items (a dry run)."""
        paths: list[Path] = []
        for kind, idents in items.items():
            for ident in idents:
                try:
                    paths.append(self._item_path(conn, kind, ident))
                except NarrationError as exc:
                    errors.append({"path": f"{kind}/{ident}", "error": exc.message})
        return paths

    def _item_path(self, conn: sqlite3.Connection, kind: str, ident: str) -> Path:
        if kind == "renders":
            return self._layout.render_dir(ident)
        if kind == "takes":
            return self._layout.take_dir(ident)
        if kind == "analyses":
            row = conn.execute("SELECT rel_path FROM analyses WHERE analysis_id = ?", (ident,)).fetchone()
            return self._layout.abs(row["rel_path"])
        if kind == "profiles":
            return self._layout.profile_dir(ident)
        if kind == "designs":
            return self._layout.design_root(ident)
        if kind == "measurements":
            row = conn.execute("SELECT rel_dir FROM measurements WHERE measurement_key = ?", (ident,)).fetchone()
            return self._layout.abs(row["rel_dir"])
        if kind == "jobs":
            return self._layout.job_dir(ident)
        raise ValueError(kind)

    def _indexed(self, conn: sqlite3.Connection) -> set[str]:
        """Every published folder or file the index names, relative to the root."""
        return {
            *(r[0] for r in conn.execute("SELECT rel_dir FROM renders")),
            *(r[0] for r in conn.execute("SELECT rel_dir FROM takes")),
            *(r[0] for r in conn.execute("SELECT rel_path FROM analyses")),
            *(r[0] for r in conn.execute("SELECT rel_dir FROM measurements")),
            *(r[0] for r in conn.execute("SELECT rel_dir FROM profiles")),
            *(r[0] for r in conn.execute("SELECT rel_dir FROM candidates")),
            *(f"{JOBS}/{r[0]}" for r in conn.execute("SELECT job_id FROM jobs")),
        }

    def _leftovers(self, cutoff: float) -> tuple[list[Path], list[Path]]:
        """Work-in-progress names older than ``cutoff`` (a crash's leftovers), and published-looking folders
        and files the index does not know (a crash between the rename and the index row)."""
        with self._read() as conn:
            indexed = self._indexed(conn)
        root = self._layout.root
        scratch = self._layout.tree(SCRATCH)
        leftovers: list[Path] = []
        for folder, subfolders, names_here in os.walk(root):
            here = Path(folder)
            if here == scratch:
                subfolders[:] = []
                continue
            for name in [*subfolders, *names_here]:
                if name.startswith(TEMP_PREFIXES) and _leftover_time(here / name) < cutoff:
                    leftovers.append(here / name)
            # never enter work in progress, links or junctions (os.walk does not follow symlinks)
            subfolders[:] = [
                s for s in subfolders if not s.startswith(TEMP_PREFIXES) and not os.path.isjunction(here / s)
            ]
        orphans: list[Path] = []
        # Published entries sit at a fixed depth in each cache tree: renders/<ab>/<id>, designs/<id>/<n>, …
        for tree, depth in ((RENDERS, 2), (TAKES, 2), (PROFILES, 2), (DESIGNS, 2), (MEASUREMENTS, 2), (JOBS, 1)):
            for entry in _entries_at(self._layout.tree(tree), depth):
                if self._layout.rel(entry) not in indexed and _mtime(entry) < cutoff:
                    orphans.append(entry)
                elif tree == TAKES and (entry / ANALYSES).is_dir():
                    for analysis in _entries_at(entry / ANALYSES, 1):
                        if self._layout.rel(analysis) not in indexed and _mtime(analysis) < cutoff:
                            orphans.append(analysis)
        return sorted(leftovers), orphans

    def _old_entries(self, base: Path, cutoff: float) -> list[Path]:
        if not base.is_dir():
            return []
        return sorted(p for p in base.iterdir() if _newest_mtime(p) < cutoff)

    def verify(self) -> dict[str, Any]:
        """Re-hash every immutable file against the index (section 15); report what is missing, changed or
        writable, and the database's own integrity check. A path that cannot be checked (one confinement
        refuses, or one that cannot be read) is listed under ``errors``, and the check goes on."""
        rows = self._conn().execute("SELECT rel_path, sha256, size FROM files ORDER BY rel_path").fetchall()
        missing: list[str] = []
        mismatched: list[dict[str, str]] = []
        writable: list[str] = []
        errors: list[dict[str, str]] = []
        for row in rows:
            rel = str(row["rel_path"])
            try:
                path = self._layout.abs(rel)
                if not path.is_file():
                    missing.append(rel)
                    continue
                sha, _ = files.sha256_file(path)
                readonly = files.is_readonly(path)
            except (OSError, NarrationError) as exc:
                errors.append({"path": rel, "error": _describe(exc)})
                continue
            if sha != row["sha256"]:
                mismatched.append({"path": rel, "expected": row["sha256"], "actual": sha})
            if not readonly:
                writable.append(rel)
        integrity = str(self._conn().execute("PRAGMA integrity_check").fetchone()[0])
        return {
            "checked": len(rows),
            "ok": len(rows) - len(missing) - len(mismatched) - len(errors),
            "missing": missing,
            "mismatched": mismatched,
            "writable": writable,
            "errors": errors,
            "database": integrity,
        }

    # ================================================================ helpers
    @staticmethod
    def _must(value: _R | None, what: str, ident: str) -> _R:
        if value is None:
            raise StoreError(f"the {what} {ident} was published but cannot be read back")
        return value


def _describe(exc: BaseException) -> str:
    return exc.message if isinstance(exc, NarrationError) else f"{type(exc).__name__}: {exc}"


def _outside(paths: list[Path], victims: list[Path]) -> list[Path]:
    """``paths`` less those inside a victim's folder: removing the victim removes them, so they are neither
    listed nor counted twice."""
    return [p for p in paths if not any(v in p.parents for v in victims)]


def _measurement_takes(record: Mapping[str, Any]) -> set[str]:
    """The take ids a measurement names: its calibration takes and its ladder's takes."""
    found: set[str] = set()
    for take in record.get("calibration") or ():
        if isinstance(take, Mapping) and isinstance(take.get("take_id"), str):
            found.add(take["take_id"])
    for rung in record.get("ladder") or ():
        for seed in (rung.get("seeds") or ()) if isinstance(rung, Mapping) else ():
            if isinstance(seed, Mapping) and isinstance(seed.get("take_id"), str):
                found.add(seed["take_id"])
    return found


def _candidate_profiles(rows: list[sqlite3.Row]) -> set[str]:
    """The audio sha256s of candidates' voice profiles: each candidate's clip, and the audio its embedded
    profile names (the same audio when the profile was made from the clip)."""
    found: set[str] = set()
    for row in rows:
        found.add(str(row["clip_sha256"]))
        profile = json.loads(row["record"]).get("profile")
        if isinstance(profile, Mapping) and isinstance(profile.get("audio_sha256"), str):
            found.add(profile["audio_sha256"])
    return found


def _hashed_fields(profile: Mapping[str, Any]) -> dict[str, Any]:
    """An engine profile's JSON without the fields its ``hash`` leaves out."""
    return {k: v for k, v in profile.items() if k not in _ENGINE_UNHASHED}


class _Staging:
    """A folder being assembled under a ``.staging-`` name, and the caller's files moved into it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._moved: list[tuple[Path, Path]] = []

    def move_in(self, src: Path, name: str) -> Path:
        """Move the caller's finished file into the folder as ``name`` (read-only)."""
        dst = self.path / name
        self._moved.append((src, dst))
        files.move_into(src, dst, readonly=True)
        return dst

    def return_sources(self) -> bool:
        """Move every file handed in back to its source path, writable again, as the caller had it. False
        if one could not be (it is then left in the staging folder, which is kept)."""
        ok = True
        for src, dst in reversed(self._moved):
            if not os.path.lexists(dst):
                if not os.path.lexists(src):
                    _log.error("%s is neither in %s nor back at %s", src.name, self.path, src)
                    ok = False
                continue
            try:
                files.make_writable(dst)
                files.rename_retrying(dst, src)
            except OSError as exc:
                _log.error("could not move %s back to %s: %s", dst, src, exc)
                ok = False
        return ok


class _Renames:
    """File renames made inside one write transaction, so they can be undone, newest first, if it does not
    commit."""

    def __init__(self) -> None:
        self.done: list[tuple[Path, Path]] = []
        self._identities: list[tuple[int, int] | None] = []

    def rename(self, src: Path, dst: Path, *, attempts: int = files.REPLACE_ATTEMPTS) -> None:
        files.rename_retrying(src, dst, attempts=attempts)
        self.done.append((src, dst))
        self._identities.append(_identity(dst))

    def undo(self) -> None:
        """Put every renamed path back, newest first. A name that no longer holds what was renamed to it is
        left alone: another writer replaced it, which can happen only if SQLite ended the transaction and
        released the lock before the undo (``db.write_txn``). That, and a path that cannot be put back
        (``_put_back``), is logged."""
        while self.done:
            src, dst = self.done.pop()
            identity = self._identities.pop()
            if identity is not None and _identity(dst) != identity:
                _log.error("did not rename %s back to %s: another writer has replaced it since", dst, src)
                continue
            _put_back(dst, src)


def _make_folder(path: Path) -> None:
    """``path.mkdir(parents=True, exist_ok=True)``, tried again if ``gc`` removes an empty parent in between:
    it removes a voice's emptied measurements folder after its commit, outside the lock."""
    for attempt in range(3):
        try:
            path.mkdir(parents=True, exist_ok=True)
            return
        except FileNotFoundError:
            if attempt == 2:
                raise


def _identity(path: Path) -> tuple[int, int] | None:
    """What ``path`` names, as the file system identifies it (volume and file id), or None if it cannot be
    read."""
    try:
        info = os.lstat(path)
    except OSError:
        return None
    return info.st_dev, info.st_ino


def _batch_open(examined: int, started: float) -> bool:
    """Whether a ``gc`` transaction may examine one more item: it always examines one, then stops at
    ``GC_BATCH_ITEMS`` items or ``GC_BATCH_SECONDS``. It counts every item it looked at, collected or not,
    so items that fail to rename (each tried once) close it as surely as items collected."""
    return examined == 0 or (examined < GC_BATCH_ITEMS and time.monotonic() - started < GC_BATCH_SECONDS)


def _discard_trash(trash: Path, final: Path) -> None:
    """Remove what a publish replaced, after its commit. The publish has succeeded, so this is best effort:
    a failure is logged, and ``gc`` removes the ``.trash-`` name later as a leftover."""
    try:
        files.remove_tree(trash)
    except OSError as exc:
        _log.warning("published %s; the replaced copy %s is left for gc: %s", final, trash, exc)


def _give_back(tmp: Path, source: Path | None) -> None:
    """After a failed file publish: move ``tmp`` back to the caller's ``source`` (writable), or discard it
    if the store wrote it itself (``source`` None). If it cannot be moved back it is kept and logged."""
    if source is None:
        files.discard(tmp)
        return
    if not os.path.lexists(tmp):
        return  # an undone rename already failed, and was logged
    try:
        files.make_writable(tmp)
        files.rename_retrying(tmp, source)
    except OSError as exc:
        _log.error("kept %s: it could not be moved back to %s: %s", tmp, source, exc)


def _put_back(src: Path, dst: Path) -> bool:
    """Undo a rename of a failed publish. Logs and returns False if even that fails."""
    try:
        files.rename_retrying(src, dst)
    except OSError as exc:
        _log.error("could not rename %s back to %s: %s", src, dst, exc)
        return False
    return True


def _engine_kind(kind: str) -> str:
    if kind not in get_args(EngineKind):
        raise ValueError(f"engine kind must be one of {get_args(EngineKind)}, got {kind!r}")
    return kind


def _command(row: sqlite3.Row) -> DaemonCommand:
    return DaemonCommand(
        command_id=row["command_id"],
        kind=row["kind"],
        requested_at=row["requested_at"],
        done_at=row["done_at"],
        result=json.loads(row["result"]) if row["result"] is not None else None,
    )


def _leftover_time(path: Path) -> float:
    """When a work-in-progress name was made: the time written in a trash name (a renamed file keeps its old
    modification time, so that says nothing about when it became trash), else the modification time."""
    written = files.trash_time(path.name)
    return written if written is not None else _mtime(path)


def _mtime(path: Path) -> float:
    try:
        return os.lstat(path).st_mtime
    except FileNotFoundError:
        return math.inf


def _newest_mtime(path: Path) -> float:
    """The newest modification time in a tree (a scratch folder in use is not old)."""
    newest = _mtime(path)
    if path.is_dir() and not os.path.islink(path) and not os.path.isjunction(path):
        for child in path.iterdir():
            newest = max(newest, _newest_mtime(child))
    return newest


def _entries_at(base: Path, depth: int) -> list[Path]:
    """The entries exactly ``depth`` levels under ``base`` (1 = directly inside), sorted, skipping work in
    progress. Links and junctions are listed where they sit but never entered."""
    level = [base] if base.is_dir() else []
    for _ in range(depth):
        below: list[Path] = []
        for folder in level:
            if os.path.islink(folder) or os.path.isjunction(folder) or not folder.is_dir():
                continue
            below.extend(p for p in folder.iterdir() if not p.name.startswith(TEMP_PREFIXES))
        level = below
    return sorted(level)
