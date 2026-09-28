"""``narration-admin failures`` (plan.md WP48): every take that failed QA or that a retake replaced, across jobs,
with why; ``--export``; and the failed takes in ``gc``'s listing.

The store is the job engine's test world (``tests.jobs``): a pinned engine, a measured synthetic voice, and fake
workers told to cut some takes short (``TOKEN_CAP_HIT``, a fail). It is built once for this module, and every
test here only reads it: the commands under test change nothing, which one test checks. Every text is invented
for these tests.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sqlite3
import stat
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from narration import keys
from narration.admin import cli
from narration.admin.__main__ import main
from narration.admin.cli import EXIT_FAILED, EXIT_OK, EXIT_USAGE
from narration.admin.failures import FAILURES_JSON_SCHEMA, FAILURES_SCHEMA_ID, INDEX_COLUMNS
from narration.contracts import codes
from narration.contracts.models import JobRecord, Progress
from narration.contracts.schemas import TOOLS_BY_NAME
from narration.platform.testing import StandInPlatform
from narration.store import NarrationStore
from narration.store.store import utc_iso
from tests.admin.conftest import AdminRun, Ran
from tests.jobs.conftest import World, anchor, make_world
from tests.jobs.support import KETTLE, LAMPS, ORCHARD, voice_hash

__all__ = ["anchor"]

DAY = 86_400.0
SQLITE_FILES = frozenset({"narration.sqlite", "narration.sqlite-wal", "narration.sqlite-shm"})


def seed(world: World, text: str, attempt: int) -> int:
    return keys.seed(voice_hash=voice_hash(world.clip_sha256), engine_text=text, attempt=attempt)


def submit_at(world: World, body: dict[str, Any], created_at: str) -> JobRecord:
    """Queue a job as the front-end would, created at ``created_at``."""
    record = JobRecord(
        job_id=keys.Keys().new_job_id(),
        kind="generate",
        request=body,
        request_sha256=hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest(),
        label=None,
        priority="batch",
        status="queued",
        phase=None,
        round=0,
        progress=Progress(done_s=0.0, total_s=0.0, fraction=0.0, segments_done=0, segments_total=0),
        outcome=None,
        error=None,
        idempotency_key=None,
        created_at=created_at,
        updated_at=created_at,
    )
    job, created = world.store.create_job(record)
    assert created
    return job


@dataclass
class Audit:
    """The store with planted failures, and ``narration-admin`` over it.

    ``older`` (created 30 days ago) narrates the orchard alone; every take of it is cut short, so attempt 0
    fails and its one retake, attempt 1, fails too. ``newer`` narrates the lamps, the kettle and the orchard:
    the lamps' attempt 0 is cut short and its retake passes; the kettle passes; the orchard's two attempts come
    from the cache, failed as before.
    """

    world: World
    config: Path
    older: JobRecord
    newer: JobRecord
    platform: StandInPlatform

    @property
    def store_root(self) -> Path:
        return self.world.config.server.store_root

    def __call__(self, *argv: str) -> Ran:
        out, err = io.StringIO(), io.StringIO()
        code = main(
            ["--config", str(self.config), *argv],
            out=out,
            err=err,
            inp=io.StringIO(""),
            platform=lambda: self.platform,
            environ={},
        )
        return Ran(code, out.getvalue(), err.getvalue())

    def json(self, *argv: str) -> dict[str, Any]:
        ran = self("failures", "--json", *argv)
        assert ran.code == EXIT_OK, ran.err
        return json.loads(ran.out)


@pytest.fixture(scope="module")
def audit(tmp_path_factory: pytest.TempPathFactory, anchor: tuple[float, ...]) -> Iterator[Audit]:
    root = tmp_path_factory.mktemp("failures")
    world = make_world(root, anchor)
    try:
        world.faults(
            {"kind": "token_cap", "when": {"text_contains": "lamplighter", "seed": seed(world, LAMPS, 0)}},
            {"kind": "token_cap", "when": {"text_contains": "orchard"}},
        )
        older = submit_at(
            world, world.request(ORCHARD, ids=["orchard"], max_retakes=1), utc_iso(time.time() - 30 * DAY)
        )
        world.run()
        body = world.request(LAMPS, KETTLE, ORCHARD, ids=["lamps", "kettle", "orchard"], max_retakes=1)
        newer = submit_at(world, body, utc_iso(time.time()))
        world.run()
        config = root / "narration.toml"
        config.write_text(
            f"[server]\nstore_root = '{world.config.server.store_root.as_posix()}'\n"
            f"models_root = '{world.config.server.models_root.as_posix()}'\n",
            encoding="utf-8",
        )
        older_done, newer_done = world.store.get_job(older.job_id), world.store.get_job(newer.job_id)
        assert older_done is not None and newer_done is not None
        assert older_done.status == newer_done.status == "completed"
    finally:
        world.engine.close()
        world.pool.close()
        world.store.close()  # quiet: the commands under test are the store's only users from here on
    yield Audit(world=world, config=config, older=older_done, newer=newer_done, platform=StandInPlatform())


def listed(body: dict[str, Any]) -> list[tuple[str, str, int, str | None, bool, int, str | None]]:
    """(job, segment, attempt, verdict, replaced, final attempt, final verdict) of each listed take, in order."""
    return [
        (
            f["job_id"],
            f["segment_id"],
            f["attempt"],
            f["verdict"],
            f["replaced"],
            f["final_take"]["attempt"],
            f["final_take"]["verdict"],
        )
        for f in body["failures"]
    ]


def expected(audit: Audit) -> list[tuple[str, str, int, str | None, bool, int, str | None]]:
    new, old = audit.newer.job_id, audit.older.job_id
    return [
        (new, "lamps", 0, "fail", True, 1, "pass"),
        (new, "orchard", 0, "fail", True, 1, "fail"),
        (new, "orchard", 1, "fail", False, 1, "fail"),
        (old, "orchard", 0, "fail", True, 1, "fail"),
        (old, "orchard", 1, "fail", False, 1, "fail"),
    ]


# ======================================================================== the list


def test_every_failed_or_replaced_take_is_listed_with_its_flags_and_a_path_wp48(audit: Audit) -> None:
    body = audit.json()
    assert listed(body) == expected(audit)  # newest job first; the passing takes are not listed
    for f in body["failures"]:
        attempt = next(
            a
            for item in (audit.newer if f["job_id"] == audit.newer.job_id else audit.older).items
            if item.segment_id == f["segment_id"]
            for a in item.attempts
            if a.attempt == f["attempt"]
        )
        assert (f["take_id"], f["analysis_id"]) == (attempt.take_id, attempt.analysis_id)
        text = LAMPS if f["segment_id"] == "lamps" else ORCHARD
        assert f["seed"] == seed(audit.world, text, f["attempt"])
        assert f["text"] == text
        wav = Path(f["wav"]["path"])
        assert wav.is_file() and hashlib.sha256(wav.read_bytes()).hexdigest() == f["wav"]["sha256"]
        assert codes.TOKEN_CAP_HIT in {flag["code"] for flag in f["flags"]}
        assert {flag["severity"] for flag in f["flags"]} <= {"fail", "warn"}
        assert f["metrics"] is not None and "wer_adj" in f["metrics"] and "spk_sim_anchor" in f["metrics"]
    lamps = body["failures"][0]
    final = next(a for i in audit.newer.items if i.segment_id == "lamps" for a in i.attempts if a.attempt == 1)
    assert lamps["final_take"]["take_id"] == final.take_id


def test_the_list_reads_as_text_by_default_wp48(audit: Audit) -> None:
    ran = audit("failures")
    assert ran.code == EXIT_OK, ran.err
    assert ran.out.startswith("5 take(s) failed QA or were replaced by a retake, in 2 job(s); newest job first.")
    assert ran.out.index(audit.newer.job_id) < ran.out.index(audit.older.job_id)
    assert f"lamps: “{LAMPS}”" in ran.out and f"orchard: “{ORCHARD}”" in ran.out
    assert "replaced; the slot was filled by attempt 1" in ran.out and "(pass)" in ran.out
    assert "not replaced: the last take of its slot" in ran.out
    assert f"fail {codes.TOKEN_CAP_HIT}" in ran.out and "      metrics: " in ran.out
    paths = [line.split("wav: ", 1)[1] for line in ran.out.splitlines() if line.strip().startswith("wav: ")]
    assert len(paths) == 5 and all(Path(p).is_file() for p in paths)
    assert KETTLE not in ran.out  # a take that passed is not a failure


def test_the_json_validates_against_its_schema_wp48(audit: Audit) -> None:
    Draft202012Validator.check_schema(FAILURES_JSON_SCHEMA)
    assert "$ref" not in json.dumps(FAILURES_JSON_SCHEMA)
    body = audit.json()
    errors = [
        f"{'/'.join(map(str, e.absolute_path))}: {e.message}"
        for e in Draft202012Validator(FAILURES_JSON_SCHEMA).iter_errors(body)
    ]
    assert errors == []
    assert body["schema"] == FAILURES_SCHEMA_ID and body["count"] == 5
    assert body["filters"] == {"since": None, "job": None, "voice": None, "code": None}


def test_each_filter_keeps_only_what_it_names_wp48(audit: Audit) -> None:
    every = expected(audit)
    newer = [row for row in every if row[0] == audit.newer.job_id]
    assert listed(audit.json("--job", audit.older.job_id)) == [row for row in every if row[0] == audit.older.job_id]
    since = utc_iso(time.time() - DAY)
    assert listed(audit.json("--since", since)) == newer
    assert listed(audit.json("--since", since[:10])) == newer  # a date is midnight UTC
    assert listed(audit.json("--since", "2000-01-01")) == every
    sha = audit.world.clip_sha256
    assert listed(audit.json("--voice", sha)) == every
    assert listed(audit.json("--voice", sha[:8].upper())) == every
    other = ("1" if sha[0] == "0" else "0") * 8
    assert listed(audit.json("--voice", other)) == []
    assert listed(audit.json("--code", codes.TOKEN_CAP_HIT.lower())) == every
    assert listed(audit.json("--code", codes.EXACT_SPAN_MISMATCH)) == []  # no request here marks a span exact
    both = audit.json("--job", audit.newer.job_id, "--code", codes.TOKEN_CAP_HIT)
    assert listed(both) == newer and both["filters"]["code"] == codes.TOKEN_CAP_HIT
    ran = audit("failures", "--voice", other)
    assert ran.code == EXIT_OK and ran.out.startswith("No take failed QA or was replaced by a retake (--voice")


@pytest.mark.parametrize(
    ("argv", "says"),
    [
        (["--since", "last tuesday"], "is not an ISO date"),
        (["--job", "job_1"], "is not a job id"),
        (["--voice", "not-hex"], "is not a sha256"),
        (["--code", "NOT_A_FLAG"], "is not a flag code"),
        (["--code", codes.RETAKEN], "is never a fail or warn flag"),
    ],
)
def test_a_filter_that_does_not_read_says_what_to_give_wp48(audit: Audit, argv: list[str], says: str) -> None:
    ran = audit("failures", *argv)
    assert ran.code == EXIT_USAGE and says in ran.err and "Give" in ran.err


def test_a_job_the_store_does_not_have_is_named_wp48(audit: Audit) -> None:
    ran = audit("failures", "--job", "job_01JBXQ7Z3M8V4T2R9K6N5P0W1D")
    assert ran.code == EXIT_FAILED and "No job job_01JBXQ7Z3M8V4T2R9K6N5P0W1D" in ran.err and "without --job" in ran.err


def test_no_store_yet_lists_nothing_and_creates_nothing_wp48(admin: AdminRun, config_path: Path) -> None:
    ran = admin("--config", str(config_path), "failures")
    assert ran.code == EXIT_OK and "No store" in ran.out
    body = json.loads(admin("--config", str(config_path), "failures", "--json").out)
    assert body["count"] == 0 and body["failures"] == []
    assert not (config_path.parent / "store").exists()


def test_failures_is_an_operator_command_never_an_mcp_tool_wp48() -> None:
    assert not any("failure" in name for name in TOOLS_BY_NAME)


# ======================================================================== --export


def test_export_writes_each_wav_its_sidecar_and_the_index_wp48(audit: Audit, tmp_path: Path) -> None:
    target = tmp_path / "audit"
    ran = audit("failures", "--export", str(target))
    assert ran.code == EXIT_OK, ran.err
    body = audit.json()
    takes = list(dict.fromkeys(f["take_id"] for f in body["failures"]))
    assert len(takes) == 3  # the orchard's two takes are listed by both jobs, and exported once
    names = {f"{t}.wav" for t in takes} | {f"{t}.json" for t in takes} | {"index.csv"}
    assert set(os.listdir(target)) == names
    for f in body["failures"]:
        assert (target / f"{f['take_id']}.wav").read_bytes() == Path(f["wav"]["path"]).read_bytes()
    for take_id in takes:
        sidecar = json.loads((target / f"{take_id}.json").read_text(encoding="utf-8"))
        assert sidecar["take_id"] == take_id and sidecar["wav"] == f"{take_id}.wav"
        mine = [f for f in body["failures"] if f["take_id"] == take_id]
        assert sidecar["failures"] == mine  # every listing of the take, with its reasons
        assert all(codes.TOKEN_CAP_HIT in {flag["code"] for flag in f["flags"]} for f in sidecar["failures"])
    with open(target / "index.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert tuple(rows[0].keys()) == INDEX_COLUMNS
    assert [(r["job_id"], r["segment_id"], int(r["attempt"]), r["replaced"]) for r in rows] == [
        (job, segment, attempt, "yes" if replaced else "no")
        for job, segment, attempt, _, replaced, _, _ in expected(audit)
    ]
    assert all(codes.TOKEN_CAP_HIT in r["flags"] and r["wav"] == f"{r['take_id']}.wav" for r in rows)
    assert {r["text"] for r in rows} == {LAMPS, ORCHARD}
    assert not [n for n in os.listdir(target) if n.startswith(".tmp-")]

    again = audit("failures", "--export", str(target), "--json")
    assert again.code == EXIT_OK
    exported = json.loads(again.out)["export"]
    assert exported["written"] == [] and sorted(exported["unchanged"]) == sorted(names)


def test_export_never_overwrites_a_different_file_wp48(audit: Audit, tmp_path: Path) -> None:
    target = tmp_path / "notes"
    target.mkdir()
    (target / "index.csv").write_text("the auditor's own notes\n", encoding="utf-8")
    ran = audit("failures", "--export", str(target))
    assert ran.code == EXIT_FAILED and "index.csv" in ran.err and "nothing was exported" in ran.err
    assert os.listdir(target) == ["index.csv"]
    assert (target / "index.csv").read_text(encoding="utf-8") == "the auditor's own notes\n"


def test_export_refuses_a_folder_inside_the_store_wp48(audit: Audit) -> None:
    for inside in (audit.store_root / "exports", audit.store_root):
        ran = audit("failures", "--export", str(inside))
        assert ran.code == EXIT_USAGE and "inside the store" in ran.err and "outside it" in ran.err
    assert not (audit.store_root / "exports").exists()


# ======================================================================== gc lists them apart


class _Later(NarrationStore):
    """The store as a later day sees it: everything was last used over a year ago."""

    def __init__(self, root: Path, platform: Any, **kwargs: Any) -> None:
        super().__init__(root, platform, clock=lambda: time.time() + 400 * DAY, **kwargs)


def test_gc_lists_the_failed_takes_apart_wp48(audit: Audit, monkeypatch: pytest.MonkeyPatch) -> None:
    report = json.loads(audit("gc", "--json").out)
    takes = sorted({f["take_id"] for f in audit.json()["failures"]})
    assert report["dry_run"] is True
    assert report["failures"]["takes"] == 3 and report["failures"]["due"] == []
    assert 29.0 <= report["failures"]["oldest_age_days"] <= 31.0  # the older job was created 30 days ago
    ran = audit("gc")
    assert ran.code == EXIT_OK
    assert "Failed or replaced takes (`narration-admin failures`): 3 are in the store" in ran.out
    assert "This run would remove none of them." in ran.out

    monkeypatch.setattr(cli, "NarrationStore", _Later)
    later = json.loads(audit("gc", "--json").out)
    assert later["dry_run"] is True and set(takes) <= set(later["items"]["takes"])
    assert later["failures"]["due"] == takes and 429.0 <= later["failures"]["oldest_age_days"] <= 431.0
    ran = audit("gc")
    assert ran.code == EXIT_OK and "This run would take 3 of them out of the audit" in ran.out
    assert "failures --export <dir>" in ran.out
    assert all(Path(f["wav"]["path"]).is_file() for f in audit.json()["failures"])  # a dry run removes nothing


# ======================================================================== the store is unchanged


def snapshot(root: Path) -> tuple[dict[str, str], list[str]]:
    """Every file of the store byte for byte (with its mode bits) but SQLite's own, and every database row."""
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_dir():
            files[rel + "/"] = "folder"
        elif path.name not in SQLITE_FILES:
            mode = stat.S_IMODE(path.stat().st_mode)
            files[rel] = f"{hashlib.sha256(path.read_bytes()).hexdigest()} {mode:o}"
    conn = sqlite3.connect(str(root / "narration.sqlite"))
    try:
        rows = list(conn.iterdump())
    finally:
        conn.close()
    return files, rows


def test_the_store_is_unchanged_by_every_failures_command_wp48(audit: Audit, tmp_path: Path) -> None:
    before = snapshot(audit.store_root)
    runs = [
        ["failures"],
        ["failures", "--json"],
        ["failures", "--job", audit.older.job_id],
        ["failures", "--since", "2000-01-01", "--code", codes.TOKEN_CAP_HIT],
        ["failures", "--voice", audit.world.clip_sha256[:8]],
        ["failures", "--export", str(tmp_path / "copy")],
        ["failures", "--export", str(tmp_path / "copy"), "--json"],
        ["gc"],
        ["gc", "--json"],
    ]
    for argv in runs:
        ran = audit(*argv)
        assert ran.code == EXIT_OK, (argv, ran.err)
    after = snapshot(audit.store_root)
    assert after[0] == before[0]  # every file, byte for byte, read-only marks included
    assert after[1] == before[1]  # every row, last-use times included: listing a take is no use of it
