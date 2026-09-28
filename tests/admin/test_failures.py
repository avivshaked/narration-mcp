"""``narration-admin failures`` (plan.md WP48): every take that failed QA or that a retake replaced, across jobs,
with why; ``--export``; the failed takes in ``gc``'s listing; and the job report agreeing with it.

The store is the job engine's test world (``tests.jobs``): a pinned engine, a measured synthetic voice, and fake
workers told to plant faults. It is built once for this module, and every test that uses it only reads it: the
commands under test change nothing, which one test checks. A test that damages a store builds its own. Every
text is invented for these tests.
"""

from __future__ import annotations

import codecs
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
from narration.admin import failures as failures_module
from narration.admin import gc as gc_module
from narration.admin.__main__ import main
from narration.admin.cli import EXIT_FAILED, EXIT_OK, EXIT_USAGE, AdminError
from narration.admin.failures import (
    ANALYSIS_CODES,
    FAILURES_JSON_SCHEMA,
    FAILURES_SCHEMA_ID,
    INDEX_COLUMNS,
    FailedTake,
    FinalTake,
    csv_cell,
    export,
    export_dir,
    retention_view,
)
from narration.backend.assemble import assemble, segments_json
from narration.contracts import codes
from narration.contracts.models import Flag, JobAttempt, JobRecord, Progress
from narration.contracts.schemas import TOOLS_BY_NAME
from narration.platform.testing import StandInPlatform
from narration.qa.report import report_json
from narration.store import NarrationStore
from narration.store import files as store_files
from narration.store.store import utc_iso
from narration.text import TextPipeline
from tests.admin.conftest import AdminRun, Ran
from tests.jobs.conftest import World, anchor, make_world
from tests.jobs.support import KETTLE, LAMPS, LANTERN, ORCHARD, voice_hash

__all__ = ["anchor"]

DAY = 86_400.0
SQLITE_FILES = frozenset({"narration.sqlite", "narration.sqlite-wal", "narration.sqlite-shm"})
GULLS = "Two gulls argue over a crust of bread on the lighthouse railing."
BELLS = "Somewhere across the water a chapel bell counts out the hour for nobody."
TEXTS = {"lamps": LAMPS, "kettle": KETTLE, "orchard": ORCHARD, "lantern": LANTERN, "gulls": GULLS, "bells": BELLS}
REASON = {
    "lamps": codes.TOKEN_CAP_HIT,
    "orchard": codes.TOKEN_CAP_HIT,
    "gulls": codes.TOKEN_CAP_HIT,
    "bells": codes.TOKEN_CAP_HIT,
    "lantern": codes.HEAD_INSERTION,
}
"""The planted fault each failed or replaced take of a segment carries."""


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


def write_config(world: World) -> Path:
    config = world.root / "narration.toml"
    config.write_text(
        f"[server]\nstore_root = '{world.config.server.store_root.as_posix()}'\n"
        f"models_root = '{world.config.server.models_root.as_posix()}'\n",
        encoding="utf-8",
    )
    return config


def run_admin(config: Path, *argv: str) -> Ran:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["--config", str(config), *argv],
        out=out,
        err=err,
        inp=io.StringIO(""),
        platform=StandInPlatform,
        environ={},
    )
    return Ran(code, out.getvalue(), err.getvalue())


def close(world: World) -> None:
    world.engine.close()
    world.pool.close()
    world.store.close()  # quiet: the commands under test are the store's only users from here on


@dataclass
class Audit:
    """The store with planted failures, and ``narration-admin`` over it. Three jobs, oldest first:

    - ``older`` (created 30 days ago, ``max_retakes`` 1): the orchard, whose every take is cut short, so attempt
      0 fails and so does its one retake;
    - ``newer`` (``max_retakes`` 2): the lamps (attempts 0 and 1 cut short, attempt 2 passes); the kettle
      (passes); the orchard (attempts 0 and 1 from the cache, attempt 2 new, all failed); the lantern (attempt 0
      starts with an extra word, a warn that triggers a retake; attempt 1 passes); the gulls (attempt 0 cut
      short, and its retake's render fails, so it has no take);
    - ``pair`` (``takes`` 2, ``max_retakes`` 1): the bells, every take cut short, so each of the two slots
      fails its attempt and its retake.
    """

    world: World
    config: Path
    older: JobRecord
    newer: JobRecord
    pair: JobRecord

    @property
    def store_root(self) -> Path:
        return self.world.config.server.store_root

    def job(self, job_id: str) -> JobRecord:
        return next(j for j in (self.older, self.newer, self.pair) if j.job_id == job_id)

    def __call__(self, *argv: str) -> Ran:
        return run_admin(self.config, *argv)

    def json(self, *argv: str) -> dict[str, Any]:
        ran = self("failures", "--json", *argv)
        assert ran.code == EXIT_OK, ran.err
        return json.loads(ran.out)


@pytest.fixture(scope="module")
def audit(tmp_path_factory: pytest.TempPathFactory, anchor: tuple[float, ...]) -> Iterator[Audit]:
    world = make_world(tmp_path_factory.mktemp("failures"), anchor)
    try:
        world.faults(
            {"kind": "token_cap", "when": {"text_contains": "lamplighter", "seed": seed(world, LAMPS, 0)}},
            {"kind": "token_cap", "when": {"text_contains": "lamplighter", "seed": seed(world, LAMPS, 1)}},
            {"kind": "token_cap", "when": {"text_contains": "orchard"}},
            {"kind": "head_insertion", "when": {"text_contains": "lantern", "seed": seed(world, LANTERN, 0)}},
            {"kind": "token_cap", "when": {"text_contains": "gulls", "seed": seed(world, GULLS, 0)}},
            {
                "kind": "error",
                "op": "synthesize",
                "code": "RENDER_FAILED",
                "message": "a planted render failure",
                "when": {"text_contains": "gulls", "seed": seed(world, GULLS, 1)},
            },
            {"kind": "token_cap", "when": {"text_contains": "chapel bell"}},
        )
        jobs = [
            (world.request(ORCHARD, ids=["orchard"], max_retakes=1), time.time() - 30 * DAY),
            (
                world.request(
                    LAMPS,
                    KETTLE,
                    ORCHARD,
                    LANTERN,
                    GULLS,
                    ids=["lamps", "kettle", "orchard", "lantern", "gulls"],
                    max_retakes=2,
                ),
                time.time(),
            ),
            (world.request(BELLS, ids=["bells"], takes=2, max_retakes=1), time.time()),
        ]
        done: list[JobRecord] = []
        for body, created in jobs:
            job = submit_at(world, body, utc_iso(created))
            world.run(max_steps=2000)
            finished = world.store.get_job(job.job_id)
            assert finished is not None and finished.status == "completed", finished
            done.append(finished)
    finally:
        close(world)
    yield Audit(world=world, config=write_config(world), older=done[0], newer=done[1], pair=done[2])


Row = tuple[str, str, int, str | None, bool, int, str | None]


def listed(body: dict[str, Any]) -> list[Row]:
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


def expected(audit: Audit) -> list[Row]:
    """Newest job first; within a job, in the request's order, slot by slot."""
    pair, new, old = audit.pair.job_id, audit.newer.job_id, audit.older.job_id
    return [
        (pair, "bells", 0, "fail", True, 2, "fail"),  # slot 1: attempt 0, retaken by attempt 2
        (pair, "bells", 2, "fail", False, 2, "fail"),
        (pair, "bells", 1, "fail", True, 3, "fail"),  # slot 2: attempt 1, retaken by attempt 3
        (pair, "bells", 3, "fail", False, 3, "fail"),
        (new, "lamps", 0, "fail", True, 2, "pass"),  # two failed attempts, the third passed
        (new, "lamps", 1, "fail", True, 2, "pass"),
        (new, "orchard", 0, "fail", True, 2, "fail"),
        (new, "orchard", 1, "fail", True, 2, "fail"),
        (new, "orchard", 2, "fail", False, 2, "fail"),
        (new, "lantern", 0, "warn", True, 1, "pass"),  # a warn that triggers a retake
        (new, "gulls", 0, "fail", False, 0, "fail"),  # its retake has no take: its render failed
        (old, "orchard", 0, "fail", True, 1, "fail"),
        (old, "orchard", 1, "fail", False, 1, "fail"),
    ]


def attempt_of(audit: Audit, job_id: str, segment_id: str, attempt: int) -> JobAttempt:
    item = next(i for i in audit.job(job_id).items if i.segment_id == segment_id)
    return next(a for a in item.attempts if a.attempt == attempt)


# ======================================================================== the list


def test_every_failed_or_replaced_take_is_listed_with_its_flags_and_a_path_wp48(audit: Audit) -> None:
    body = audit.json()
    assert listed(body) == expected(audit)  # the passing takes, and the retake with no take, are not listed
    for f in body["failures"]:
        attempt = attempt_of(audit, f["job_id"], f["segment_id"], f["attempt"])
        assert (f["take_id"], f["analysis_id"]) == (attempt.take_id, attempt.analysis_id)
        final = attempt_of(audit, f["job_id"], f["segment_id"], f["final_take"]["attempt"])
        assert f["final_take"]["take_id"] == final.take_id
        text = TEXTS[f["segment_id"]]
        assert f["seed"] == seed(audit.world, text, f["attempt"]) and f["text"] == text
        wav = Path(f["wav"]["path"])
        assert f["wav"]["present"] and wav.is_file()
        assert hashlib.sha256(wav.read_bytes()).hexdigest() == f["wav"]["sha256"]
        assert REASON[f["segment_id"]] in {flag["code"] for flag in f["flags"]}
        assert {flag["severity"] for flag in f["flags"]} <= {"fail", "warn"}
        assert f["metrics"] is not None and "wer_adj" in f["metrics"] and "spk_sim_anchor" in f["metrics"]
    lantern = next(f for f in body["failures"] if f["segment_id"] == "lantern")
    assert {(flag["code"], flag["severity"]) for flag in lantern["flags"]} >= {(codes.HEAD_INSERTION, "warn")}
    gulls = next(i for i in audit.newer.items if i.segment_id == "gulls")
    retake = gulls.attempts[1]
    assert retake.take_id is None and codes.RETAKEN in {f.code for f in retake.flags}
    assert codes.RENDER_FAILED in {f.code for f in gulls.flags}


def agreed(f: dict[str, Any]) -> tuple[Any, ...]:
    """What the job report and the command must agree on for a listed take."""
    final = f["final_take"]
    return (
        f["segment_id"],
        f["attempt"],
        f["take_id"],
        f["verdict"],
        f["replaced"],
        final["attempt"],
        final["take_id"],
        final["verdict"],
    )


def test_the_report_and_the_command_agree_on_every_job_wp48(audit: Audit) -> None:
    """One rule for both (``narration.qa.report.replacements``): the report reads ``get_results``' takes, the
    command the job's attempts, and a retake with no take (the gulls' attempt 1) replaces nothing in either."""
    body = audit.json()
    with NarrationStore(audit.store_root, StandInPlatform()) as store:
        for job in (audit.older, audit.newer, audit.pair):
            assembled = assemble(
                store, TextPipeline(audit.world.config.text), job, include_words=False, include_transcripts=False
            )
            results = {"job": {"job_id": job.job_id}, "segments": segments_json(assembled.segments)}
            report = [agreed(f) for f in report_json(results)["failures"]]
            command = [agreed(f) for f in body["failures"] if f["job_id"] == job.job_id]
            assert sorted(report, key=str) == sorted(command, key=str), job.job_id
            assert len(report) == len([row for row in expected(audit) if row[0] == job.job_id])


def test_the_list_reads_as_text_by_default_wp48(audit: Audit) -> None:
    ran = audit("failures")
    assert ran.code == EXIT_OK, ran.err
    assert ran.out.startswith("13 take(s) failed QA or were replaced by a retake, in 3 job(s); newest job first.")
    order = [ran.out.index(j.job_id) for j in (audit.pair, audit.newer, audit.older)]
    assert order == sorted(order)
    assert f"lamps: “{LAMPS}”" in ran.out and f"gulls: “{GULLS}”" in ran.out
    assert "replaced; the slot was filled by attempt 2" in ran.out and "(pass)" in ran.out
    assert "not replaced: the last take of its slot" in ran.out
    assert f"fail {codes.TOKEN_CAP_HIT}" in ran.out and f"warn {codes.HEAD_INSERTION}" in ran.out
    assert "      metrics: " in ran.out
    paths = [line.split("wav: ", 1)[1] for line in ran.out.splitlines() if line.strip().startswith("wav: ")]
    assert len(paths) == 13 and all(Path(p).is_file() for p in paths)
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
    assert body["schema"] == FAILURES_SCHEMA_ID and body["count"] == 13
    assert body["filters"] == {"since": None, "job": None, "voice": None, "code": None}


def test_each_filter_keeps_only_what_it_names_wp48(audit: Audit) -> None:
    every = expected(audit)
    recent = [row for row in every if row[0] != audit.older.job_id]
    assert listed(audit.json("--job", audit.older.job_id)) == [row for row in every if row[0] == audit.older.job_id]
    since = utc_iso(time.time() - DAY)
    assert listed(audit.json("--since", since)) == recent
    assert listed(audit.json("--since", since[:10])) == recent  # a date is midnight UTC
    assert listed(audit.json("--since", "2000-01-01")) == every
    sha = audit.world.clip_sha256
    assert listed(audit.json("--voice", sha)) == every
    assert listed(audit.json("--voice", sha[:8].upper())) == every
    other = ("1" if sha[0] == "0" else "0") * 8
    assert listed(audit.json("--voice", other)) == []
    assert listed(audit.json("--code", codes.TOKEN_CAP_HIT.lower())) == [r for r in every if r[1] != "lantern"]
    assert listed(audit.json("--code", codes.HEAD_INSERTION)) == [r for r in every if r[1] == "lantern"]
    assert listed(audit.json("--code", codes.EXACT_SPAN_MISMATCH)) == []  # no request here marks a span exact
    both = audit.json("--job", audit.newer.job_id, "--code", codes.TOKEN_CAP_HIT)
    assert listed(both) == [r for r in every if r[0] == audit.newer.job_id and r[1] != "lantern"]
    assert both["filters"]["code"] == codes.TOKEN_CAP_HIT
    ran = audit("failures", "--voice", other)
    assert ran.code == EXIT_OK and ran.out.startswith("No take failed QA or was replaced by a retake (--voice")


@pytest.mark.parametrize(
    ("argv", "says"),
    [
        (["--since", "last tuesday"], "is not an ISO date"),
        (["--job", "job_1"], "is not a job id"),
        (["--voice", "not-hex"], "is not a sha256"),
        (["--code", "NOT_A_FLAG"], "is not a flag code"),
        (["--code", codes.RETAKEN], "is never a fail or warn flag of an analysis"),
        (["--code", codes.CUE_BOUNDARY_NO_PAUSE], "is never a fail or warn flag of an analysis"),  # info only
        (["--code", codes.SEGMENT_TOO_LONG], "is never a fail or warn flag of an analysis"),  # a text warning
    ],
)
def test_a_filter_that_does_not_read_says_what_to_give_wp48(audit: Audit, argv: list[str], says: str) -> None:
    ran = audit("failures", *argv)
    assert ran.code == EXIT_USAGE and says in ran.err and "Give" in ran.err
    if argv[0] == "--code":
        assert all(code in ran.err for code in ANALYSIS_CODES)


def test_the_codes_are_the_fail_and_warn_codes_of_qa_and_the_aligner_wp48() -> None:
    assert codes.CUE_BOUNDARY_NO_PAUSE not in ANALYSIS_CODES and codes.SPK_OUTLIER not in ANALYSIS_CODES
    assert {codes.TOKEN_CAP_HIT, codes.HEAD_INSERTION, codes.CUE_UNALIGNED, codes.ALIGNMENT_ERROR} <= ANALYSIS_CODES
    assert all(set(codes.FLAGS[c].severities) & {"fail", "warn"} for c in ANALYSIS_CODES)


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
    assert len(takes) == 11  # the orchard's takes are listed by two jobs, and exported once
    names = {f"{t}.wav" for t in takes} | {f"{t}.json" for t in takes} | {"index.csv"}
    assert set(os.listdir(target)) == names
    for f in body["failures"]:
        assert (target / f"{f['take_id']}.wav").read_bytes() == Path(f["wav"]["path"]).read_bytes()
    for take_id in takes:
        sidecar = json.loads((target / f"{take_id}.json").read_text(encoding="utf-8"))
        assert sidecar["take_id"] == take_id and sidecar["wav"] == f"{take_id}.wav"
        mine = [f for f in body["failures"] if f["take_id"] == take_id]
        for f in mine:
            f["wav"]["path"] = f"{take_id}.wav"  # the bundle names its own file, never the store's path
        assert sidecar["failures"] == mine  # every listing of the take, with its reasons
    store_root = str(audit.store_root)
    for name in names - {f"{t}.wav" for t in takes}:
        text = (target / name).read_text(encoding="utf-8-sig")
        assert store_root not in text and store_root.replace("\\", "\\\\") not in text, name
    raw = (target / "index.csv").read_bytes()
    assert raw.startswith(codecs.BOM_UTF8)  # a spreadsheet reads it as UTF-8
    with open(target / "index.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert tuple(rows[0].keys()) == INDEX_COLUMNS
    assert [(r["job_id"], r["segment_id"], int(r["attempt"]), r["replaced"]) for r in rows] == [
        (job, segment, attempt, "yes" if replaced else "no")
        for job, segment, attempt, _, replaced, _, _ in expected(audit)
    ]
    assert all(REASON[r["segment_id"]] in r["flags"] and r["wav"] == f"{r['take_id']}.wav" for r in rows)
    assert {r["text"] for r in rows} == {LAMPS, ORCHARD, LANTERN, GULLS, BELLS}
    assert not [n for n in os.listdir(target) if n.startswith(".tmp-")]

    again = audit("failures", "--export", str(target), "--json")
    assert again.code == EXIT_OK
    exported = json.loads(again.out)["export"]
    assert exported["written"] == [] and sorted(exported["unchanged"]) == sorted(names)


def test_index_cells_a_spreadsheet_would_run_are_neutralised_wp48() -> None:
    for text in ("=1+2", "+a", "-b", "@c", "\td", "\re"):
        assert csv_cell(text) == "'" + text
    assert csv_cell("A plain sentence, with a - inside.") == "A plain sentence, with a - inside."
    assert csv_cell(-0.25) == "-0.25" and csv_cell(3) == "3" and csv_cell(None) == ""


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


def test_export_refuses_another_name_for_the_store_wp48(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, alias = tmp_path / "store", tmp_path / "alias"
    root.mkdir()
    alias.mkdir()
    real = os.path.samefile

    def samefile(a: Any, b: Any) -> bool:  # ``alias`` stands for a UNC path back to this machine: the same folder
        return (Path(a) == alias and Path(b) == root) or real(a, b)

    monkeypatch.setattr(os.path, "samefile", samefile)
    with pytest.raises(AdminError, match="inside the store") as caught:
        export_dir(alias / "audit", root)
    assert caught.value.exit_code == EXIT_USAGE
    assert export_dir(tmp_path / "elsewhere", root) == tmp_path / "elsewhere"


def test_export_says_what_to_do_when_a_file_cannot_be_written_wp48(
    audit: Audit, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder\n", encoding="utf-8")
    ran = audit("failures", "--export", str(blocker / "audit"))
    assert ran.code == EXIT_FAILED and "Could not create" in ran.err and "Nothing was exported" in ran.err

    target = tmp_path / "half"
    real = failures_module.write_temp
    calls: list[Path] = []

    def refuse_the_second(path: Path, data: Any, **kwargs: Any) -> Any:
        calls.append(path)
        if len(calls) == 2:
            raise PermissionError(13, "Access is denied", str(path))
        return real(path, data, **kwargs)

    monkeypatch.setattr(failures_module, "write_temp", refuse_the_second)
    ran = audit("failures", "--export", str(target))
    assert ran.code == EXIT_FAILED and f"Could not write {calls[1]}" in ran.err and "Access is denied" in ran.err
    assert "export again into the same folder to finish" in ran.err
    assert os.listdir(target) == [calls[0].name]  # the first file is complete; no temporary file is left
    monkeypatch.undo()
    ran = audit("failures", "--export", str(target))
    assert ran.code == EXIT_OK and "1 already there with the same bytes" in ran.out


def failed_take(take_id: str = "tk_0123456789abcdef", **changes: Any) -> FailedTake:
    fields: dict[str, Any] = {
        "job_id": "job_01JBXQ7Z3M8V4T2R9K6N5P0W1D",
        "job_kind": "generate",
        "job_status": "completed",
        "job_created_at": "2026-09-01T00:00:00.000Z",
        "voice_sha256": "a" * 64,
        "segment_id": "p01",
        "attempt": 0,
        "seed": 1,
        "round": 0,
        "fresh": True,
        "take_id": take_id,
        "render_id": "rn_0123456789abcdef",
        "analysis_id": "an_0123456789abcdef",
        "verdict": "fail",
        "replaced": False,
        "final": FinalTake(attempt=0, take_id=take_id, verdict="fail"),
        "wav": None,
        "wav_present": False,
        "wav_sha256": None,
        "duration_s": None,
        "flags": (Flag(code=codes.WER_HIGH, severity="fail", message="m"),),
        "metrics": None,
        "thresholds": None,
        "text": "An invented line for a unit test.",
    }
    fields.update(changes)
    return FailedTake(**fields)


def test_export_names_no_file_after_something_that_is_not_a_take_id_wp48(tmp_path: Path) -> None:
    with pytest.raises(AdminError, match="not a take id"):
        export([failed_take("tk_../../outside")], tmp_path / "audit")
    assert not (tmp_path / "audit").exists()


# ======================================================================== gc lists them apart


class _Later(NarrationStore):
    """The store as a later day sees it: everything was last used over a year ago."""

    def __init__(self, root: Path, platform: Any, **kwargs: Any) -> None:
        super().__init__(root, platform, clock=lambda: time.time() + 400 * DAY, **kwargs)


def test_gc_lists_the_failed_takes_apart_wp48(audit: Audit, monkeypatch: pytest.MonkeyPatch) -> None:
    report = json.loads(audit("gc", "--json").out)
    takes = sorted({f["take_id"] for f in audit.json()["failures"]})
    assert report["dry_run"] is True
    assert report["failures"]["takes"] == 11 and report["failures"]["due"] == []
    assert report["failures"]["lose_analysis"] == []
    assert 29.0 <= report["failures"]["oldest_age_days"] <= 31.0  # the older job was created 30 days ago
    ran = audit("gc")
    assert ran.code == EXIT_OK
    assert "Failed or replaced takes (`narration-admin failures`): 11 are in the store" in ran.out
    assert "This run would remove none of them." in ran.out

    monkeypatch.setattr(cli, "NarrationStore", _Later)
    later = json.loads(audit("gc", "--json").out)
    assert later["dry_run"] is True and set(takes) <= set(later["items"]["takes"])
    assert later["failures"]["due"] == takes and 429.0 <= later["failures"]["oldest_age_days"] <= 431.0
    ran = audit("gc")
    assert ran.code == EXIT_OK and "This run would take 11 of them out of the audit" in ran.out
    assert "failures --export <dir>" in ran.out
    assert all(Path(f["wav"]["path"]).is_file() for f in audit.json()["failures"])  # a dry run removes nothing


def test_gc_counts_the_takes_that_would_only_lose_their_analysis_wp48(
    audit: Audit, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {"takes": 2, "oldest_job_created_at": None, "oldest_age_days": 3.0}
    view = {**seen, "due": ["tk_0000000000000001"], "lose_analysis": ["tk_0000000000000002"]}
    monkeypatch.setattr(gc_module, "retention_view", lambda failures, report: view)
    ran = audit("gc")
    assert "1 would lose their flags and metrics" in ran.out and "would take 1 of them out of the audit" in ran.out
    assert json.loads(audit("gc", "--json").out)["failures"] == view


def test_retention_view_parts_what_gc_takes_from_what_it_leaves_wp48() -> None:
    present = {"wav": Path("delivery.wav"), "wav_present": True, "wav_sha256": "b" * 64, "duration_s": 1.0}
    a = failed_take("tk_000000000000000a", job_id="job_A", analysis_id="an_a", **present)
    b1 = failed_take("tk_000000000000000b", job_id="job_B", analysis_id="an_b", **present)
    b2 = failed_take("tk_000000000000000b", job_id="job_C", analysis_id="an_b", **present)
    c = failed_take("tk_000000000000000c", job_id="job_D", analysis_id="an_c", **present)
    gone = failed_take("tk_000000000000000d", job_id="job_D", analysis_id="an_d")  # its file is not there
    now = utc_iso(time.time())
    items = {"takes": ["tk_000000000000000a"], "analyses": ["an_c"], "jobs": ["job_B"]}
    view = retention_view([a, b1, b2, c, gone], {"now": now, "items": items})
    assert view["takes"] == 3  # a take with no file left is not counted
    assert view["due"] == ["tk_000000000000000a"]  # b stays in the audit through job_C
    assert view["lose_analysis"] == ["tk_000000000000000c"]
    items["jobs"] = ["job_B", "job_C"]
    assert retention_view([a, b1, b2, c], {"now": now, "items": items})["due"] == [
        "tk_000000000000000a",
        "tk_000000000000000b",
    ]


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


def test_a_missing_wav_is_listed_and_neither_failures_nor_gc_repairs_the_index_wp48(
    tmp_path: Path, anchor: tuple[float, ...]
) -> None:
    world = make_world(tmp_path, anchor)
    try:
        world.faults({"kind": "token_cap", "when": {"text_contains": "orchard"}})
        job = submit_at(world, world.request(ORCHARD, ids=["orchard"], max_retakes=0), utc_iso(time.time()))
        world.run()
        done = world.store.get_job(job.job_id)
        assert done is not None and done.status == "completed"
        take_id = done.items[0].attempts[0].take_id
        assert take_id is not None
        take = world.store.get_take_by_id(take_id)
        assert take is not None
        wav = Path(take.delivery.path)
    finally:
        close(world)
    config = write_config(world)
    store_files.discard(wav)  # by hand, outside the service

    def rows() -> dict[str, int]:
        conn = sqlite3.connect(str(world.config.server.store_root / "narration.sqlite"))
        try:
            tables = ("jobs", "renders", "takes", "analyses", "files")
            return {t: int(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) for t in tables}
        finally:
            conn.close()

    before, verified = rows(), run_admin(config, "verify", "--json")
    assert verified.code == EXIT_FAILED and wav.name in verified.out  # verify sees the file missing
    for argv in (
        ["gc"],
        ["gc", "--json"],
        ["failures"],
        ["failures", "--json"],
        ["failures", "--export", str(tmp_path / "audit")],
    ):
        ran = run_admin(config, *argv)
        assert ran.code == EXIT_OK, (argv, ran.err)
    assert rows() == before  # no index row was dropped
    assert run_admin(config, "verify", "--json").out == verified.out
    listing = run_admin(config, "failures").out
    assert f"wav: file not in the store ({wav})" in listing
    (row,) = json.loads(run_admin(config, "failures", "--json").out)["failures"]
    assert row["take_id"] == take_id and row["wav"]["present"] is False and Path(row["wav"]["path"]) == wav
    assert codes.TOKEN_CAP_HIT in {f["code"] for f in row["flags"]}  # its analysis is still read
    assert sorted(os.listdir(tmp_path / "audit")) == sorted([f"{take_id}.json", "index.csv"])
    sidecar = json.loads((tmp_path / "audit" / f"{take_id}.json").read_text(encoding="utf-8"))
    assert sidecar["wav"] is None and sidecar["failures"][0]["wav"]["path"] is None
