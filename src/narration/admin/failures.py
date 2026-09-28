"""``narration-admin failures``: an audit of failed takes, across jobs (plan.md WP48).

::

    narration-admin failures [--since <ISO date>] [--job <job_id>] [--voice <sha256>] [--code <FLAG>]
                             [--json] [--export <dir>]

**What it lists.** Every take of a ``generate`` or ``analyse`` job that failed QA, or that a retake replaced
(design section 8), newest job first and, within a job, in the request's order. For each: the job, the segment,
the attempt and its seed; the take id and its delivery WAV in the store; each fail and warn flag with its
severity and message; the QA metrics (adjusted WER and word errors, similarity to the voice's anchor, the pace
metrics the analysis has) and the thresholds they were judged by; the take that finally filled the slot, with
its verdict; and the segment's text as the request sent it. A ``measure`` job's calibration and ladder takes
are the measurement's own probes and are not listed.

**Replaced, and the final take**, follow one rule, the job report's too (``narration.qa.report.replacements``):
an attempt counts as replaced only when a later attempt of its take slot has a take (its take and render in
the store, with their files), and the take that finally filled the slot is the slot's last attempt that has
one. A retake whose render failed, or whose take is no longer in the store, replaces nothing.

The filters combine: ``--since`` keeps jobs created at or after a date (UTC unless it names a zone),
``--job`` one job, ``--voice`` the jobs whose voice clip has that sha256 (or starts with it), ``--code`` the
takes with a fail or warn flag of that code (one QA or the aligner can raise: ``ANALYSIS_CODES``). ``--json``
prints the list as data (``FAILURES_SCHEMA_ID``, described by ``FAILURES_JSON_SCHEMA``).

**``--export <dir>``** copies each listed take's WAV to ``<dir>/<take_id>.wav``, beside
``<dir>/<take_id>.json`` (every listing of that take, with its reasons), and writes ``<dir>/index.csv`` (one
row per listing). The bundle names its WAVs by their file name in ``<dir>``, never by a path in the store.
``index.csv`` is UTF-8 with a byte-order mark, so a spreadsheet reads its text right, and a text cell that
starts with ``=``, ``+``, ``-``, ``@``, a tab or a carriage return gets a ``'`` in front, so a spreadsheet
shows it rather than runs it as a formula; the sidecars keep every text exactly. Each file is written under a
temporary name and renamed. A file already there with the same bytes is left as it is; a different file under
one of those names refuses the whole export before anything is written. A ``<dir>`` inside the store is
refused, also when it reaches the store through another name (a link, or a UNC path back to this machine).

**Read-only.** Nothing in the store changes: no row, no file, no last-use time, so listing a take never keeps
it from ``gc``. Jobs come from ``NarrationStore.iter_jobs``, and takes, renders and analyses from its
``peek_*`` reads, which never repair the index: a take whose WAV is missing is listed with "file not in the
store" and left as it is (the store's other readers drop such a row). Nothing new is recorded (the stateless
rule, section 0.2): the list exists only in this command's output. Opening the store runs its idempotent schema
check, as every command does; that touches only SQLite's own file header.

**The segment's text** is the operator's local output: it is printed and exported, and never logged.

**Never an MCP tool:** an operator's view of this machine's cache, as ``gc`` and ``verify`` are.

``narration-admin gc`` lists these takes separately (``retention_view``), so an audit can finish before
retention removes them.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from narration.backend.requests import segments_of
from narration.backend.service import GENERATION_KINDS
from narration.contracts import codes, names
from narration.contracts.errors import NarrationError
from narration.contracts.models import AnalysisRecord, Flag, JobAttempt, JobRecord, JobSegment, TakeRecord
from narration.contracts.serial import to_json
from narration.platform import real_path
from narration.qa.report import replacements
from narration.store import NarrationStore
from narration.store.files import CHUNK, discard, rename_retrying, sha256_file, write_temp
from narration.store.layout import is_under
from narration.store.store import parse_iso, utc_iso
from narration.text import join

from .cli import EXIT_OK, EXIT_USAGE, PROGRAM, Admin, AdminError, Subparsers

FAILURES_SCHEMA_ID: Final = "narration.failures/v1"
"""The schema id of ``failures --json`` and of each export sidecar (operator output; not a record the store
keeps)."""
INDEX_CSV: Final = "index.csv"
METRIC_COLUMNS: Final = (
    "wer_adj",
    "word_errors",
    "spk_sim_anchor",
    "spoken_wpm",
    "expected_spoken_wpm",
    "spoken_cps",
    "articulation_cps",
    "expected_articulation_cps",
    "pause_s",
    "speaking_share",
)
"""The QA metrics ``index.csv`` has a column for (``QaMetrics``' names). The pace QA judges since contracts
1.6.9 is ``articulation_cps`` against ``expected_articulation_cps`` (WP47); a take scored before has them
empty."""
INDEX_COLUMNS: Final = (
    "job_id",
    "job_created_at",
    "segment_id",
    "attempt",
    "seed",
    "take_id",
    "verdict",
    "replaced",
    "final_attempt",
    "final_take_id",
    "final_verdict",
    "flags",
    *METRIC_COLUMNS,
    "wav",
    "sidecar",
    "text",
)
"""The columns of ``index.csv``, one row per listed take. A metric the analysis does not have is empty."""
_REASONS: Final = ("fail", "warn")
ANALYSIS_CODES: Final = frozenset(
    code
    for code in (
        # QA (section 11.1)
        codes.WER_HIGH,
        codes.EXACT_SPAN_MISMATCH,
        codes.TERM_UNVERIFIED,
        codes.SPK_SIM_LOW,
        codes.PACE_FAST,
        codes.PACE_SLOW,
        codes.HEAD_INSERTION,
        codes.END_INSERTION,
        codes.SILENCE_LONG,
        codes.CLIPPING,
        codes.SIGNAL_INVALID,
        codes.TOKEN_CAP_HIT,
        # the aligner (section 11.2)
        codes.CUE_UNALIGNED,
        codes.CUE_LOW_CONFIDENCE,
        codes.CUE_ALIGNMENT_DISAGREE,
        codes.ALIGNMENT_ERROR,
    )
    if set(codes.FLAGS[code].severities) & set(_REASONS)
)
"""The flag codes an analysis can carry as a fail or a warn: QA's and the aligner's (``--code`` takes one)."""
_HEX: Final = re.compile(r"[0-9a-fA-F]{1,64}")
_FORMULA_START: Final = ("=", "+", "-", "@", "\t", "\r")
_DAY: Final = 86_400.0
_SINCE_SLACK_S: Final = _DAY
"""``--since`` is matched to a job's ``created_at`` exactly; the store is asked for the jobs it inserted since a
day before, in case its clock and the front-end's differ (``NarrationStore.iter_jobs``)."""

_NUM_OR_NULL: Final[dict[str, Any]] = {"type": ["number", "null"]}
_STR_OR_NULL: Final[dict[str, Any]] = {"type": ["string", "null"]}
_VERDICT_OR_NULL: Final[dict[str, Any]] = {"enum": ["pass", "warn", "fail", None]}
_FLAG_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "required": ["code", "severity", "message"],
    "properties": {
        "code": {"type": "string"},
        "severity": {"enum": list(_REASONS)},
        "message": {"type": "string"},
        "cue": {"type": "integer"},
        "details": {"type": "object"},
    },
}
_ROW_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "job_id",
        "job_kind",
        "job_status",
        "job_created_at",
        "voice_sha256",
        "segment_id",
        "attempt",
        "seed",
        "round",
        "fresh",
        "take_id",
        "render_id",
        "analysis_id",
        "verdict",
        "replaced",
        "final_take",
        "wav",
        "flags",
        "metrics",
        "thresholds",
        "text",
    ],
    "properties": {
        "job_id": {"type": "string", "pattern": names.id_schema_pattern("job_id")},
        "job_kind": {"type": "string"},
        "job_status": {"type": "string"},
        "job_created_at": {"type": "string"},
        "voice_sha256": {"type": "string"},
        "segment_id": {"type": "string"},
        "attempt": {"type": "integer"},
        "seed": {"type": "integer"},
        "round": {"type": "integer"},
        "fresh": {"type": "boolean"},
        "take_id": {"type": "string", "pattern": names.id_schema_pattern("take_id")},
        "render_id": _STR_OR_NULL,
        "analysis_id": _STR_OR_NULL,
        "verdict": _VERDICT_OR_NULL,
        "replaced": {"type": "boolean"},
        "final_take": {
            "type": "object",
            "additionalProperties": False,
            "required": ["attempt", "take_id", "verdict"],
            "properties": {"attempt": {"type": "integer"}, "take_id": _STR_OR_NULL, "verdict": _VERDICT_OR_NULL},
        },
        "wav": {
            "description": (
                "the take's delivery file; null when the take is not in the store. 'path' is where the store "
                "keeps it (in a sidecar: its file name in the export, or null when it was not exported), and "
                "'present' whether that file is there"
            ),
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": ["path", "sha256", "duration_s", "present"],
            "properties": {
                "path": _STR_OR_NULL,
                "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "duration_s": {"type": "number"},
                "present": {"type": "boolean"},
            },
        },
        "flags": {"type": "array", "items": _FLAG_SCHEMA},
        "metrics": {
            "type": ["object", "null"],
            "required": ["wer_adj", "word_errors", "spk_sim_anchor"],
            "properties": {
                "wer_adj": _NUM_OR_NULL,
                "wer_raw": _NUM_OR_NULL,
                "word_errors": {"type": ["integer", "null"]},
                "spk_sim_anchor": _NUM_OR_NULL,
            },
        },
        "thresholds": {"type": ["object", "null"]},
        "text": {"type": "string"},
    },
}
FAILURES_JSON_SCHEMA: Final[dict[str, Any]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": FAILURES_SCHEMA_ID,
    "type": "object",
    "additionalProperties": False,
    "required": ["schema", "store_root", "filters", "count", "failures"],
    "properties": {
        "schema": {"const": FAILURES_SCHEMA_ID},
        "store_root": {"type": "string"},
        "filters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["since", "job", "voice", "code"],
            "properties": {"since": _STR_OR_NULL, "job": _STR_OR_NULL, "voice": _STR_OR_NULL, "code": _STR_OR_NULL},
        },
        "count": {"type": "integer", "minimum": 0},
        "failures": {"type": "array", "items": _ROW_SCHEMA},
        "export": {
            "type": "object",
            "required": ["dir", "written", "unchanged"],
            "properties": {
                "dir": {"type": "string"},
                "written": {"type": "array", "items": {"type": "string"}},
                "unchanged": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}
"""What ``failures --json`` prints (JSON Schema 2020-12, with no ``$ref``)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class FinalTake:
    """The take that finally filled a slot: its attempt, take id and verdict (the slot's last attempt that has a
    take; the listed take itself when nothing replaced it)."""

    attempt: int
    take_id: str | None
    verdict: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class FailedTake:
    """One take a job lists that failed QA or that a retake replaced, with why (see the module docstring).

    ``wav`` is where the store keeps the take's delivery file, and ``wav_present`` whether the file is there;
    ``wav`` is None when the take itself is not in the store while its job is. ``flags`` holds the fail and
    warn flags of its analysis; ``metrics`` and ``thresholds`` are the analysis's QA numbers as JSON (None
    when the analysis is not in the store).
    """

    job_id: str
    job_kind: str
    job_status: str
    job_created_at: str
    voice_sha256: str
    segment_id: str
    attempt: int
    seed: int
    round: int
    fresh: bool
    take_id: str
    render_id: str | None
    analysis_id: str | None
    verdict: str | None
    replaced: bool
    final: FinalTake
    wav: Path | None
    wav_present: bool
    wav_sha256: str | None
    duration_s: float | None
    flags: tuple[Flag, ...]
    metrics: dict[str, Any] | None
    thresholds: dict[str, Any] | None
    text: str

    @property
    def exported_wav(self) -> str | None:
        """The WAV's file name in an export, or None when there is no file to copy."""
        return f"{self.take_id}.wav" if self.wav is not None and self.wav_present else None

    def as_json(self, *, bundle: bool = False) -> dict[str, Any]:
        """The take as ``failures --json`` lists it (``FAILURES_JSON_SCHEMA``'s items). ``bundle`` names the WAV
        by its file name in an export instead of its path in the store (for the sidecars)."""
        wav = None
        if self.wav is not None:
            path = self.exported_wav if bundle else str(self.wav)
            wav = {"path": path, "sha256": self.wav_sha256, "duration_s": self.duration_s, "present": self.wav_present}
        return {
            "job_id": self.job_id,
            "job_kind": self.job_kind,
            "job_status": self.job_status,
            "job_created_at": self.job_created_at,
            "voice_sha256": self.voice_sha256,
            "segment_id": self.segment_id,
            "attempt": self.attempt,
            "seed": self.seed,
            "round": self.round,
            "fresh": self.fresh,
            "take_id": self.take_id,
            "render_id": self.render_id,
            "analysis_id": self.analysis_id,
            "verdict": self.verdict,
            "replaced": self.replaced,
            "final_take": {"attempt": self.final.attempt, "take_id": self.final.take_id, "verdict": self.final.verdict},
            "wav": wav,
            "flags": [to_json(f) for f in self.flags],
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Filters:
    """``failures``'s filters, checked. ``since`` is Unix seconds; ``voice`` is lower-case hex; ``code`` one of
    ``ANALYSIS_CODES``."""

    since: float | None = None
    job_id: str | None = None
    voice: str | None = None
    code: str | None = None

    def as_json(self) -> dict[str, str | None]:
        """The filters as ``failures --json`` echoes them."""
        return {
            "since": utc_iso(self.since) if self.since is not None else None,
            "job": self.job_id,
            "voice": self.voice,
            "code": self.code,
        }


# ======================================================================== the command


def register(subparsers: Subparsers) -> None:
    """Add ``failures`` (``narration.admin.cli``)."""
    parser = subparsers.add_parser(
        "failures",
        help="list the takes that failed QA or were replaced by a retake, across jobs, for an audit (plan WP48)",
        description=(
            "List every take of a generate or analyse job that failed QA or that a retake replaced, newest job "
            "first: the job, segment, attempt and seed; the take's WAV in the store; each fail and warn flag; "
            "the QA metrics; the take that finally filled the slot; and the segment's text. Reads the store; "
            "changes nothing in it."
        ),
    )
    parser.add_argument(
        "--since", metavar="DATE", help="only jobs created at or after this ISO date or time (UTC unless it says)"
    )
    parser.add_argument("--job", metavar="JOB_ID", help="only this job")
    parser.add_argument("--voice", metavar="SHA256", help="only jobs whose voice clip has this sha256 (or a prefix)")
    parser.add_argument("--code", metavar="FLAG", help="only takes with a fail or warn flag of this QA or aligner code")
    parser.add_argument("--json", action="store_true", help=f"print the list as JSON ({FAILURES_SCHEMA_ID})")
    parser.add_argument(
        "--export",
        metavar="DIR",
        type=Path,
        help="copy each listed take's WAV to DIR with a JSON sidecar of its reasons, plus index.csv",
    )
    parser.set_defaults(handler=run_failures)


def run_failures(admin: Admin, args: argparse.Namespace) -> int:
    """``failures``: see the module docstring."""
    filters = parse_filters(since=args.since, job_id=args.job, voice=args.voice, code=args.code)
    root = admin.config().server.store_root
    target: Path | None = None
    if args.export is not None:
        target = export_dir(args.export, root)
    if not admin.store_exists():
        if args.json:
            admin.say(json.dumps(listing_json([], root, filters), ensure_ascii=False, indent=2))
        else:
            admin.say(f"No store at {root} yet; no take has failed.")
        return EXIT_OK
    store = admin.store()
    failures = collect(store, filters)
    exported: dict[str, list[str]] | None = None
    if target is not None:
        exported = export(failures, target) if failures else {"written": [], "unchanged": []}
    if args.json:
        body = listing_json(failures, store.root, filters)
        if target is not None and exported is not None:
            body["export"] = {"dir": str(target), **exported}
        admin.say(json.dumps(body, ensure_ascii=False, indent=2))
        return EXIT_OK
    for line in render_text(failures, filters):
        admin.say(line)
    if target is not None and exported is not None:
        if not failures:
            admin.say(f"Nothing to export; {target} was not created.")
        else:
            written, same = len(exported["written"]), len(exported["unchanged"])
            kept = f", {same} already there with the same bytes" if same else ""
            admin.say(
                f"\nExported to {target}: {written} file(s) written{kept}: each take's WAV and sidecar, "
                f"and {INDEX_CSV}."
            )
    return EXIT_OK


# ======================================================================== the filters


def parse_filters(
    *, since: str | None = None, job_id: str | None = None, voice: str | None = None, code: str | None = None
) -> Filters:
    """The filters as the operator gave them, checked; ``AdminError`` (``EXIT_USAGE``) saying what to give
    instead when one does not read."""
    when: float | None = None
    if since is not None:
        try:
            value = dt.datetime.fromisoformat(since.strip())
        except ValueError:
            raise AdminError(
                f"--since {since!r} is not an ISO date or time. Give one such as 2026-09-27 or 2026-09-27T18:30:00Z.",
                exit_code=EXIT_USAGE,
            ) from None
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.UTC)
        when = value.timestamp()
    if job_id is not None and not names.is_id("job_id", job_id.strip()):
        raise AdminError(
            f"--job {job_id!r} is not a job id. Give it as submit_job or get_job returned it (job_ and 26 "
            f"characters), or leave --job out to list every job.",
            exit_code=EXIT_USAGE,
        )
    if voice is not None and not _HEX.fullmatch(voice.strip()):
        raise AdminError(
            f"--voice {voice!r} is not a sha256. Give the voice clip's sha256 (64 hex characters) or its first "
            f"characters, as the request's voice.sha256 has it.",
            exit_code=EXIT_USAGE,
        )
    flag_code: str | None = None
    if code is not None:
        flag_code = code.strip().upper()
        if flag_code not in ANALYSIS_CODES:
            why = "is never a fail or warn flag of an analysis" if flag_code in codes.FLAGS else "is not a flag code"
            raise AdminError(
                f"--code {code!r} {why}, so no failure carries it. Give one of the codes QA or the aligner raises: "
                f"{', '.join(sorted(ANALYSIS_CODES))} (design sections 11 and 14).",
                exit_code=EXIT_USAGE,
            )
    return Filters(
        since=when,
        job_id=job_id.strip() if job_id is not None else None,
        voice=voice.strip().lower() if voice is not None else None,
        code=flag_code,
    )


# ======================================================================== collecting


def collect(store: NarrationStore, filters: Filters | None = None) -> list[FailedTake]:
    """Every take that failed QA or that a retake replaced, over the jobs the filters keep, newest job first.

    Reads the store without changing it (the module docstring). ``--job`` naming a job the store does not have
    is an ``AdminError`` saying what to do.
    """
    filters = filters or Filters()
    jobs: Iterable[JobRecord]
    if filters.job_id is not None:
        job = store.get_job(filters.job_id)
        if job is None:
            raise AdminError(
                f"No job {filters.job_id} in the store at {store.root}. A job is kept for [retention] retention_days "
                f"after its last use; run `{PROGRAM} failures` without --job to list the jobs that still are."
            )
        jobs = (job,)
    else:
        since = filters.since - _SINCE_SLACK_S if filters.since is not None else None
        jobs = store.iter_jobs(kinds=GENERATION_KINDS, inserted_since=since)
    out: list[FailedTake] = []
    for job in jobs:
        if job.kind not in GENERATION_KINDS or not _kept(job, filters):
            continue
        texts = _texts(job)
        for item in job.items:
            out.extend(_failed(store, job, item, texts.get(item.segment_id, "")))
    if filters.code is not None:
        out = [f for f in out if any(flag.code == filters.code for flag in f.flags)]
    return out


def _kept(job: JobRecord, filters: Filters) -> bool:
    if filters.since is not None:
        try:
            if parse_iso(job.created_at) < filters.since:
                return False
        except ValueError:
            pass  # a time that does not read is kept: the audit hides nothing it cannot judge
    return filters.voice is None or _voice_sha256(job).startswith(filters.voice)


def _voice_sha256(job: JobRecord) -> str:
    voice = job.request.get("voice")
    sha = voice.get("sha256") if isinstance(voice, Mapping) else None
    return sha.lower() if isinstance(sha, str) else ""


def _texts(job: JobRecord) -> dict[str, str]:
    """Each segment's text as the request sent it (its cues joined as section 7.2 joins them)."""
    try:
        segments = segments_of(job.request)
    except NarrationError:
        return {}
    return {s.segment_id: s.text if s.text is not None else join(c.text for c in s.cues) for s in segments}


def _failed(store: NarrationStore, job: JobRecord, item: JobSegment, text: str) -> Iterator[FailedTake]:
    """The segment's failed or replaced takes (the replacement rule is the module docstring's)."""
    if not any(a.verdict == "fail" or any(f.code == codes.RETAKEN for f in a.flags) for a in item.attempts):
        return  # nothing failed and nothing was retaken: no record needs reading
    takes = {a.attempt: store.peek_take(a.take_id) if a.take_id is not None else None for a in item.attempts}
    with_take = [a for a in item.attempts if _has_take(store, a, takes[a.attempt])]
    last_of = replacements((a.attempt, [to_json(f) for f in a.flags]) for a in with_take)
    by_attempt = {a.attempt: a for a in item.attempts}
    for attempt in item.attempts:
        replaced = attempt.attempt in last_of
        if attempt.take_id is None or (attempt.verdict != "fail" and not replaced):
            continue
        analysis = store.peek_analysis(attempt.analysis_id) if attempt.analysis_id is not None else None
        final = by_attempt[last_of[attempt.attempt]] if replaced else attempt
        yield _failed_take(
            job,
            item,
            attempt,
            final,
            takes[attempt.attempt],
            analysis[0] if analysis is not None else None,
            text,
            replaced=replaced,
        )


def _has_take(store: NarrationStore, attempt: JobAttempt, take: tuple[TakeRecord, bool] | None) -> bool:
    """Whether the attempt produced a take that is in the store: its take and its render, with their files (what
    the job report's ``takes[]`` holds, ``narration.backend.assemble``)."""
    if take is None or not take[1] or attempt.render_id is None:
        return False
    render = store.peek_render(attempt.render_id)
    return render is not None and render[1]


def _failed_take(
    job: JobRecord,
    item: JobSegment,
    attempt: JobAttempt,
    final: JobAttempt,
    take: tuple[TakeRecord, bool] | None,
    analysis: AnalysisRecord | None,
    text: str,
    *,
    replaced: bool,
) -> FailedTake:
    assert attempt.take_id is not None
    record = take[0] if take is not None else None
    return FailedTake(
        job_id=job.job_id,
        job_kind=job.kind,
        job_status=job.status,
        job_created_at=job.created_at,
        voice_sha256=_voice_sha256(job),
        segment_id=item.segment_id,
        attempt=attempt.attempt,
        seed=attempt.seed,
        round=attempt.round,
        fresh=attempt.fresh,
        take_id=attempt.take_id,
        render_id=attempt.render_id,
        analysis_id=attempt.analysis_id,
        verdict=analysis.qa.verdict if analysis is not None else attempt.verdict,
        replaced=replaced,
        final=FinalTake(attempt=final.attempt, take_id=final.take_id, verdict=final.verdict),
        wav=Path(record.delivery.path) if record is not None else None,
        wav_present=take is not None and take[1],
        wav_sha256=record.delivery.sha256 if record is not None else None,
        duration_s=record.delivery.duration_s if record is not None else None,
        flags=_reasons(analysis),
        metrics=to_json(analysis.qa.metrics) if analysis is not None else None,
        thresholds=to_json(analysis.qa.thresholds) if analysis is not None else None,
        text=text,
    )


def _reasons(analysis: AnalysisRecord | None) -> tuple[Flag, ...]:
    """The analysis's fail and warn flags, each once (QA carries the aligner's flags, and so may repeat one)."""
    if analysis is None:
        return ()
    out: list[Flag] = []
    seen: set[tuple[Any, ...]] = set()
    for flag in (*analysis.qa.flags, *analysis.alignment.flags):
        key = (flag.code, flag.severity, flag.cue, flag.message)
        if flag.severity in _REASONS and key not in seen:
            seen.add(key)
            out.append(flag)
    return tuple(out)


# ======================================================================== output


def listing_json(failures: Sequence[FailedTake], root: Path, filters: Filters) -> dict[str, Any]:
    """``failures --json`` (``FAILURES_JSON_SCHEMA``)."""
    return {
        "schema": FAILURES_SCHEMA_ID,
        "store_root": str(root),
        "filters": filters.as_json(),
        "count": len(failures),
        "failures": [f.as_json() for f in failures],
    }


def render_text(failures: Sequence[FailedTake], filters: Filters) -> list[str]:
    """The human-readable list: per job, per segment (with its text), each failed or replaced take."""
    given = [f"--{k} {v}" for k, v in filters.as_json().items() if v is not None]
    among = f" ({', '.join(given)})" if given else ""
    if not failures:
        return [
            f"No take failed QA or was replaced by a retake{among}. Jobs, and their takes, are kept for "
            "[retention] retention_days after their last use."
        ]
    jobs = len({f.job_id for f in failures})
    lines = [
        f"{len(failures)} take(s) failed QA or were replaced by a retake, in {jobs} job(s){among}; newest job first."
    ]
    job_id: str | None = None
    segment: str | None = None
    for f in failures:
        if f.job_id != job_id:
            job_id, segment = f.job_id, None
            voice = f.voice_sha256[:16] or "unknown"
            lines += ["", f"Job {f.job_id} ({f.job_kind}, {f.job_status}), created {f.job_created_at}, voice {voice}"]
        if f.segment_id != segment:
            segment = f.segment_id
            lines.append(f"  {f.segment_id}: “{' '.join(f.text.split())}”")
        lines.append(f"    attempt {f.attempt} (seed {f.seed}) {f.take_id}: {f.verdict or 'not scored'}; {_filled(f)}")
        if f.wav is None:
            lines.append("      wav: the take is not in the store")
        elif not f.wav_present:
            lines.append(f"      wav: file not in the store ({f.wav})")
        else:
            lines.append(f"      wav: {f.wav}")
        if f.metrics is not None:
            lines.append(f"      metrics: {metrics_line(f.metrics)}")
        for flag in f.flags:
            cue = f" (cue {flag.cue})" if flag.cue is not None else ""
            lines.append(f"      {flag.severity} {flag.code}{cue}: {flag.message}")
    return lines


def _filled(f: FailedTake) -> str:
    if not f.replaced:
        return "not replaced: the last take of its slot"
    made = f.final.take_id or "(not made)"
    return f"replaced; the slot was filled by attempt {f.final.attempt} {made} ({f.final.verdict or 'not scored'})"


def metrics_line(metrics: Mapping[str, Any]) -> str:
    """The key QA metrics on one line: adjusted WER and word errors, similarity to the anchor, and whichever
    pace metrics the analysis has (the pace QA judged, in characters per second of speaking time since contracts
    1.6.9; words per minute; characters per second over the voiced span)."""

    def num(key: str, digits: int) -> str | None:
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return f"{value:.{digits}f}"

    parts: list[str] = []
    wer, errors = num("wer_adj", 3), metrics.get("word_errors")
    if wer is not None:
        parts.append(f"wer_adj {wer}" + (f" ({errors} word errors)" if isinstance(errors, int) else ""))
    if (similarity := num("spk_sim_anchor", 4)) is not None:
        parts.append(f"similarity {similarity}")
    if (pace := num("articulation_cps", 2)) is not None:
        expected = num("expected_articulation_cps", 2)
        parts.append(f"pace {pace} cps of speaking" + (f" (expected {expected})" if expected is not None else ""))
    if (wpm := num("spoken_wpm", 0)) is not None:
        expected = num("expected_spoken_wpm", 0)
        parts.append(f"{wpm} spoken wpm" + (f" (expected {expected})" if expected is not None else ""))
    if (cps := num("spoken_cps", 2)) is not None:
        expected = num("expected_spoken_cps", 2)
        parts.append(f"{cps} spoken cps" + (f" (expected {expected})" if expected is not None else ""))
    return " · ".join(parts) or "none"


# ======================================================================== export


def export_dir(given: Path, store_root: Path) -> Path:
    """The export folder, absolute; ``AdminError`` (``EXIT_USAGE``) when it is not a folder, or is the store or
    inside it: by its name, after resolving links, or through another name for the same folder (an existing
    ancestor that ``os.path.samefile`` finds to be the store root, such as a UNC path back to this machine through
    the administrative share of the store's drive)."""
    target = Path(os.path.abspath(given.expanduser()))
    root = Path(os.path.abspath(store_root))
    inside = is_under(target, root) or is_under(Path(real_path(target)), Path(real_path(root)))
    if not inside and os.path.isdir(root):
        for folder in (target, *target.parents):
            try:
                if os.path.isdir(folder) and os.path.samefile(folder, root):
                    inside = True
                    break
            except OSError:
                continue
    if inside:
        raise AdminError(
            f"{target} is inside the store ({root}), which holds only the service's own files. Export to a folder "
            "outside it, for example one in your home or project folder.",
            exit_code=EXIT_USAGE,
        )
    if target.exists() and not target.is_dir():
        raise AdminError(f"{target} is a file, not a folder. Give a folder to export into.", exit_code=EXIT_USAGE)
    return target


@dataclass(frozen=True, slots=True)
class _Output:
    name: str
    sha256: str
    data: bytes | None = None
    source: Path | None = None


def export(failures: Sequence[FailedTake], target: Path) -> dict[str, list[str]]:
    """Copy each listed take's WAV into ``target`` with its sidecar, and write ``index.csv`` (the module
    docstring). Returns the names written and the names already there with the same bytes.

    Every name is checked before anything is written: a different file under one of them refuses the whole
    export (``AdminError``), so nothing is ever overwritten. Each file is written under a temporary name and
    renamed; a WAV's bytes are checked against the take's sha256 as they are copied. A file that cannot be read
    or written is an ``AdminError`` naming it and what to do.
    """
    outputs = _plan(failures)
    same: list[str] = []
    clashes: list[str] = []
    for out in outputs:
        path = _inside(target, out.name)
        if not os.path.lexists(path):
            continue
        try:
            identical = path.is_file() and not path.is_symlink() and sha256_file(path)[0] == out.sha256
        except OSError as exc:
            raise AdminError(
                f"Could not read {path} to compare it with what the export would write ({_why(exc)}). Nothing was "
                "exported. Move that file away, or export into another folder."
            ) from exc
        (same if identical else clashes).append(out.name)
    if clashes:
        listed = ", ".join(clashes[:5]) + (f" and {len(clashes) - 5} more" if len(clashes) > 5 else "")
        raise AdminError(
            f"{target} already has different files named {listed}; nothing was exported, and nothing there was "
            "changed. Export into a new or empty folder, or move those files away first."
        )
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AdminError(
            f"Could not create {target} ({_why(exc)}). Nothing was exported. Export into a folder you can write to."
        ) from exc
    written: list[str] = []
    for out in outputs:
        if out.name not in same:
            _write(target / out.name, out)
            written.append(out.name)
    return {"written": written, "unchanged": same}


def _inside(target: Path, name: str) -> Path:
    """``target / name``, refused unless it resolves to a file directly inside ``target`` (a name is a take id or
    ``index.csv``, but a link already sitting under that name could point anywhere)."""
    path = target / name
    if Path(real_path(path)).parent != Path(real_path(target)):
        raise AdminError(
            f"{path} leads outside {target} (a link?); nothing was exported. Move it away, or export into a new "
            "or empty folder."
        )
    return path


def _plan(failures: Sequence[FailedTake]) -> list[_Output]:
    """What the export writes: per take, its WAV (when its file is in the store) and its sidecar; then the index.
    Every take id is checked before it names a file."""
    by_take: dict[str, list[FailedTake]] = {}
    for f in failures:
        if not names.is_id("take_id", f.take_id):
            raise AdminError(
                f"Job {f.job_id} names {f.take_id!r} as a take of {f.segment_id}, which is not a take id, so no file "
                f"is named after it. Nothing was exported. Run `{PROGRAM} verify` to check the store."
            )
        by_take.setdefault(f.take_id, []).append(f)
    outputs: list[_Output] = []
    for take_id, listed in by_take.items():
        first = listed[0]
        if first.exported_wav is not None and first.wav is not None and first.wav_sha256 is not None:
            outputs.append(_Output(first.exported_wav, first.wav_sha256, source=first.wav))
        outputs.append(_data(f"{take_id}.json", _sidecar(take_id, listed)))
    outputs.append(_data(INDEX_CSV, _index(failures)))
    return outputs


def _data(name: str, data: bytes) -> _Output:
    return _Output(name, hashlib.sha256(data).hexdigest(), data=data)


def _sidecar(take_id: str, listed: Sequence[FailedTake]) -> bytes:
    first = listed[0]
    body = {
        "schema": FAILURES_SCHEMA_ID,
        "take_id": take_id,
        "wav": first.exported_wav,
        "sha256": first.wav_sha256,
        "failures": [f.as_json(bundle=True) for f in listed],
    }
    return (json.dumps(body, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def csv_cell(value: Any) -> str:
    """A value as an ``index.csv`` cell: empty for None; a text that a spreadsheet would run as a formula
    (it starts with ``=``, ``+``, ``-``, ``@``, a tab or a carriage return) gets a ``'`` in front. Numbers are
    written as they are: a spreadsheet reads them as numbers, never as formulas."""
    if value is None:
        return ""
    if isinstance(value, str):
        return "'" + value if value.startswith(_FORMULA_START) else value
    return str(value)


def _index(failures: Sequence[FailedTake]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(INDEX_COLUMNS)
    for f in failures:
        metrics = f.metrics or {}
        row: list[Any] = [
            f.job_id,
            f.job_created_at,
            f.segment_id,
            f.attempt,
            f.seed,
            f.take_id,
            f.verdict,
            "yes" if f.replaced else "no",
            f.final.attempt,
            f.final.take_id,
            f.final.verdict,
            "; ".join(f"{flag.severity} {flag.code}" for flag in f.flags),
            *(metrics.get(key) for key in METRIC_COLUMNS),
            f.exported_wav,
            f"{f.take_id}.json",
            f.text,
        ]
        writer.writerow([csv_cell(cell) for cell in row])
    return buffer.getvalue().encode("utf-8-sig")


def _why(exc: OSError) -> str:
    return exc.strerror or str(exc)


def _write(path: Path, out: _Output) -> None:
    """Write one output under a temporary name beside ``path``, check its bytes, and rename it into place (never
    over a file that appeared meanwhile)."""
    where = f"{path}" + (f" from {out.source}" if out.source is not None else "")
    try:
        if out.source is not None:
            tmp, sha, _ = write_temp(path, _chunks(out.source), readonly=False)
        else:
            assert out.data is not None
            tmp, sha, _ = write_temp(path, out.data, readonly=False)
    except OSError as exc:
        raise AdminError(
            f"Could not write {where} ({_why(exc)}). The files already written are complete; fix that, then "
            "export again into the same folder to finish."
        ) from exc
    try:
        if sha != out.sha256:
            raise AdminError(
                f"{out.source} does not match the sha256 its take records, so it was not exported. Run "
                f"`{PROGRAM} verify` to check the store."
            )
        rename_retrying(tmp, path)
    except FileExistsError:
        discard(tmp)
        raise AdminError(
            f"{path} appeared while the export ran; it was left as it is. The files already written are complete; "
            "export again into a new folder."
        ) from None
    except OSError as exc:
        discard(tmp)
        raise AdminError(
            f"Could not write {path} ({_why(exc)}). The files already written are complete; fix that, then export "
            "again into the same folder to finish."
        ) from exc
    except BaseException:
        discard(tmp)
        raise


def _chunks(path: Path) -> Iterator[bytes]:
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            yield chunk


# ======================================================================== retention (gc)


def retention_view(failures: Sequence[FailedTake], report: Mapping[str, Any]) -> dict[str, Any]:
    """What ``gc`` shows of the failed and replaced takes, from ``collect`` (read before the collection) and the
    store's ``gc`` report: how many are in the store, how old the oldest is (by the earliest job that lists it),
    which of them this run takes out of the audit (``due``: it removes the take itself, or every job that lists
    it), and which only lose their flags and metrics (``lose_analysis``: the take and a job stay, but an analysis
    that one of its listings reads goes)."""
    items: Mapping[str, Collection[str]] = report.get("items", {})
    takes, analyses, jobs = (set(items.get(kind, ())) for kind in ("takes", "analyses", "jobs"))
    try:
        now = parse_iso(str(report.get("now")))
    except ValueError:
        now = None
    made: dict[str, float] = {}
    listed: dict[str, list[FailedTake]] = {}
    for f in failures:
        if f.wav is None or not f.wav_present:
            continue  # the take's file is not in the store: nothing left for retention to remove
        listed.setdefault(f.take_id, []).append(f)
        try:
            created = parse_iso(f.job_created_at)
        except ValueError:
            continue
        made[f.take_id] = min(made.get(f.take_id, created), created)
    due = sorted(t for t, rows in listed.items() if t in takes or all(r.job_id in jobs for r in rows))
    lose = sorted(
        t
        for t, rows in listed.items()
        if t not in due and any(r.analysis_id in analyses for r in rows if r.job_id not in jobs)
    )
    oldest = min(made.values()) if made else None
    return {
        "takes": len(listed),
        "oldest_job_created_at": utc_iso(oldest) if oldest is not None else None,
        "oldest_age_days": round((now - oldest) / _DAY, 1) if oldest is not None and now is not None else None,
        "due": due,
        "lose_analysis": lose,
    }


__all__ = [
    "ANALYSIS_CODES",
    "FAILURES_JSON_SCHEMA",
    "FAILURES_SCHEMA_ID",
    "INDEX_COLUMNS",
    "INDEX_CSV",
    "METRIC_COLUMNS",
    "FailedTake",
    "Filters",
    "FinalTake",
    "collect",
    "csv_cell",
    "export",
    "export_dir",
    "listing_json",
    "metrics_line",
    "parse_filters",
    "register",
    "render_text",
    "retention_view",
    "run_failures",
]
