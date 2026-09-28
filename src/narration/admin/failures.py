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

The filters combine: ``--since`` keeps jobs created at or after a date (UTC unless it names a zone),
``--job`` one job, ``--voice`` the jobs whose voice clip has that sha256 (or starts with it), ``--code`` the
takes with a fail or warn flag of that code. ``--json`` prints the list as data (``FAILURES_SCHEMA_ID``,
described by ``FAILURES_JSON_SCHEMA``).

**``--export <dir>``** copies each listed take's WAV to ``<dir>/<take_id>.wav``, beside
``<dir>/<take_id>.json`` (every listing of that take, with its reasons), and writes ``<dir>/index.csv`` (one
row per listing). Each file is written under a temporary name and renamed. A file already there with the same
bytes is left as it is; a different file under one of those names refuses the whole export before anything is
written. A ``<dir>`` inside the store is refused.

**Read-only.** Everything is read through the store's API (``NarrationStore.list_jobs``, ``get_take_by_id``,
``get_analysis_by_id``), which records no use: listing a take never keeps it from ``gc``. Nothing new is
recorded (the stateless rule, section 0.2): the list is built from what the store already caches, and exists
only in this command's output. Opening the store runs its idempotent schema check, as every command does,
which touches only SQLite's own header; no row and no other file changes. Like every reader of the store,
``get_take_by_id`` forgets an index entry whose file was removed by hand.

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
from collections.abc import Collection, Iterator, Mapping, Sequence
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
"""The schema id of ``failures --json`` (operator output; not a record the store keeps)."""
INDEX_CSV: Final = "index.csv"
METRIC_COLUMNS: Final = ("wer_adj", "word_errors", "spk_sim_anchor", "spoken_wpm", "expected_spoken_wpm", "spoken_cps")
"""The QA metrics ``index.csv`` has a column for (``QaMetrics``' names)."""
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
_HEX: Final = re.compile(r"[0-9a-fA-F]{1,64}")
_DAY: Final = 86_400.0

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
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": ["path", "sha256", "duration_s"],
            "properties": {
                "path": {"type": "string"},
                "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "duration_s": {"type": "number"},
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
    """The take that finally filled a slot: its attempt, its take id (None if it was never made) and verdict."""

    attempt: int
    take_id: str | None
    verdict: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class FailedTake:
    """One take a job lists that failed QA or that a retake replaced, with why (see the module docstring).

    ``wav`` is the delivery file in the store; it and the other fields read from the take's own records are
    None when the take has left the store (``gc`` removed it after the retention period) while its job is
    still there. ``flags`` holds the fail and warn flags of its analysis; ``metrics`` and ``thresholds`` are
    the analysis's QA numbers as JSON.
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
    wav_sha256: str | None
    duration_s: float | None
    flags: tuple[Flag, ...]
    metrics: dict[str, Any] | None
    thresholds: dict[str, Any] | None
    text: str

    def as_json(self) -> dict[str, Any]:
        """The take as ``failures --json`` lists it (``FAILURES_JSON_SCHEMA``'s items)."""
        wav = None
        if self.wav is not None:
            wav = {"path": str(self.wav), "sha256": self.wav_sha256, "duration_s": self.duration_s}
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
    """``failures``'s filters, checked. ``since`` is Unix seconds; ``voice`` is lower-case hex; ``code`` a flag
    code that can be a fail or warn flag."""

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
    parser.add_argument("--code", metavar="FLAG", help="only takes with a fail or warn flag of this code")
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
        known = codes.FLAGS.get(flag_code)
        if known is None or not set(known.severities) & set(_REASONS):
            reasons = ", ".join(sorted(c for c, f in codes.FLAGS.items() if set(f.severities) & set(_REASONS)))
            why = "is not a flag code" if known is None else "is never a fail or warn flag"
            raise AdminError(
                f"--code {code!r} {why}, so no failure carries it. Give one of: {reasons} (design section 14).",
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

    Reads through the store's API and changes nothing (the module docstring). ``--job`` naming a job the
    store does not have is an ``AdminError`` saying what to do.
    """
    filters = filters or Filters()
    if filters.job_id is not None:
        job = store.get_job(filters.job_id)
        if job is None:
            raise AdminError(
                f"No job {filters.job_id} in the store at {store.root}. A job is kept for [retention] retention_days "
                f"after its last use; run `{PROGRAM} failures` without --job to list the jobs that still are."
            )
        jobs: Sequence[JobRecord] = (job,)
    else:
        jobs = store.list_jobs()
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
    last_of = replacements((a.attempt, [to_json(f) for f in a.flags]) for a in item.attempts)
    by_attempt = {a.attempt: a for a in item.attempts}
    for attempt in item.attempts:
        if attempt.take_id is None or (attempt.verdict != "fail" and attempt.attempt not in last_of):
            continue
        take = store.get_take_by_id(attempt.take_id)
        analysis = store.get_analysis_by_id(attempt.analysis_id) if attempt.analysis_id is not None else None
        final = by_attempt.get(last_of.get(attempt.attempt, attempt.attempt), attempt)
        yield _failed_take(job, item, attempt, final, take, analysis, text, replaced=attempt.attempt in last_of)


def _failed_take(
    job: JobRecord,
    item: JobSegment,
    attempt: JobAttempt,
    final: JobAttempt,
    take: TakeRecord | None,
    analysis: AnalysisRecord | None,
    text: str,
    *,
    replaced: bool,
) -> FailedTake:
    assert attempt.take_id is not None
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
        wav=Path(take.delivery.path) if take is not None else None,
        wav_sha256=take.delivery.sha256 if take is not None else None,
        duration_s=take.delivery.duration_s if take is not None else None,
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
        if f.wav is not None:
            lines.append(f"      wav: {f.wav}")
        else:
            lines.append("      wav: no longer in the store (removed after the retention period)")
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
    pace metrics the analysis has (words per minute, and characters per second where measured)."""

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
    if (wpm := num("spoken_wpm", 0)) is not None:
        expected = num("expected_spoken_wpm", 0)
        parts.append(f"{wpm} spoken wpm" + (f" (expected {expected})" if expected is not None else ""))
    if (cps := num("spoken_cps", 2)) is not None:
        expected = num("expected_spoken_cps", 2)
        parts.append(f"{cps} spoken cps" + (f" (expected {expected})" if expected is not None else ""))
    return " · ".join(parts) or "none"


# ======================================================================== export


def export_dir(given: Path, store_root: Path) -> Path:
    """The export folder, absolute; ``AdminError`` (``EXIT_USAGE``) when it is inside the store (the store
    holds only its own files) or is not a folder."""
    target = Path(os.path.abspath(given.expanduser()))
    root = Path(os.path.abspath(store_root))
    if is_under(Path(real_path(target)), Path(real_path(root))) or is_under(target, root):
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
    renamed; a WAV's bytes are checked against the take's sha256 as they are copied.
    """
    outputs = _plan(failures)
    same: list[str] = []
    clashes: list[str] = []
    for out in outputs:
        path = target / out.name
        if not os.path.lexists(path):
            continue
        if path.is_file() and not path.is_symlink() and sha256_file(path)[0] == out.sha256:
            same.append(out.name)
        else:
            clashes.append(out.name)
    if clashes:
        listed = ", ".join(clashes[:5]) + (f" and {len(clashes) - 5} more" if len(clashes) > 5 else "")
        raise AdminError(
            f"{target} already has different files named {listed}; nothing was exported, and nothing there was "
            "changed. Export into a new or empty folder, or move those files away first."
        )
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for out in outputs:
        if out.name not in same:
            _write(target / out.name, out)
            written.append(out.name)
    return {"written": written, "unchanged": same}


def _plan(failures: Sequence[FailedTake]) -> list[_Output]:
    """What the export writes: per take, its WAV (when the store still has it) and its sidecar; then the index."""
    by_take: dict[str, list[FailedTake]] = {}
    for f in failures:
        by_take.setdefault(f.take_id, []).append(f)
    outputs: list[_Output] = []
    for take_id, listed in by_take.items():
        first = listed[0]
        if first.wav is not None and first.wav_sha256 is not None:
            outputs.append(_Output(f"{take_id}.wav", first.wav_sha256, source=first.wav))
        outputs.append(_data(f"{take_id}.json", _sidecar(take_id, listed)))
    outputs.append(_data(INDEX_CSV, _index(failures, by_take)))
    return outputs


def _data(name: str, data: bytes) -> _Output:
    return _Output(name, hashlib.sha256(data).hexdigest(), data=data)


def _sidecar(take_id: str, listed: Sequence[FailedTake]) -> bytes:
    first = listed[0]
    body = {
        "schema": FAILURES_SCHEMA_ID,
        "take_id": take_id,
        "wav": f"{take_id}.wav" if first.wav is not None else None,
        "sha256": first.wav_sha256,
        "failures": [f.as_json() for f in listed],
    }
    return (json.dumps(body, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _index(failures: Sequence[FailedTake], by_take: Mapping[str, Sequence[FailedTake]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(INDEX_COLUMNS)
    for f in failures:
        metrics = f.metrics or {}
        has_wav = by_take[f.take_id][0].wav is not None
        writer.writerow(
            [
                f.job_id,
                f.job_created_at,
                f.segment_id,
                f.attempt,
                f.seed,
                f.take_id,
                f.verdict or "",
                "yes" if f.replaced else "no",
                f.final.attempt,
                f.final.take_id or "",
                f.final.verdict or "",
                "; ".join(f"{flag.severity} {flag.code}" for flag in f.flags),
                *(_cell(metrics.get(key)) for key in METRIC_COLUMNS),
                f"{f.take_id}.wav" if has_wav else "",
                f"{f.take_id}.json",
                f.text,
            ]
        )
    return buffer.getvalue().encode("utf-8")


def _cell(value: Any) -> str:
    return "" if value is None else str(value)


def _write(path: Path, out: _Output) -> None:
    """Write one output under a temporary name beside ``path``, check its bytes, and rename it into place (never
    over a file that appeared meanwhile)."""
    if out.source is not None:
        tmp, sha, _ = write_temp(path, _chunks(out.source), readonly=False)
    else:
        assert out.data is not None
        tmp, sha, _ = write_temp(path, out.data, readonly=False)
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
            f"{path} appeared while the export ran; it was left as it is. Run the export again into a new folder."
        ) from None
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
    and which of them this run removes from the audit. A take leaves the audit when ``gc`` removes the take
    itself, or every job or analysis that lists it."""
    items: Mapping[str, Collection[str]] = report.get("items", {})
    takes, analyses, jobs = (set(items.get(kind, ())) for kind in ("takes", "analyses", "jobs"))
    try:
        now = parse_iso(str(report.get("now")))
    except ValueError:
        now = None
    made: dict[str, float] = {}
    listed: dict[str, list[FailedTake]] = {}
    for f in failures:
        if f.wav is None:
            continue  # the take has already left the store
        listed.setdefault(f.take_id, []).append(f)
        try:
            created = parse_iso(f.job_created_at)
        except ValueError:
            continue
        made[f.take_id] = min(made.get(f.take_id, created), created)
    due = sorted(
        take_id
        for take_id, rows in listed.items()
        if take_id in takes or all(r.job_id in jobs or r.analysis_id in analyses for r in rows)
    )
    oldest = min(made.values()) if made else None
    return {
        "takes": len(listed),
        "oldest_job_created_at": utc_iso(oldest) if oldest is not None else None,
        "oldest_age_days": round((now - oldest) / _DAY, 1) if oldest is not None and now is not None else None,
        "due": due,
    }


__all__ = [
    "FAILURES_JSON_SCHEMA",
    "FAILURES_SCHEMA_ID",
    "INDEX_COLUMNS",
    "INDEX_CSV",
    "METRIC_COLUMNS",
    "FailedTake",
    "Filters",
    "FinalTake",
    "collect",
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
