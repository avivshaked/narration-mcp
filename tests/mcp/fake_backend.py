"""A fake ``Backend`` for the front-end tests: canned results built from the contract records.

Every result is built with ``serial.to_json`` from the records of ``narration.contracts.models`` (or, where a
tool's result is not a record, from the fields its output schema names), so a result that passes here is
one the real backend (WP36) can produce. File paths are under ``store_root`` and are never written: the
fake touches no disk.

Knobs: ``raise_for`` (tool -> exception), ``result_for`` (tool -> a result to return instead),
``job_status`` (``completed``, ``failed`` or ``running``), ``progress_steps`` (what ``get_job`` reports
while it waits), and ``block_get_job`` (``get_job`` waits until cancelled). ``calls`` records every call.
"""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal, Union, get_args, get_origin

import anyio

from narration import __version__
from narration.config import LimitsConfig
from narration.contracts import codes, models
from narration.contracts.errors import NarrationError
from narration.contracts.interfaces import ProgressCallback, ResourceContent
from narration.contracts.names import (
    ENGINE_PROFILE_BASE,
    MODEL_ALIGNER,
    SPEC_REVISION,
    TEXT_CHECKS_VERSION,
)
from narration.contracts.serial import to_json

JOB_ID = "job_01jbxq7z3m8v4t2r9k6n5p0w1c"
DESIGN_ID = "01jby0000000000000000000aa"
TAKE_ID = "tk_8c41d2e07a9b3f55"
VOICE_HASH = "sha256:" + "3f" * 32
ENGINE = models.EngineRef(id=ENGINE_PROFILE_BASE, hash="sha256:" + "9e" * 32)
UPDATED_AT = "2026-09-26T12:00:00Z"
TERMINAL_TTL_MS = 86_400_000
RUNNING_TTL_MS = 2_000

JobStatusKnob = Literal["completed", "failed", "running"]


# ---------------------------------------------------------------- generic records


def sample(tp: Any) -> Any:
    """A value of type ``tp`` with every field filled (as ``tests/contracts`` builds records)."""
    if tp is Any:
        return 1
    scalars: dict[Any, Any] = {str: "x", int: 1, float: 0.5, bool: True, Path: Path("x")}
    if tp in scalars:
        return scalars[tp]
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        return sample(next(a for a in get_args(tp) if a is not type(None)))
    if origin is Literal:
        return get_args(tp)[0]
    if dataclasses.is_dataclass(tp):
        return record(tp)  # pyright: ignore[reportArgumentType]
    if origin is tuple:
        args = get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return (sample(args[0]),)
        return tuple(sample(a) for a in args)
    if origin is dict:
        return {"k": sample(get_args(tp)[1])}
    raise TypeError(tp)


def record[T](cls: type[T], **overrides: Any) -> T:
    """A record of ``cls``: a field with a default other than None keeps it (a schema id stays the record's
    own), every other field is sampled, then ``overrides`` apply."""
    hints = typing.get_type_hints(cls)
    values: dict[str, Any] = {}
    for f in dataclasses.fields(cls):  # pyright: ignore[reportArgumentType]
        has_default = f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
        if not has_default or f.default is None:
            values[f.name] = sample(hints[f.name])
    values.update(overrides)
    return cls(**values)


# ---------------------------------------------------------------- the fake


class FakeBackend:
    """Implements ``narration.contracts.interfaces.Backend`` with canned, schema-valid results."""

    def __init__(self, store_root: Path, *, job_status: JobStatusKnob = "completed") -> None:
        self.store_root = store_root
        self.job_status: JobStatusKnob = job_status
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.raise_for: dict[str, Exception] = {}
        self.result_for: dict[str, Any] = {}
        self.progress_steps: list[tuple[float, float | None, str | None]] = []
        self.block_get_job = False
        self.get_job_progress: list[ProgressCallback | None] = []
        self.wait_cancelled = anyio.Event()
        self.wait_started = anyio.Event()
        self.resource_reads: list[str] = []

    # ------------------------------------------------------------ helpers

    def path(self, *parts: str) -> str:
        """An absolute path under the store root, as the store reports it."""
        return str(self.store_root.joinpath(*parts))

    def _enter(self, tool: str, args: Mapping[str, Any]) -> Any:
        self.calls.append((tool, dict(args)))
        if tool in self.raise_for:
            raise self.raise_for[tool]
        return self.result_for.get(tool)

    def job_error(self) -> models.Error:
        return NarrationError(codes.ENGINE_DRIFT, "the canary similarity is below its threshold").error

    def take(self) -> models.TakeResult:
        delivery = models.DeliveryAudio(
            path=self.path("takes", "8c", TAKE_ID, "delivery.wav"),
            sha256="ab" * 32,
            sample_rate=48_000,
            samples=420_480,
            duration_s=8.76,
        )
        return record(
            models.TakeResult,
            take_id=TAKE_ID,
            render_id="rn_77e0c4a1b2d93f08",
            delivery=delivery,
            cues=(
                models.CueTiming(index=0, start_s=0.08, end_s=3.02, confidence=0.93),
                models.CueTiming(index=1, start_s=None, end_s=None, confidence=None),
            ),
        )

    def job_json(self) -> dict[str, Any]:
        """``get_job``'s result for the fake's one job."""
        running = self.job_status == "running"
        progress = models.Progress(
            done_s=31.5 if running else 128.0,
            total_s=128.0,
            fraction=0.25 if running else 1.0,
            segments_done=1 if running else 4,
            segments_total=4,
        )
        job: dict[str, Any] = {
            "job_id": JOB_ID,
            "kind": "generate",
            "label": None,
            "status": self.job_status,
            "phase": "rendering" if running else None,
            "round": 0,
            "outcome": "all_passed" if self.job_status == "completed" else None,
            "progress": to_json(progress),
            "eta_s": 90.0 if running else None,
            "queue_position": None,
            "poll_after_s": 5.0 if running else 0.0,
            "message": "Round 0: rendering p03 take 2/3" if running else None,
            "updated_at": UPDATED_AT,
        }
        if self.job_status == "failed":
            job["error"] = to_json(self.job_error())
        return job

    # ------------------------------------------------------------ the Backend protocol

    async def get_server_status(self, args: Mapping[str, Any]) -> dict[str, Any]:
        if (override := self._enter("get_server_status", args)) is not None:
            return override
        gpu = models.GpuStatus(name=None, total_mb=24_564, free_mb=19_000, in_use=False, holder=None, unload_in_s=None)
        return {
            "version": __version__,
            "spec_revision": SPEC_REVISION,
            "store_root": str(self.store_root),
            "daemon": {"state": "idle", "pid": 4242, "current_job": None, "workers": []},
            "gpu": to_json(gpu),
            "cpu_threads": 8,
            "queue": {"length": 0, "jobs": []},
            "admission": {
                "accepting": True,
                "queue": {"length": 0, "max": 20, "est_drain_s": None},
                "rate": {"remaining": 10, "resets_in_s": 60.0},
                "gpu": {"in_use": False, "holder": None, "free_mb": 19_000, "need_mb": {}, "waiting_since": None},
            },
            "engine_profiles": [
                {"id": ENGINE.id, "hash": ENGINE.hash, "installed": True, "env_ok": True, "determinism_tier": None}
            ],
            "capabilities": {
                "controls": {"pace": False, "context": False, "instruct": False},
                "text_modes": ["spoken"],
            },
            "limits": to_json(LimitsConfig()),
            "text_checks_version": TEXT_CHECKS_VERSION,
            "alignment": {
                "method_id": None,
                "model": MODEL_ALIGNER,
                "revision": None,
                "measured_error": {"p50_s": None, "p95_s": None, "n": None, "benchmark": None, "by_kind": None},
                "benchmark": None,
                "measured_at": None,
            },
        }

    async def release_gpu(self, args: Mapping[str, Any]) -> dict[str, Any]:
        if (override := self._enter("release_gpu", args)) is not None:
            return override
        return {"released": True, "holder_before": "qwen", "busy_job": None}

    async def get_job(self, args: Mapping[str, Any], progress: ProgressCallback | None) -> dict[str, Any]:
        self.get_job_progress.append(progress)
        if (override := self._enter("get_job", args)) is not None:
            return override
        if progress is not None:
            for step in self.progress_steps:
                await progress(*step)
        if self.block_get_job:
            self.wait_started.set()
            try:
                await anyio.sleep_forever()
            finally:
                self.wait_cancelled.set()
        return self.job_json()

    async def get_results(self, args: Mapping[str, Any]) -> dict[str, Any]:
        if (override := self._enter("get_results", args)) is not None:
            return override
        segment = record(
            models.SegmentResult,
            segment_id="p03",
            status="passed" if self.job_status == "completed" else "error",
            suggested_take_id=TAKE_ID,
            takes=(self.take(),),
        )
        job: dict[str, Any] = {
            "job_id": JOB_ID,
            "kind": "generate",
            "status": self.job_status,
            "outcome": "all_passed" if self.job_status == "completed" else None,
            "label": None,
            "error": to_json(self.job_error()) if self.job_status == "failed" else None,
        }
        return {
            "job": job,
            "voice": {"voice_hash": VOICE_HASH, "clip_sha256": "5b" * 32},
            "engine_profile": to_json(ENGINE),
            "measurement": {"max_segment_chars": 450, "max_segment_seconds": 31.5, "measured_at": "2026-09-20"},
            "segments": [to_json(segment)],
            "consistency": to_json(models.Consistency(min=0.978, median=0.986)),
            "listen_first": [
                to_json(
                    models.ListenFirstItem(
                        segment_id="p03", take_id=TAKE_ID, cue=1, reason="CUE_UNALIGNED", from_s=None, to_s=None
                    )
                )
            ],
            "report_md": self.path("jobs", JOB_ID, "report.md"),
            "licence": {"generation_model": "Apache-2.0"},
        }

    async def cancel_job(self, args: Mapping[str, Any]) -> dict[str, Any]:
        if (override := self._enter("cancel_job", args)) is not None:
            return override
        return {"status": "cancelling", "completed": False}

    async def design_voice(self, args: Mapping[str, Any]) -> dict[str, Any]:
        if (override := self._enter("design_voice", args)) is not None:
            return override
        lint = models.LintResult(
            policy="warn",
            findings=(models.LintFinding(phrase="Not gravelly", offset=21, suggestion="smooth, clean tone"),),
            note="Negated qualities tend to come out as that quality.",
        )
        return {
            "job_id": JOB_ID,
            "status": "queued",
            "poll_after_s": 2.0,
            "design_id": DESIGN_ID,
            "lint": to_json(lint),
        }

    async def profile_voice(self, args: Mapping[str, Any]) -> dict[str, Any]:
        if (override := self._enter("profile_voice", args)) is not None:
            return override
        return {"job_id": JOB_ID, "status": "queued", "poll_after_s": 1.0}

    async def measure_voice(self, args: Mapping[str, Any]) -> dict[str, Any]:
        if (override := self._enter("measure_voice", args)) is not None:
            return override
        measurement = record(models.MeasurementRecord, voice_hash=VOICE_HASH, engine_profile=ENGINE)
        return {
            "job_id": JOB_ID,
            "status": "completed",
            "poll_after_s": 0.0,
            "voice_hash": VOICE_HASH,
            "measurement": to_json(measurement),
        }

    async def check_text(self, args: Mapping[str, Any]) -> dict[str, Any]:
        if (override := self._enter("check_text", args)) is not None:
            return override
        segments: list[dict[str, Any]] = []
        for segment in args["segments"]:
            cues = segment.get("cues") or [{"text": segment["text"]}]
            spoken = " ".join(c["text"] for c in cues)
            text = models.SegmentText(
                segment_id=segment["segment_id"],
                cues=tuple(
                    models.CueText(
                        index=i,
                        received=c["text"],
                        spoken=c["text"],
                        engine=c["text"],
                        spoken_span=(0, len(c["text"])),
                        engine_span=(0, len(c["text"])),
                    )
                    for i, c in enumerate(cues)
                ),
                spoken_text=spoken,
                engine_text=spoken,
                spoken_chars=len(spoken),
            )
            segments.append({**to_json(text), "max_segment_chars": None, "over_by_chars": None, "est_duration_s": None})
        return {
            "voice_hash": None,
            "measurement": None,
            "segments": segments,
            "text_checks_version": TEXT_CHECKS_VERSION,
        }

    async def audition_pronunciation(self, args: Mapping[str, Any]) -> dict[str, Any]:
        if (override := self._enter("audition_pronunciation", args)) is not None:
            return override
        return {"job_id": JOB_ID, "status": "queued", "poll_after_s": 2.0}

    async def submit_job(self, args: Mapping[str, Any]) -> dict[str, Any]:
        if (override := self._enter("submit_job", args)) is not None:
            return override
        dry_run = bool(args.get("options", {}).get("dry_run", False))
        warning = models.Flag(
            code=codes.SEGMENT_TOO_LONG,
            severity="warn",
            message="the segment is longer than the voice's reliable length; it is still rendered",
            segment_id=args["segments"][0]["segment_id"],
        )
        return {
            "job_id": None if dry_run else JOB_ID,
            "status": "planned" if dry_run else "queued",
            "voice_hash": VOICE_HASH,
            "engine_profile": to_json(ENGINE),
            "plan": {
                "segments_total": len(args["segments"]),
                "segments_cached": 0,
                "renders_needed": len(args["segments"]),
                "deliveries_needed": len(args["segments"]),
                "analyses_needed": len(args["segments"]),
                "est_audio_s": 8.0,
                "est_wall_s": 40.0,
                "queue_position": 0,
            },
            "poll_after_s": 2.0,
            "warnings": [to_json(warning)],
        }

    async def read_resource(self, uri: str) -> ResourceContent:
        self.resource_reads.append(uri)
        if "read_resource" in self.raise_for:
            raise self.raise_for["read_resource"]
        terminal = self.job_status != "running"
        job_ttl = TERMINAL_TTL_MS if terminal else RUNNING_TTL_MS
        known: dict[str, tuple[str, str, int]] = {
            "narration://status": ("application/json", json.dumps({"daemon": {"state": "idle"}}), 5_000),
            f"narration://jobs/{JOB_ID}": ("application/json", json.dumps(self.job_json()), job_ttl),
            f"narration://jobs/{JOB_ID}/report": ("text/markdown", f"# Job {JOB_ID}\n\nAll passed.\n", job_ttl),
            f"narration://designs/{DESIGN_ID}": ("application/json", json.dumps({"design_id": DESIGN_ID}), 60_000),
            f"narration://takes/{TAKE_ID}": ("application/json", json.dumps(to_json(self.take())), TERMINAL_TTL_MS),
            f"narration://measurements/{VOICE_HASH}": (
                "application/json",
                json.dumps({"voice_hash": VOICE_HASH}),
                60_000,
            ),
        }
        if uri not in known:
            raise NarrationError(codes.NOT_FOUND, "no such resource", details={"uri": uri})
        mime_type, text, ttl_ms = known[uri]
        return ResourceContent(uri=uri, mime_type=mime_type, text=text, ttl_ms=ttl_ms)


# ---------------------------------------------------------------- valid arguments for every tool

VOICE: Final[dict[str, str]] = {
    "path": "C:\\voices\\narrator.wav",
    "sha256": "5b" * 32,
    "transcript": "Before dawn, the reef belongs to the shrimp.",
}

VALID_ARGUMENTS: Final[dict[str, dict[str, Any]]] = {
    "get_server_status": {},
    "release_gpu": {},
    "get_job": {"job_id": JOB_ID},
    "get_results": {"job_id": JOB_ID},
    "cancel_job": {"job_id": JOB_ID, "reason": "changed my mind"},
    "design_voice": {"name": "reef narrator", "description": "A warm, low, unhurried voice.", "takes": 3},
    "profile_voice": {"audio": {"path": VOICE["path"], "sha256": VOICE["sha256"]}},
    "measure_voice": {"voice": VOICE},
    "check_text": {"segments": [{"segment_id": "p03", "cues": [{"text": "Before dawn."}, {"text": "By sunrise."}]}]},
    "audition_pronunciation": {
        "voice": VOICE,
        "term": "Ossavine",
        "variants": [{"label": "a", "respell": "Oss-a-veen"}, {"label": "b", "respell": "Oh-sa-vine"}],
    },
    "submit_job": {
        "voice": VOICE,
        "segments": [{"segment_id": "p03", "cues": [{"text": "Before dawn, the reef belongs to the shrimp."}]}],
        "options": {"takes": 2},
    },
}
