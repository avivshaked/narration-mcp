"""A ``JobRunner`` that drives the fake worker, for testing the daemon without the job engine (plan.md WP30).

``FakeWorkerRunner`` is not the job engine (WP31): it has no scheduler, no post-processing, no QA and no
retakes. It does just enough, through the real seam, to exercise the daemon's process management: it
claims queued jobs, renders each segment's text through the ``qwen`` group's worker (the fake role, with
``--fake-workers``) one segment per ``step``, publishes each render through the store, and completes the
job. It keeps the seam's contract: it reports the job and its phases to the host, stops at a segment
boundary, gives its job back on shutdown, and removes the scratch files of the segment it abandoned.

A job's ``request`` must have ``segments``: ``[{"segment_id": "p01", "text": "…"}, …]``. A segment whose
render is already in the store is skipped, so a job given back by a ``stop`` resumes where it stopped.

Each ``synthesize`` and ``design`` passes its own ``max_new_tokens`` (DC-4, ``names.max_new_tokens_for``
with the configured ``[engines.qwen3_base]`` rule), as the job engine will.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any, Final

from narration import keys
from narration.contracts import codes, names
from narration.contracts.errors import NarrationError, WorkerCrashed, WorkerFailure, WorkerTimeout
from narration.contracts.models import (
    CanaryRecord,
    JobRecord,
    Licence,
    Progress,
    RawAudio,
    RenderEngine,
    RenderRecord,
    RenderVoice,
)
from narration.contracts.names import GpuHolder

from .seam import RunnerHost, ShutdownReason, return_job

log = logging.getLogger(__name__)

GROUP: Final[GpuHolder] = "qwen"
SCRATCH_DIR: Final = "daemon-fake"
"""The runner's folder under ``scratch/``: one subfolder per job, plus the voice clip."""
ENGINE_PROFILE_ID: Final = "fake-qwen.p1"
ENGINE_PROFILE_HASH: Final = names.HASH_PREFIX + hashlib.sha256(b"narration.daemon.testing fake engine").hexdigest()
MODEL_REPO: Final = "narration-worker/fake"
MODEL_REVISION: Final = hashlib.sha1(b"narration-worker fake 1").hexdigest()
VOICE_DESCRIPTION: Final = "A calm, even narrator, synthesised by the fake worker."
VOICE_SEED: Final = 2001
REQUEST_TIMEOUT_S: Final = 120.0
MAX_FAILURES: Final = 2
"""A segment whose render fails this many times (a crash, a timeout, an error) fails the job."""


class FakeWorkerRunner:
    """See the module docstring. It keeps its state between steps; one instance serves one daemon."""

    def __init__(self) -> None:
        self._job: JobRecord | None = None
        self._segments: list[tuple[str, str]] = []
        self._next = 0
        self._failures = 0
        self._voice: tuple[str, Path] | None = None
        self._prepared_in: int | None = None

    # ------------------------------------------------------------------ the JobRunner protocol
    def step(self, host: RunnerHost) -> bool:
        if self._job is None:
            job = host.store.claim_next_job(host.holder)
            if job is None:
                return False
            self._take(host, job)
            return True
        job = self._job
        segment_id, text = self._segments[self._next]
        try:
            self._render(host, job, segment_id, text)
        except WorkerFailure as exc:
            if host.should_stop():
                return True
            if exc.code == codes.BACKEND_NOT_INSTALLED:
                self._fail(host, NarrationError(codes.BACKEND_NOT_INSTALLED, exc.message, details=exc.details))
                return True
            self._failed_once(host, f"{exc.code}: {exc.message}")
            return True
        except (WorkerCrashed, WorkerTimeout) as exc:
            if host.should_stop():
                return True
            self._failed_once(host, str(exc))
            return True
        self._next += 1
        self._failures = 0
        done = self._next
        total = len(self._segments)
        progress = Progress(
            done_s=float(done), total_s=float(total), fraction=done / total, segments_done=done, segments_total=total
        )
        if done < total:
            updated = host.store.update_job(job.job_id, expect_status="running", progress=progress)
            if updated is None:  # cancelled meanwhile: give it back as its status says
                self._let_go(host, "the job changed while it ran")
            return True
        host.store.update_job(
            job.job_id,
            expect_status="running",
            status="completed",
            phase=None,
            outcome="all_passed",
            progress=progress,
            message=f"rendered {total} segment(s) with the fake worker",
        )
        self._clear_scratch(host, job.job_id)
        self._release(host)
        return True

    def shutdown(self, host: RunnerHost, reason: ShutdownReason) -> None:
        if self._job is None:
            return
        self._let_go(host, f"the daemon stopped ({reason})")

    # ------------------------------------------------------------------ one job
    def _take(self, host: RunnerHost, job: JobRecord) -> None:
        segments = job.request.get("segments")
        parsed: list[tuple[str, str]] = []
        if isinstance(segments, list):
            for item in segments:
                if isinstance(item, dict):
                    segment_id, text = item.get("segment_id"), item.get("text")
                    if isinstance(segment_id, str) and isinstance(text, str):
                        parsed.append((segment_id, text))
        self._job = job
        host.job_started(job)
        if not parsed or len(parsed) != len(segments or []):
            error = NarrationError(codes.INVALID_ARGUMENT, "the fake runner needs segments with segment_id and text")
            self._fail(host, error)
            return
        self._segments = parsed
        self._next = 0
        self._failures = 0
        self._clear_scratch(host, job.job_id)
        log.info("took job %s (%d segments)", job.job_id, len(parsed))

    def _render(self, host: RunnerHost, job: JobRecord, segment_id: str, text: str) -> None:
        voice_hash, _ = self._ready(host)
        seed = keys.seed(voice_hash=voice_hash, engine_text=text, attempt=0)
        render_key = keys.render_key(
            engine_profile_hash=ENGINE_PROFILE_HASH, voice_hash=voice_hash, engine_text=text, seed=seed
        )
        if host.store.get_render(render_key) is not None:
            return
        host.job_phase("rendering")
        out = host.store.scratch_path(SCRATCH_DIR, job.job_id, f"{segment_id}.wav")
        cap = self._cap(host, text)
        reply = host.workers.client(GROUP).request(
            "synthesize",
            {
                "voice_hash": voice_hash,
                "engine_text": text,
                "language": names.LANGUAGE,
                "seed": seed,
                "out_path": str(out),
                "max_new_tokens": cap,
            },
            timeout_s=REQUEST_TIMEOUT_S,
        )
        samples = int(reply["samples"])
        rate = int(reply["sample_rate"])
        gen_s = float(reply["gen_s"])
        record = RenderRecord(
            render_id=keys.render_id(render_key),
            render_key=render_key,
            voice=RenderVoice(voice_hash=voice_hash, clip_sha256=self._clip_sha256()),
            engine=RenderEngine(
                engine_profile_id=ENGINE_PROFILE_ID,
                engine_profile_hash=ENGINE_PROFILE_HASH,
                model_repo=MODEL_REPO,
                model_revision=MODEL_REVISION,
                non_streaming_mode=False,
                generation={"max_new_tokens": cap},
            ),
            engine_text=text,
            seed=seed,
            attempt=0,
            raw=RawAudio(path=str(out), sha256="", sample_rate=rate, samples=samples),
            hit_token_cap=bool(reply["hit_token_cap"]),
            gen_s=gen_s,
            rtf=round(gen_s / (samples / rate), 4) if samples else 0.0,
            canary=CanaryRecord(batch_status="not_run"),
            licence=Licence(),
        )
        host.store.put_render(record, out)
        log.info("job %s: rendered %s", job.job_id, segment_id)

    def _ready(self, host: RunnerHost) -> tuple[str, Path]:
        """Load the fake model and prepare the voice in the group's current worker (again after a restart)."""
        client = host.workers.client(GROUP)
        if self._prepared_in == client.pid and self._voice is not None and GROUP in host.workers.loaded():
            return self._voice
        host.job_phase("loading_model")
        host.workers.load(GROUP, {"device": "cpu"}, gpu=True, timeout_s=REQUEST_TIMEOUT_S)
        design_text = host.config.voice_design.design_text
        clip = host.store.scratch_path(SCRATCH_DIR, f"voice-{VOICE_SEED}.wav")
        if not clip.is_file():
            client.request(
                "design",
                {
                    "description": VOICE_DESCRIPTION,
                    "design_text": design_text,
                    "language": names.LANGUAGE,
                    "seed": VOICE_SEED,
                    "out_path": str(clip),
                    "max_new_tokens": self._cap(host, design_text),
                },
                timeout_s=REQUEST_TIMEOUT_S,
            )
        voice_hash = keys.voice_hash(
            model=names.MODEL_QWEN_BASE,
            clip_sha256=hashlib.sha256(clip.read_bytes()).hexdigest(),
            transcript=design_text,
            language=names.LANGUAGE,
            x_vector_only_mode=False,
        )
        client.request(
            "prepare_voice",
            {"voice_hash": voice_hash, "ref_wav": str(clip), "ref_text": design_text, "x_vector_only_mode": False},
            timeout_s=REQUEST_TIMEOUT_S,
        )
        self._voice = (voice_hash, clip)
        self._prepared_in = client.pid
        return self._voice

    def _clip_sha256(self) -> str:
        assert self._voice is not None
        return hashlib.sha256(self._voice[1].read_bytes()).hexdigest()

    @staticmethod
    def _cap(host: RunnerHost, text: str) -> int:
        engine = host.config.engines.qwen3_base
        return names.max_new_tokens_for(
            text,
            per_char=engine.max_new_tokens_per_char,
            floor=engine.max_new_tokens_floor,
            ceiling=names.MAX_NEW_TOKENS_CEILING,
        )

    # ------------------------------------------------------------------ endings
    def _failed_once(self, host: RunnerHost, why: str) -> None:
        self._failures += 1
        self._prepared_in = None
        log.warning("job %s: a render failed (%d of %d): %s", self._job_id(), self._failures, MAX_FAILURES, why)
        if self._failures >= MAX_FAILURES:
            self._fail(host, NarrationError(codes.INTERNAL, f"the fake worker failed {self._failures} times: {why}"))

    def _fail(self, host: RunnerHost, error: NarrationError) -> None:
        job = self._job
        assert job is not None
        host.store.update_job(job.job_id, expect_status="running", status="failed", phase=None, error=error.error)
        self._clear_scratch(host, job.job_id)
        self._release(host)

    def _let_go(self, host: RunnerHost, reason: str) -> None:
        job = self._job
        assert job is not None
        self._clear_scratch(host, job.job_id)
        return_job(host.store, job.job_id, reason=reason)
        self._release(host)

    def _release(self, host: RunnerHost) -> None:
        self._job = None
        self._segments = []
        self._next = 0
        host.job_finished()

    def _job_id(self) -> str:
        return self._job.job_id if self._job is not None else "-"

    @staticmethod
    def _clear_scratch(host: RunnerHost, job_id: str) -> None:
        """Remove the job's scratch folder: the files of a segment in flight are never published."""
        folder = host.store.scratch_path(SCRATCH_DIR, job_id, "placeholder").parent
        shutil.rmtree(folder, ignore_errors=True)


def job_request(*texts: str) -> dict[str, Any]:
    """A request this runner accepts: one segment per text, ids ``p01``, ``p02``, …"""
    return {"segments": [{"segment_id": f"p{i:02d}", "text": text} for i, text in enumerate(texts, start=1)]}


__all__ = ["FakeWorkerRunner", "job_request"]
