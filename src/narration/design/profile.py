"""The ``profile`` job kind (design sections 3.6, 7.6, 15 and 17.3; plan.md WP34): ``profile_voice``.

``ProfileHandler`` is a ``narration.jobs.handlers.JobHandler`` for ``profile`` jobs: section 3.6's measurements
(pitch, speaking rate, pauses, loudness, brightness, HNR, CPPS) and pictures (a spectrogram and a pitch contour)
of any WAV the owner can read, by path and sha256 (section 17.3: any absolute path on a local drive; nothing is
cloned, so the synthetic-voices rule does not apply). It shares the job engine's core, as every kind does.

**The job.** The request is ``{audio: {path, sha256}}``, checked at submit (``narration.backend``: the path, the
sha256, a WAV of at most 20 MB).

1. **The cache first** (section 15): a profile of the same bytes under the current profile version
   (``names.PROFILE_VERSION``) completes the job at once. A profile is a function of the audio's bytes alone:
   ``profile_voice`` is sent no transcript, so its speaking rate is null.
2. **The audio** is copied into the store before a worker sees it (``narration.jobs.voice.stage_clip``: through
   the daemon's path check, and only if the file still has the sha256 sent).
3. **The profile** runs on the CPU, in seconds (section 7.6): the QA worker's ``profile`` op. When the QA group is
   already loaded, its worker is used as it is; otherwise the worker is loaded with no model, on the CPU
   (App. A: ``models: {}``), which never waits for or holds the GPU.
4. **Publish**: ``profiles/<ab>/<sha256>/`` with ``profile.json`` and the two PNGs (``Store.put_profile``); the job
   completes with the profile's handle (``{audio_sha256, profile_version}``), which ``get_results`` reads.

A file the worker cannot read, or one with no sound at all to measure, is ``UNSUPPORTED_AUDIO`` on
``audio.path``. Other failures are handled as a design's are (``work.Pieces``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from narration.contracts import codes, names
from narration.contracts.errors import NarrationError, WorkerFailure
from narration.contracts.models import AudioRef, JobRecord, ProfileRecord, Progress
from narration.jobs.core import worker_code
from narration.jobs.engine import SCRATCH_JOBS, JobEngine
from narration.jobs.gpu import GroupNeed
from narration.jobs.host import RunnerHost
from narration.jobs.plan import VoiceSpec
from narration.jobs.stages import QA_TIMEOUT_S, Stages
from narration.jobs.state import Outcome
from narration.jobs.voice import stage_clip

from .work import RETRY_AFTER_S, Endings, Pieces, ProfileReplyError, StepRun, profile_record, set_phase

log = logging.getLogger(__name__)

KIND: Final = "profile"
"""The job kind this handler runs (``JobRecord.kind``)."""
PROFILE_NEED: Final = GroupNeed(
    group="qa",
    key="qa:profile-only",
    label="the QA worker with no model (a profile, on the CPU)",
    payload={"device": "cpu", "models": {}},
    need_mb=0,
    gpu=False,
)
"""What a profile needs when the QA group is not loaded: the QA worker loaded with no model, on the CPU."""
NOMINAL_S: Final = 1.0
"""A profile's size in the job's progress (``narration.jobs.admission``'s nominal figure for the kind)."""


@dataclass(slots=True, eq=False, kw_only=True)
class ProfileRun(StepRun):
    """A ``profile`` job the handler holds: the audio as the request sent it, its working copy in the store,
    and the profile once found in the cache (``current``) or made (``record``)."""

    audio: AudioRef
    wav: Path | None = None
    current: ProfileRecord | None = None
    record: ProfileRecord | None = None


class ProfileHandler:
    """Runs ``profile`` jobs (``JobHandler[ProfileRun]``; the module docstring). Build it with
    ``build_profile_handler``."""

    def __init__(self, engine: JobEngine) -> None:
        self.engine = engine
        self.core = engine.core
        self.stages = Stages(self.core)
        self.pieces = Pieces(self.core, self.stages)
        self.endings = Endings()

    def open(self, host: RunnerHost, job: JobRecord) -> ProfileRun:
        """Plan a claimed ``profile`` job: the profile from the cache, or the audio's working copy. Raises
        ``NarrationError`` for a file that cannot be read or has changed (naming ``audio.path`` or
        ``audio.sha256``)."""
        if job.kind != KIND:
            raise NarrationError(
                codes.INTERNAL,
                f"the profile handler was given a {job.kind} job",
                details={"kind": job.kind},
                hint="This is a bug in the service; nothing was read. Report it.",
            )
        store, parts = host.store, self.core.parts
        audio = _audio(job.request)
        run = ProfileRun(
            job=job,
            scratch=store.scratch_path(SCRATCH_JOBS, job.job_id, "work").parent,
            done_floor=job.progress.done_s,
            audio=audio,
        )
        found = store.get_profile(audio.sha256, names.PROFILE_VERSION)
        if found is not None:
            store.touch("profile", audio.sha256)
            run.current = found
            run.message = "already profiled: the profile of these bytes is in the cache"
            return run
        # Section 17.3: a caller's file is read only through the daemon's path check (or the one built in).
        check = parts.check_path if parts.check_path is not None else host.platform.check_readable_path
        try:
            run.wav = stage_clip(
                store, VoiceSpec(path=audio.path, sha256=audio.sha256, transcript=""), check_path=check
            )
        except NarrationError as exc:
            raise _as_audio(exc) from exc
        run.message = "planned: a profile on the CPU"
        return run

    def advance(self, host: RunnerHost, run: ProfileRun) -> Outcome:
        """Make the profile, or finish; ``finished`` once the job's record is written."""
        if run.current is None and run.record is None:
            return self.pieces.run(host, run, "qa", "profile the audio", lambda: self._profile(host, run))
        self.finish(host, run)
        return "finished"

    def _ready(self, host: RunnerHost, run: ProfileRun) -> bool:
        """The QA worker, loaded: the QA group as it is when resident, else with no model on the CPU."""
        residency = self.core.residency
        if residency.is_ready(host, self.stages.qa_need()):
            return True
        try:
            state = residency.ensure(host, PROFILE_NEED, phase=lambda p: set_phase(host, run, p))
        except WorkerFailure as exc:
            code = worker_code(exc)
            if code == codes.GPU_OOM:
                raise
            raise NarrationError(
                codes.BACKEND_NOT_INSTALLED if code == codes.BACKEND_NOT_INSTALLED else codes.INTERNAL,
                f"the QA worker could not be loaded for a profile: {exc.message}",
                details={"worker_code": exc.code, **exc.details},
                retryable=False,
                hint="Ask the operator to run narration-admin doctor; the QA worker could not be loaded.",
            ) from exc
        return state != "waiting"

    def _profile(self, host: RunnerHost, run: ProfileRun) -> Outcome:
        if not self._ready(host, run):
            return "stopped" if host.should_stop() else "waited"
        set_phase(host, run, "scoring")
        run.message = "profiling the audio on the CPU"
        assert run.wav is not None
        out_dir = run.scratch / "profile"
        started = self.core.parts.clock()
        try:
            reply = host.workers.client("qa").request(
                "profile", {"wav": str(run.wav), "out_dir": str(out_dir)}, timeout_s=QA_TIMEOUT_S
            )
        except WorkerFailure as exc:
            if worker_code(exc) == codes.UNSUPPORTED_AUDIO:
                raise NarrationError(
                    codes.UNSUPPORTED_AUDIO,
                    f"the QA worker could not read the audio: {exc.message}",
                    field="audio.path",
                    details={"worker_code": exc.code},
                    hint="Send a WAV file the service reads (PCM or float, mono or stereo).",
                ) from exc
            raise
        try:
            record = profile_record(run.audio.sha256, reply)
        except ProfileReplyError as exc:
            measurements = reply.get("measurements")
            silent = isinstance(measurements, dict) and measurements.get("spectral_centroid_hz") is None
            if silent:
                raise NarrationError(
                    codes.UNSUPPORTED_AUDIO,
                    "the audio has no sound to measure: every sample is zero",
                    field="audio.path",
                    hint="Send a clip or take with speech in it.",
                ) from exc
            raise NarrationError(
                codes.INTERNAL,
                f"the audio's profile cannot be read: {exc}",
                retryable=True,
                retry_after_s=RETRY_AFTER_S,
                hint="Send the same request again; if it fails again, the daemon's log has the details.",
            ) from exc
        run.record = host.store.put_profile(record, out_dir)
        self.core.throughput.record(self.core.parts.clock() - started, NOMINAL_S)
        return "worked"

    # ------------------------------------------------------------------ the record and the endings
    @staticmethod
    def result(run: ProfileRun) -> dict[str, Any] | None:
        """``JobRecord.result``: the profile's handle, from which ``get_results`` reads it."""
        profile = run.record if run.record is not None else run.current
        if profile is None:
            return None
        return {"audio_sha256": profile.audio_sha256, "profile_version": profile.profile_version}

    def progress(self, run: ProfileRun, *, complete: bool = False) -> Progress:
        """Progress: one nominal second of work, done once the profile is made or found."""
        made = run.record is not None or run.current is not None
        return self.endings.progress(run, NOMINAL_S if made else 0.0, NOMINAL_S, 1 if made else 0, 1, complete=complete)

    def finish(self, host: RunnerHost, run: ProfileRun) -> JobRecord | None:
        """Complete the job with the profile's handle."""
        result = self.result(run)
        assert result is not None
        profile = run.record if run.record is not None else run.current
        assert profile is not None
        m = profile.measurements
        run.message = (
            f"profiled {m.duration_s:.1f} s of audio"
            + (f": pitch median {m.pitch_median_hz:.0f} Hz" if m.pitch_median_hz is not None else "")
            + ("" if run.record is not None else " (from the cache)")
        )
        return self.endings.complete(
            host, run, outcome="all_passed", progress=self.progress(run, complete=True), result=result
        )

    def cancel(self, host: RunnerHost, run: ProfileRun) -> JobRecord | None:
        """Finish a cancelled profile job; a profile already made stays in the cache."""
        return self.endings.cancel(host, run, self.progress(run), self.result(run))

    def fail(self, host: RunnerHost, job: JobRecord, error: NarrationError, run: ProfileRun | None) -> JobRecord | None:
        """Record a job-level failure (``UNSUPPORTED_AUDIO``, ``VOICE_FILE_MISMATCH`` among them)."""
        return self.endings.fail(host, job, error, run, self.progress(run) if run is not None else None)

    def save(self, host: RunnerHost, run: ProfileRun) -> bool:
        """Write the job's phase, progress and message; False when it is no longer ``running``."""
        return self.endings.save(host, run, self.progress(run), self.result(run))

    def release(self, run: ProfileRun) -> None:
        """Remove the job's scratch files."""
        self.endings.release(run)

    def remaining_audio_s(self, run: ProfileRun) -> float:
        """Audio seconds of work left, for the queue's drain estimate."""
        progress = self.progress(run)
        return max(0.0, progress.total_s - progress.done_s)

    def close(self) -> None:
        """Stop the thread that renews the engine's leases (shared with the job engine; ``Registry.close``)."""
        self.core.leases.close()


def _audio(request: Mapping[str, Any]) -> AudioRef:
    """The request's audio (``profile_voice``'s input schema: ``{audio: {path, sha256}}``)."""
    audio = request.get("audio")
    fields = cast(dict[str, Any], audio) if isinstance(audio, dict) else {}
    path, sha = fields.get("path"), fields.get("sha256")
    if not isinstance(path, str) or not path or not isinstance(sha, str) or len(sha) != 64:
        raise NarrationError(
            codes.INTERNAL,
            "the profile job's request cannot be read (audio.path, audio.sha256)",
            retryable=False,
            hint="This is a bug in the service; nothing was read. Report it.",
        )
    return AudioRef(path=path, sha256=sha)


AUDIO_HINTS: Final[dict[str, str]] = {
    codes.VOICE_FILE_MISMATCH: "The file changed or moved since it was sent: send its path and the sha256 of the file "
    "that is there now.",
    codes.UNSUPPORTED_AUDIO: "Send a WAV file of at most 20 MB.",
    codes.PATH_NOT_ALLOWED: "Send an absolute path to a file on a local drive.",
}
"""The hint of a refusal of the audio that came without one (section 14: every error says what to do)."""


def _as_audio(exc: NarrationError) -> NarrationError:
    """A refusal of the working copy (``stage_clip``, which names a voice's fields) naming ``audio.*``, with a
    hint."""
    field = exc.field
    if field is not None and field.startswith("voice."):
        field = "audio." + field.removeprefix("voice.")
    elif field is None and exc.code == codes.PATH_NOT_ALLOWED:
        field = "audio.path"
    hint = exc.hint if exc.hint is not None else AUDIO_HINTS.get(exc.code)
    if field == exc.field and hint == exc.hint:
        return exc
    return NarrationError(
        exc.code,
        exc.message.replace("the clip", "the audio"),
        field=field,
        hint=hint,
        details=exc.details,
        retryable=exc.retryable,
        retry_after_s=exc.retry_after_s,
    )


def build_profile_handler(engine: JobEngine) -> ProfileHandler:
    """The ``profile`` kind's handler, for the runner's ``Registry`` (``narration.engine.installed``)."""
    return ProfileHandler(engine)


__all__ = ["AUDIO_HINTS", "KIND", "PROFILE_NEED", "ProfileHandler", "ProfileRun", "build_profile_handler"]
