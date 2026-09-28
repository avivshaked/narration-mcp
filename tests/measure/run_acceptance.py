"""WP33's acceptance run: a designed voice is not flagged across its own calibration takes (plan.md WP33).

Run it from the checkout, with the daemon, the QA models (WP22) and the engine pins (WP32) installed::

    uv run python -m tests.measure.run_acceptance --config <path> --voice d4

**What it does.**

1. Reads the voice's clip from the bakeoff (``NARRATION_BAKEOFF_ROOT``, read-only; design section 16 names
   the clips) and its exact transcript from the clip's sidecar. It goes on only if the clip's sha256 is in
   ``NARRATION_SPIKE_ALLOW_SHA256`` (synthetic voices only, section 17.4); the service checks its own
   allowlist again.
2. Takes the developers' GPU lock (AGENTS.md section 5) for ``--minutes`` (at most 30; a longer run needs the
   lead's OK first) and releases it on every exit path, after asking the daemon to unload its models
   (``release_gpu``). A lock this holder already has (a wrapper's) is used and left to the wrapper.
3. Queues a ``measure`` job in the store of the configuration (``--config``, else ``config.find_config``'s
   rule), starts the daemon if none runs (``narration.daemon.start``), and waits for the job, printing its
   progress in numbers.
4. **The check.** Each calibration take's similarity to the finished anchor must not be flagged by the
   speaker check a generation take gets (``SPK_SIM_LOW``: warn below ``anchor_p5 - sim_warn_margin``, fail
   below ``sim_fail_floor``). Then, unless ``--no-generate``, a ``generate`` job of the calibration paragraphs
   with the same attempts (``takes`` = seeds, no retakes) is scored through the job engine as any caller's
   would be: every take comes from the cache, and none may carry ``SPK_SIM_LOW``.
5. Writes the numbers (never a transcript or anything the recogniser heard) to ``--out``, by default
   ``<checkout>/.dev/acceptance/wp33-<voice>.json`` (gitignored), and exits 0 when accepted, 1 when not,
   2 when it could not run, 3 when the time ran out.

**Time** (ESTIMATE): the full measurement renders 12 calibration takes and up to 27 ladder takes, about
13 minutes of audio. If ``--minutes`` runs out first, the job is cancelled and the daemon asked to release
the GPU; every take made stays in the cache, so the next run resumes where this one stopped.

Nothing here prints or stores private text: the transcript goes only into the local store's job request, as a
caller's would.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from narration import keys
from narration.config import CONFIG_ENV, Config, ConfigError, MeasurementConfig, find_config, load_config
from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.models import JobRecord, MaterialSet, MeasurementRecord, Progress
from narration.contracts.names import JobKind
from narration.daemon.start import ensure_daemon as start_daemon
from narration.measure import load_corpus
from narration.platform import get_platform
from narration.qa.checks import speaker_flags
from narration.store import NarrationStore
from narration.store.store import utc_iso

CHECKOUT: Final = Path(__file__).resolve().parents[2]
LOCK_TOOL: Final = CHECKOUT / "tools" / "gpu_lock.py"
HOLDER: Final = "WP33"
NEED_MB: Final = 8000
"""The larger group's VRAM (Qwen; plan.md's placeholder until spike (h)'s numbers are pinned)."""
MAX_LOCK_MINUTES: Final = 30
BAKEOFF_ENV: Final = "NARRATION_BAKEOFF_ROOT"
ALLOW_ENV: Final = "NARRATION_SPIKE_ALLOW_SHA256"
VOICES: Final[Mapping[str, str]] = {
    "d2": "refs/auditions/qwen3-tts-voicedesign_d2-late-night_take1.wav",
    "d4": "refs/auditions/qwen3-tts-voicedesign_d4-radio-drama_take2.wav",
}
"""The bakeoff's two designed voices (design section 16), relative to the bakeoff's folder."""
TERMINAL: Final = ("completed", "failed", "cancelled")
POLL_S: Final = 5.0

ACCEPTED, REJECTED, UNAVAILABLE, OUT_OF_TIME = 0, 1, 2, 3


class Unavailable(Exception):
    """The acceptance cannot run here: a resource is missing. The message says which, and what to set."""


# ---------------------------------------------------------------------- the voice


@dataclass(frozen=True, slots=True)
class Voice:
    """An allowlisted bakeoff voice: its name (``d4``), its clip and sha256, and its exact transcript."""

    name: str
    clip: Path
    sha256: str
    transcript: str

    def request(self) -> dict[str, Any]:
        """``measure_voice``'s input (and ``submit_job``'s ``voice``)."""
        return {"voice": {"path": str(self.clip), "sha256": self.sha256, "transcript": self.transcript}}


def bakeoff_voice(name: str, env: Mapping[str, str] | None = None) -> Voice:
    """The voice ``name`` from the bakeoff; ``Unavailable`` when the bakeoff, the clip or the allowlist is not
    here, or the clip is not on the allowlist."""
    env = os.environ if env is None else env
    if name not in VOICES:
        raise Unavailable(f"unknown voice {name!r}: one of {', '.join(VOICES)}")
    root = env.get(BAKEOFF_ENV)
    if not root:
        raise Unavailable(f"set {BAKEOFF_ENV} to the bakeoff's folder (read-only)")
    allowed = frozenset(env.get(ALLOW_ENV, "").replace(",", " ").lower().split())
    if not allowed:
        raise Unavailable(f"set {ALLOW_ENV} to the sha256 of each clip that may be cloned (synthetic voices only)")
    clip = Path(root) / VOICES[name]
    try:
        digest = hashlib.sha256(clip.read_bytes()).hexdigest()
        sidecar = json.loads(clip.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Unavailable(f"{name}: the clip or its sidecar cannot be read ({type(exc).__name__})") from exc
    if digest not in allowed:
        raise Unavailable(f"{name}: the clip's sha256 is not in {ALLOW_ENV}; only an allowlisted clip is cloned")
    transcript = sidecar.get("text") if isinstance(sidecar, dict) else None
    if not isinstance(transcript, str) or not transcript.strip():
        raise Unavailable(f"{name}: the clip's sidecar has no transcript")
    return Voice(name=name, clip=clip, sha256=digest, transcript=transcript)


# ---------------------------------------------------------------------- the check


@dataclass(frozen=True, slots=True)
class TakeCheck:
    """One calibration take against the speaker check a generation take gets."""

    paragraph_id: str
    attempt: int
    sim_anchor: float | None
    flag: str | None
    """``"warn"`` or ``"fail"`` when ``SPK_SIM_LOW`` would be raised, else None."""


@dataclass(frozen=True, slots=True)
class CalibrationCheck:
    """Whether a measured voice is flagged across its own calibration takes."""

    warn_below: float
    fail_below: float
    takes: tuple[TakeCheck, ...]

    @property
    def flagged(self) -> tuple[TakeCheck, ...]:
        """The takes the speaker check would flag, or that have no similarity to check."""
        return tuple(t for t in self.takes if t.flag is not None or t.sim_anchor is None)

    @property
    def accepted(self) -> bool:
        """No calibration take is flagged, and there are takes to check."""
        return bool(self.takes) and not self.flagged


def calibration_check(measurement: MeasurementRecord, settings: MeasurementConfig) -> CalibrationCheck:
    """Each calibration take's similarity to the finished anchor, against QA's speaker thresholds for this
    measurement (``narration.qa.checks.speaker_check``: warn below ``anchor_p5 - sim_warn_margin``, rounded to 6
    places, and fail below ``sim_fail_floor``)."""
    warn_below = round(measurement.similarity.anchor_p5 - settings.sim_warn_margin, 6)
    fail_below = settings.sim_fail_floor
    takes: list[TakeCheck] = []
    for take in measurement.calibration:
        flags = speaker_flags(take.sim_anchor, warn_below, fail_below)
        takes.append(
            TakeCheck(
                paragraph_id=take.paragraph_id,
                attempt=take.attempt,
                sim_anchor=take.sim_anchor,
                flag=flags[0].severity if flags else None,
            )
        )
    return CalibrationCheck(warn_below=warn_below, fail_below=fail_below, takes=tuple(takes))


def generation_flags(job: JobRecord) -> dict[str, Any]:
    """What a ``generate`` job of the calibration paragraphs found: ``SPK_SIM_LOW`` flags by severity, the takes
    it scored and how many came from the cache, and each take's verdict count."""
    attempts = [a for item in job.items for a in item.attempts]
    speaker = Counter(f.severity for a in attempts for f in a.flags if f.code == codes.SPK_SIM_LOW)
    return {
        "status": job.status,
        "outcome": job.outcome,
        "takes": len(attempts),
        "from_cache": sum(1 for a in attempts if not a.fresh),
        "verdicts": dict(Counter(a.verdict or "none" for a in attempts)),
        "spk_sim_low": dict(speaker),
    }


def report(voice: Voice, measurement: MeasurementRecord, check: CalibrationCheck) -> dict[str, Any]:
    """The run's numbers, keyed by the service's own ids: no transcript, nothing the recogniser heard."""
    return {
        "voice": voice.name,
        "clip_sha256": voice.sha256,
        "measurement_key": measurement.measurement_key,
        "engine_profile": measurement.engine_profile.id,
        "corpus": measurement.corpus,
        "transcript_check": {"ok": measurement.transcript_check.ok, "wer": measurement.transcript_check.wer},
        "similarity": {
            "anchor_p5": measurement.similarity.anchor_p5,
            "anchor_p50": measurement.similarity.anchor_p50,
            "consistency_p5": measurement.similarity.consistency_p5,
        },
        "thresholds": {"warn_below": check.warn_below, "fail_below": check.fail_below},
        "calibration": [
            {"paragraph_id": t.paragraph_id, "attempt": t.attempt, "sim_anchor": t.sim_anchor, "flag": t.flag}
            for t in check.takes
        ],
        "pace": {
            "method": measurement.pace.method,
            "level_cps": measurement.pace.level_cps,
            "intercept_cps": measurement.pace.trend.intercept_cps,
            "per_100_chars": measurement.pace.trend.per_100_chars,
            "tol": measurement.pace.tol,
            "speaking_share": measurement.pace.speaking_share,
        },
        "ladder": [{"paragraph_id": r.paragraph_id, "chars": r.chars, "passes": r.passes} for r in measurement.ladder],
        "max_segment_chars": measurement.max_segment_chars,
        "max_segment_seconds": measurement.max_segment_seconds,
        "accepted": check.accepted,
    }


# ---------------------------------------------------------------------- the store and the daemon


def queue_job(store: NarrationStore, kind: JobKind, body: Mapping[str, Any]) -> JobRecord:
    """Queue a job as the front-end would: the request by value, at ``batch`` priority."""
    now = utc_iso(time.time())
    record = JobRecord(
        job_id=keys.Keys().new_job_id(),
        kind=kind,
        request=dict(body),
        request_sha256=hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest(),
        label=f"WP33 acceptance ({kind})",
        priority="batch",
        status="queued",
        phase=None,
        round=0,
        progress=Progress(done_s=0.0, total_s=0.0, fraction=0.0, segments_done=0, segments_total=0),
        outcome=None,
        error=None,
        idempotency_key=None,
        created_at=now,
        updated_at=now,
    )
    job, _ = store.create_job(record)
    return job


def ensure_daemon(store: NarrationStore, config_path: Path, job_id: str) -> None:
    """Start the daemon for the store unless one runs (``narration.daemon.start``), after the job is queued. A
    daemon that cannot be started cancels the job (``Unavailable``)."""
    try:
        start_daemon(store, config_path, wait_s=30.0)
    except NarrationError as exc:
        give_up(store, job_id)
        raise Unavailable(f"the daemon cannot be started: {exc.code}: {exc.message}") from exc


def wait_for(store: NarrationStore, job_id: str, deadline: float) -> JobRecord | None:
    """Wait for the job to end, printing its progress when it changes; None when ``deadline`` passes first."""
    shown: tuple[object, ...] = ()
    while True:
        job = store.get_job(job_id)
        if job is None:
            raise RuntimeError(f"job {job_id} is gone from the store")
        now = (job.status, job.phase, job.progress.segments_done, job.progress.segments_total, job.round)
        if now != shown:
            shown = now
            print(
                f"  {job.kind} {job.status} {job.phase or '-'}: {job.progress.segments_done}/"
                f"{job.progress.segments_total} segments, {job.progress.fraction:.0%}",
                flush=True,
            )
        if job.status in TERMINAL:
            return job
        if time.monotonic() >= deadline:
            return None
        time.sleep(POLL_S)


def give_up(store: NarrationStore, job_id: str) -> None:
    """Cancel the job (what it made stays in the cache) and ask the daemon to let go of the GPU."""
    job = store.get_job(job_id)
    if job is not None and job.status == "queued":
        store.update_job(job_id, expect_status="queued", status="cancelled")
    elif job is not None and job.status == "running":
        store.update_job(job_id, expect_status="running", status="cancelling")
    store.post_command("release_gpu")


def error_line(job: JobRecord) -> str:
    """The job's error as code and numbers: never ``details.heard`` (the recogniser's words of the clip)."""
    error = job.error
    if error is None:
        return f"{job.status} with no error"
    details = error.details or {}
    numbers = {k: v for k, v in details.items() if k in ("wer", "word_errors", "words", "segment_id", "attempt")}
    return f"{error.code} (retryable: {error.retryable}) {json.dumps(numbers)}"


# ---------------------------------------------------------------------- the GPU lock


def _lock(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(LOCK_TOOL), *args], cwd=CHECKOUT, capture_output=True, text=True, check=False
    )


def take_lock(minutes: int) -> bool:
    """Take the GPU lock; True when this run took it (and must release it), False when this holder already
    had it. Raises ``Unavailable`` when it cannot be taken."""
    if minutes > MAX_LOCK_MINUTES:
        raise Unavailable(f"a run longer than {MAX_LOCK_MINUTES} min needs the lead's OK first (AGENTS.md section 5)")
    status = _lock("status")
    if status.returncode == 0 and status.stdout.strip() != "free":
        owner = json.loads(status.stdout).get("owner") or {}
        if owner.get("holder") == HOLDER:
            return False
        raise Unavailable(f"the GPU lock is held by someone else: {status.stdout.strip()}")
    taken = _lock("acquire", "--holder", HOLDER, "--minutes", str(minutes), "--need-mb", str(NEED_MB))
    if taken.returncode != 0:
        raise Unavailable(f"the GPU lock was not taken: {(taken.stderr or taken.stdout).strip()}")
    return True


# ---------------------------------------------------------------------- the run


def calibration_request(
    voice: Voice, measurement: MeasurementRecord, config: Config, corpus: MaterialSet | None = None
) -> dict[str, Any]:
    """A ``generate`` request of the calibration paragraphs with the measurement's attempts (``takes`` = seeds,
    no retakes), so every take is the calibration take the cache holds. ``corpus``: the calibration corpus
    (default: the service's own, ``[measurement] corpus``)."""
    corpus = corpus if corpus is not None else load_corpus(config.measurement.corpus)
    measured = {t.paragraph_id for t in measurement.calibration}
    segments = [
        {
            "segment_id": p.segment_id,
            "cues": [
                {"text": c.text, **({"exact": [{"start": s.start, "end": s.end} for s in c.exact]} if c.exact else {})}
                for c in p.cues
            ],
        }
        for p in corpus.paragraphs
        if p.segment_id in measured
    ]
    return {
        **voice.request(),
        "segments": segments,
        "options": {"takes": config.measurement.seeds, "max_retakes": 0},
    }


def run(args: argparse.Namespace) -> int:
    """The acceptance run (the module docstring). Returns the exit code."""
    try:
        config_path = find_config(args.config)
    except ConfigError as exc:
        raise Unavailable(str(exc)) from exc
    voice = bakeoff_voice(args.voice)
    config = load_config(config_path)
    out = Path(args.out) if args.out else CHECKOUT / ".dev" / "acceptance" / f"wp33-{voice.name}.json"
    took = take_lock(args.minutes)
    store = None
    try:
        store = NarrationStore.from_config(config, get_platform())
        deadline = time.monotonic() + args.minutes * 60
        print(f"measuring {voice.name} (clip sha256 {voice.sha256[:12]}...)", flush=True)
        job = queue_job(store, "measure", voice.request())
        ensure_daemon(store, config_path, job.job_id)
        done = wait_for(store, job.job_id, deadline)
        if done is None:
            give_up(store, job.job_id)
            print(f"out of time after {args.minutes} min; the next run resumes from the cache", flush=True)
            return OUT_OF_TIME
        if done.status != "completed" or done.result is None:
            print(f"the measurement did not complete: {error_line(done)}", flush=True)
            return REJECTED
        measurement = store.get_measurement(str(done.result["voice_hash"]), str(done.result["engine_profile"]["id"]))
        if measurement is None:
            raise RuntimeError("the job completed but its measurement is not in the store")
        check = calibration_check(measurement, config.measurement)
        result = report(voice, measurement, check)
        if not args.no_generate:
            generation = queue_job(store, "generate", calibration_request(voice, measurement, config))
            ensure_daemon(store, config_path, generation.job_id)
            scored = wait_for(store, generation.job_id, deadline)
            if scored is None:
                give_up(store, generation.job_id)
                result["generation"] = {"status": "out_of_time"}
                result["accepted"] = False
            else:
                result["generation"] = generation_flags(scored)
                clean = scored.status == "completed" and not result["generation"]["spk_sim_low"]
                result["accepted"] = bool(result["accepted"] and clean)
        result["measured_at"] = measurement.measured_at
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".partial")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, out)
        flagged = [f"{t.paragraph_id}#{t.attempt}" for t in check.flagged]
        print(
            f"anchor_p5 {measurement.similarity.anchor_p5:.4f}, warn below {check.warn_below:.4f}; "
            f"lowest calibration take {min((t.sim_anchor or 0.0) for t in check.takes):.4f}; "
            f"flagged: {flagged or 'none'}; generation: {result.get('generation', 'skipped')}",
            flush=True,
        )
        print(f"{'ACCEPTED' if result['accepted'] else 'NOT ACCEPTED'}; numbers in {out}", flush=True)
        return ACCEPTED if result["accepted"] else REJECTED
    finally:
        if store is not None:
            try:  # the daemon unloads its models now, not after its idle timeout, before the lock is released
                store.post_command("release_gpu")
            except Exception as exc:  # best effort on the way out: the lock is released regardless
                print(f"could not ask the daemon to release the GPU: {type(exc).__name__}", file=sys.stderr)
            store.close()
        if took:
            _lock("release", "--holder", HOLDER)


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--config", help=f"the service's configuration file (default: ${CONFIG_ENV}, then the service's)"
    )
    parser.add_argument("--voice", default="d4", choices=sorted(VOICES))
    parser.add_argument("--minutes", type=int, default=MAX_LOCK_MINUTES, help="GPU lock and wait, at most 30")
    parser.add_argument("--out", help="where the numbers go (default: <checkout>/.dev/acceptance/wp33-<voice>.json)")
    parser.add_argument("--no-generate", action="store_true", help="skip the generate job's cross-check")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """The command line (the module docstring)."""
    try:
        return run(parse(argv))
    except Unavailable as exc:
        print(f"cannot run: {exc}", file=sys.stderr)
        return UNAVAILABLE


if __name__ == "__main__":
    sys.exit(main())
