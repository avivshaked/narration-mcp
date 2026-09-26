"""Support for the job engine's tests: a daemon host, a pool of fake workers, a small aligner, and the records
a job needs (a pinned engine profile, the QA pins, a measured voice).

Nothing here needs a GPU or a model: the workers are ``narration_worker``'s ``fake`` role, which renders a
deterministic synthetic take for any text and hears it back. Every text is written for these tests (design
section 9.3); none comes from a caller's script.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import soundfile
from narration_worker.fake.faults import SPEC_ENV
from narration_worker.fake.handler import FAKE_ALIGNER_MODEL, FAKE_ASR_MODEL, FAKE_REVISION, FAKE_SV_MODEL

from narration import keys
from narration.config import Config, MeasurementConfig
from narration.contracts import codes, names
from narration.contracts.errors import NarrationError, WorkerCrashed
from narration.contracts.interfaces import AlignTranscript
from narration.contracts.models import (
    Alignment,
    Anchor,
    CrossCheck,
    CueTiming,
    Determinism,
    EngineProfile,
    EngineRef,
    Flag,
    Hint,
    JobRecord,
    MeasuredError,
    MeasurementRecord,
    Pace,
    PaceTrend,
    Progress,
    SegmentText,
    SimilarityBaseline,
    TranscriptCheck,
    WordTiming,
)
from narration.contracts.names import GpuHolder, JobKind, JobPhase, Priority
from narration.contracts.worker import AlignReply, AsrWord, HelloReply
from narration.jobs.gpu import VramReading
from narration.jobs.host import DAEMON_HOLDER, GpuFacts, ResidencyError, StopMode, WorkerPool
from narration.jobs.pins import ModelPin, QaPins
from narration.store import NarrationStore
from narration.store.store import utc_iso
from narration.text import words
from narration.workers import SubprocessWorkerClient, worker_command

# ======================================================================== the tests' own text

VOICE_TRANSCRIPT: Final = "Quiet mornings suit the old ferry, which crosses the bay without any hurry at all."
LAMPS: Final = "The lamplighter walks the canal path, counting bridges under her breath."
KETTLE: Final = "A copper kettle hums on the stove while rain draws lines across the window."
ORCHARD: Final = "Nobody in the valley remembers who planted the crooked orchard beside the mill."
LANTERN: Final = "At the far end of the pier, a green lantern swings above the sleeping boats."

ENGINE_ID: Final = "qwen3-base-1.7b.test"
ENGINE_REVISION: Final = "1" * 40
ENGINE_HASH: Final = "sha256:" + hashlib.sha256(b"narration job engine tests: engine profile").hexdigest()
QA_VRAM_MB: Final = 5000
QWEN_VRAM_MB: Final = 7000
METHOD_ID: Final = f"{names.ALIGNMENT_METHOD}/{FAKE_ALIGNER_MODEL}@{FAKE_REVISION}"
DESIGN_ID: Final = "01J0000000000000000000TEST"
"""The design that made the test clip, as the provenance list records it (section 17.4)."""
FRAME_S: Final = 0.02


# ======================================================================== the daemon's side


class MonotonicClock:
    """Monotonic seconds that move only when the host sleeps (or a test says so)."""

    def __init__(self) -> None:
        self.now = 1000.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += seconds


class CountingClient:
    """A worker client that counts the requests sent through it, by (group, op), and what they asked for."""

    def __init__(self, inner: SubprocessWorkerClient, group: GpuHolder, pool: FakePool) -> None:
        self.inner = inner
        self.group: GpuHolder = group
        self._pool = pool

    @property
    def role(self) -> Any:
        return self.inner.role

    @property
    def pid(self) -> int | None:
        return self.inner.pid

    @property
    def hello(self) -> HelloReply | None:
        return self.inner.hello

    def start(self) -> HelloReply:
        return self.inner.start()

    def request(self, op: str, payload: Mapping[str, Any], *, timeout_s: float) -> dict[str, Any]:
        with self._pool.lock:
            self._pool.calls[(self.group, op)] += 1
            self._pool.requests.append((self.group, op, dict(payload)))
        return self.inner.request(op, payload, timeout_s=timeout_s)

    def is_alive(self) -> bool:
        return self.inner.is_alive()

    def close(self, *, timeout_s: float = 10.0) -> None:
        self.inner.close(timeout_s=timeout_s)


class FakePool:
    """The daemon's ``WorkerPool`` over fake workers: one process per model group, started on demand and again
    after a crash, and one GPU group at a time (a second GPU load while one is loaded raises
    ``ResidencyError``, as the daemon's pool does, and is counted in ``refused``)."""

    def __init__(self, config: Config, spec_path: Path) -> None:
        base = {k: v for k, v in os.environ.items() if k != SPEC_ENV}
        base[SPEC_ENV] = str(spec_path)
        self._command = worker_command(config, "fake", base_env=base)
        self._clients: dict[GpuHolder, CountingClient] = {}
        self._loaded: set[GpuHolder] = set()
        self._gpu: GpuHolder | None = None
        self.lock = threading.Lock()
        self.calls: Counter[tuple[str, str]] = Counter()
        self.requests: list[tuple[str, str, dict[str, Any]]] = []
        self.loads: list[GpuHolder] = []
        self.unloads: list[GpuHolder] = []
        self.starts = 0
        self.refused = 0

    @property
    def gpu_holder(self) -> GpuHolder | None:
        return self._gpu

    def loaded(self) -> frozenset[GpuHolder]:
        return frozenset(self._loaded)

    def client(self, group: GpuHolder, *, cublas_workspace_config: str | None = None) -> CountingClient:
        current = self._clients.get(group)
        if current is None or not current.is_alive():
            if current is not None:
                current.close(timeout_s=5.0)
                self._forget(group)
            inner = SubprocessWorkerClient(self._command)
            inner.start()
            self.starts += 1
            current = CountingClient(inner, group, self)
            self._clients[group] = current
        return current

    def load(
        self,
        group: GpuHolder,
        payload: Mapping[str, Any],
        *,
        gpu: bool = True,
        timeout_s: float,
        cublas_workspace_config: str | None = None,
    ) -> dict[str, Any]:
        if gpu and self._gpu is not None and self._gpu != group:
            self.refused += 1
            raise ResidencyError(f"{group} loaded on the GPU while {self._gpu} is there")
        client = self.client(group, cublas_workspace_config=cublas_workspace_config)
        try:
            reply = client.request("load", payload, timeout_s=timeout_s)
        except WorkerCrashed:
            self._forget(group)
            raise
        self._loaded.add(group)
        if gpu:
            self._gpu = group
        self.loads.append(group)
        return reply

    def unload(self, group: GpuHolder, *, timeout_s: float) -> None:
        client = self._clients.get(group)
        if client is not None and client.is_alive():
            client.request("unload", {}, timeout_s=timeout_s)
        self.unloads.append(group)
        self._forget(group)

    def stop(self, group: GpuHolder) -> None:
        client = self._clients.pop(group, None)
        if client is not None:
            client.close(timeout_s=5.0)
        self._forget(group)

    def close(self) -> None:
        for group in list(self._clients):
            self.stop(group)

    def _forget(self, group: GpuHolder) -> None:
        self._loaded.discard(group)
        if self._gpu == group:
            self._gpu = None

    def texts(self, op: str = "synthesize") -> list[str]:
        """The engine texts of every request of ``op`` so far, in order."""
        return [str(p.get("engine_text")) for _, o, p in self.requests if o == op]


@dataclass
class Host:
    """The ``RunnerHost`` the daemon would pass: the store, the config and the workers, a stop switch, and a
    record of everything the runner told the daemon. ``sleep`` moves ``clock`` instead of waiting."""

    store: NarrationStore
    config: Config
    workers: WorkerPool
    clock: MonotonicClock
    holder: str = DAEMON_HOLDER
    stop_mode: StopMode | None = None
    started: list[str] = field(default_factory=list)
    phases: list[JobPhase | None] = field(default_factory=list)
    finished: int = 0
    gpu_facts: list[GpuFacts] = field(default_factory=list)
    drains: list[float | None] = field(default_factory=list)
    sleeps: list[float] = field(default_factory=list)
    real_sleep_s: float = 0.0
    """Real seconds each ``sleep`` also waits, at most (for tests where another thread does the work)."""

    def should_stop(self) -> bool:
        return self.stop_mode is not None

    def sleep(self, seconds: float) -> bool:
        self.sleeps.append(seconds)
        self.clock.advance(seconds)
        if self.real_sleep_s > 0:
            time.sleep(min(seconds, self.real_sleep_s))
        return self.stop_mode is None

    def job_started(self, job: JobRecord) -> None:
        self.started.append(job.job_id)

    def job_phase(self, phase: JobPhase | None) -> None:
        self.phases.append(phase)

    def job_finished(self) -> None:
        self.finished += 1

    def set_gpu_facts(self, facts: GpuFacts) -> None:
        self.gpu_facts.append(facts)

    def set_est_drain(self, seconds: float | None) -> None:
        self.drains.append(seconds)


class FixedProbe:
    """A VRAM probe that reports whatever a test sets."""

    def __init__(self, free_mb: int | None = 20_000, total_mb: int = 24_000) -> None:
        self.free_mb = free_mb
        self.total_mb = total_mb
        self.reads = 0

    def read(self) -> VramReading | None:
        self.reads += 1
        if self.free_mb is None:
            return None
        return VramReading(name="Test GPU", total_mb=self.total_mb, free_mb=self.free_mb)


def check_readable_path(path: str) -> Path:
    """Section 17.3's check as the tests need it (the platform's is Windows only): an absolute path to a
    regular file, or ``PATH_NOT_ALLOWED``."""
    candidate = Path(path)
    if not candidate.is_absolute() or not candidate.is_file():
        raise NarrationError(codes.PATH_NOT_ALLOWED, f"{path} is not an absolute path to a file", field="voice.path")
    return candidate.resolve()


# ======================================================================== a small aligner


class TestAligner:
    """An ``AlignerCore`` for these tests (WP15's aligner is built elsewhere): letters per word, ``|`` between
    words, each word timed by its letters' spans, each cue by its words. No snapping, no cross-check. What it
    must keep, it keeps: an unplaceable cue has null times and ``CUE_UNALIGNED``, never interpolated ones."""

    __test__ = False
    method_id = METHOD_ID
    model = FAKE_ALIGNER_MODEL
    revision = FAKE_REVISION
    device = "cpu"

    def __init__(self) -> None:
        self.errors: list[Mapping[str, Any] | None] = []
        """The ``error`` each ``resolve`` call was given, in order."""

    def build_transcript(self, segment: SegmentText, hints: Sequence[Hint]) -> AlignTranscript:
        tokens: list[str] = []
        owners: list[tuple[int, int] | None] = []
        placed: list[tuple[int, int, str]] = []
        for cue in segment.cues:
            for word in words(cue.spoken):
                letters = [ch.upper() for ch in word.text if ch.isalpha() or ch == "'"]
                if not any(ch.isalpha() for ch in letters):
                    continue
                if tokens:
                    tokens.append("|")
                    owners.append(None)
                tokens.extend(letters)
                owners.extend((cue.index, word.index) for _ in letters)
                placed.append((cue.index, word.index, word.text))
        return AlignTranscript(
            tokens=tuple(tokens), token_words=tuple(owners), words=tuple(placed), cue_count=len(segment.cues)
        )

    def guard(self, transcript: AlignTranscript, num_frames: int) -> bool:
        return num_frames >= len(transcript.tokens) + self._repeats(transcript)

    def guard_details(self, transcript: AlignTranscript, num_frames: int) -> dict[str, Any]:
        return {
            "reason": "guard",
            "frames": num_frames,
            "tokens": len(transcript.tokens),
            "repeats": self._repeats(transcript),
        }

    @staticmethod
    def _repeats(transcript: AlignTranscript) -> int:
        return sum(1 for a, b in zip(transcript.tokens, transcript.tokens[1:], strict=False) if a == b)

    def resolve(
        self,
        transcript: AlignTranscript,
        reply: AlignReply | None,
        audio: npt.NDArray[np.float32],
        sample_rate: int,
        asr_words: Sequence[AsrWord],
        measured_error: MeasuredError | None,
        *,
        error: Mapping[str, Any] | None = None,
    ) -> Alignment:
        self.errors.append(error)
        flags: list[Flag] = []
        times: dict[tuple[int, int], list[float]] = {}
        if not any(ch.isalpha() for t in transcript.tokens for ch in t):
            reason = codes.CUE_NO_ALIGNABLE_WORDS
        elif reply is None:
            reason = "alignment_error"
            flags.append(Flag(code=codes.ALIGNMENT_ERROR, severity="fail", message="the aligner could not align"))
        else:
            reason = None
            frame = float(reply["frame_s"])
            for span in reply["spans"]:
                owner = transcript.token_words[span["token_index"]]
                if owner is not None:
                    times.setdefault(owner, []).extend((span["start_frame"] * frame, span["end_frame"] * frame))
        cues: list[CueTiming] = []
        for index in range(transcript.cue_count):
            own = [(c, w, text) for c, w, text in transcript.words if c == index]
            spans = [times[(c, w)] for c, w, _ in own if (c, w) in times]
            if reason is None and spans:
                word_times = tuple(
                    WordTiming(text=t, start_s=min(times[(c, w)]), end_s=max(times[(c, w)]))
                    for c, w, t in own
                    if (c, w) in times
                )
                cues.append(
                    CueTiming(
                        index=index,
                        start_s=round(min(min(s) for s in spans), 3),
                        end_s=round(max(max(s) for s in spans), 3),
                        confidence=0.9,
                        words=word_times,
                    )
                )
            else:
                cues.append(CueTiming(index=index, start_s=None, end_s=None, confidence=None))
                flags.append(
                    Flag(
                        code=codes.CUE_UNALIGNED,
                        severity="warn",
                        message=f"cue {index} could not be placed",
                        cue=index,
                        details={"reason": reason or codes.CUE_NO_ALIGNABLE_WORDS},
                    )
                )
        return Alignment(
            method=names.ALIGNMENT_METHOD,
            model=self.model,
            revision=self.revision,
            device=self.device,
            cross_check=CrossCheck(model=FAKE_ASR_MODEL, max_disagreement_s=None),
            measured_error=measured_error,
            cues=tuple(cues),
            flags=tuple(flags),
        )


# ======================================================================== the world a job needs


def snapshot(models_root: Path, repo: str, revision: str) -> str:
    """A snapshot folder named by its revision, as the fake worker's ``load`` checks it exists."""
    folder = models_root / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder)


def qa_pins(models_root: Path) -> QaPins:
    """The QA group's pins, naming the fake worker's models."""

    def pin(repo: str) -> ModelPin:
        return ModelPin(repo=repo, revision=FAKE_REVISION, snapshot_dir=snapshot(models_root, repo, FAKE_REVISION))

    return QaPins(
        asr=pin(FAKE_ASR_MODEL), sv=pin(FAKE_SV_MODEL), aligner=pin(FAKE_ALIGNER_MODEL), vram_need_mb=QA_VRAM_MB
    )


GENERATION: Final[dict[str, Any]] = {
    "do_sample": True,
    "top_k": 50,
    "top_p": 1.0,
    "temperature": 0.9,
    "repetition_penalty": 1.05,
    "subtalker_dosample": True,
    "subtalker_top_k": 50,
    "subtalker_top_p": 1.0,
    "subtalker_temperature": 0.9,
    "max_new_tokens": names.MAX_NEW_TOKENS_CEILING,
}


def engine_profile(models_root: Path, **settings: Any) -> EngineProfile:
    """A pinned Base engine profile, as WP32 would record it, with every audio-changing setting."""
    return EngineProfile(
        engine_profile_id=ENGINE_ID,
        hash=ENGINE_HASH,
        model_repo=names.MODEL_QWEN_BASE,
        model_revision=ENGINE_REVISION,
        snapshot_dir=snapshot(models_root, names.MODEL_QWEN_BASE, ENGINE_REVISION),
        weights={"model.safetensors": "e" * 64},
        worker_project="workers/qwen3tts",
        uv_lock_sha256="f" * 64,
        packages={"qwen-tts": "0.1.1"},
        dtype="bfloat16",
        determinism=Determinism(
            attn_implementation="sdpa",
            tf32=False,
            cudnn_deterministic=True,
            cudnn_benchmark=False,
            deterministic_algorithms="warn_only",
            cublas_workspace_config=":4096:8",
        ),
        settings={
            "non_streaming_mode": False,
            "generation": dict(GENERATION),
            "max_new_tokens_per_char": 2.5,
            "max_new_tokens_floor": 128,
            **settings,
        },
        capabilities={"controls": {"pace": False, "context": False, "instruct": False}},
        licence="Apache-2.0",
        vram_need_mb=QWEN_VRAM_MB,
    )


def write_clip(path: Path) -> str:
    """A short synthetic clip (a tone), standing in for a designed voice; returns its sha256."""
    rate = 24_000
    t = np.arange(int(rate * 2.0), dtype=np.float64) / rate
    tone = (0.2 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), tone, rate, subtype="PCM_16")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def voice_hash(clip_sha256: str) -> str:
    return keys.voice_hash(
        model=names.MODEL_QWEN_BASE,
        clip_sha256=clip_sha256,
        transcript=VOICE_TRANSCRIPT,
        language=names.LANGUAGE,
        x_vector_only_mode=False,
    )


def measurement(
    clip_sha256: str,
    anchor: Sequence[float],
    *,
    max_segment_chars: int | None = 400,
    intercept_wpm: float = 130.0,
    corpus_hex: str = "0" * 64,
) -> MeasurementRecord:
    """A finished measurement of the test voice: the anchor the fake's embeddings of it sit near, a wide
    pace tolerance, and the similarity baselines of a steady voice. Another ``corpus_hex`` gives another
    measurement key, so the store replaces the measurement it has."""
    vh = voice_hash(clip_sha256)
    key = keys.measurement_key(
        voice_hash=vh,
        engine_profile_hash=ENGINE_HASH,
        corpus_version=f"{names.CORPUS}@sha256:{corpus_hex}",
        settings=MeasurementConfig(),
    )
    return MeasurementRecord(
        voice_hash=vh,
        clip_sha256=clip_sha256,
        engine_profile=EngineRef(id=ENGINE_ID, hash=ENGINE_HASH),
        measurement_key=key,
        transcript_check=TranscriptCheck(heard=VOICE_TRANSCRIPT, wer=0.0, ok=True),
        corpus=names.CORPUS,
        similarity=SimilarityBaseline(anchor_p5=0.95, anchor_p50=0.98, consistency_p5=0.95),
        pace=Pace(
            trend=PaceTrend(intercept_wpm=intercept_wpm, per_100_chars=0.0, band_max_chars=300), tol=0.6, curve=()
        ),
        max_segment_chars=max_segment_chars,
        max_segment_seconds=30.0 if max_segment_chars is not None else None,
        ladder=(),
        anchor=Anchor(model=FAKE_SV_MODEL, dim=len(anchor), embedding=tuple(anchor)),
        calibration=(),
        measured_at=utc_iso(time.time()),
    )


def request(
    clip: Path,
    clip_sha256: str,
    *texts: str,
    ids: Sequence[str] | None = None,
    takes: int | None = None,
    max_retakes: int | None = None,
    priority: Priority | None = None,
    hints: Sequence[Mapping[str, Any]] = (),
    **extra: Any,
) -> dict[str, Any]:
    """A ``submit_job`` request, as its input schema has it: one segment per text."""
    segment_ids = list(ids) if ids is not None else [f"p{i:02d}" for i in range(1, len(texts) + 1)]
    options: dict[str, Any] = {}
    if takes is not None:
        options["takes"] = takes
    if max_retakes is not None:
        options["max_retakes"] = max_retakes
    if priority is not None:
        options["priority"] = priority
    out: dict[str, Any] = {
        "voice": {"path": str(clip), "sha256": clip_sha256, "transcript": VOICE_TRANSCRIPT},
        "segments": [{"segment_id": s, "text": t} for s, t in zip(segment_ids, texts, strict=True)],
        **extra,
    }
    if hints:
        out["hints"] = [dict(h) for h in hints]
    if options:
        out["options"] = options
    return out


def submit(store: NarrationStore, body: Mapping[str, Any], *, kind: JobKind = "generate") -> JobRecord:
    """Queue a job as the front-end would (WP36): the request by value, its sha256, its priority."""
    options = body.get("options") or {}
    priority: Priority = options.get("priority", "batch")
    now = utc_iso(time.time())
    record = JobRecord(
        job_id=keys.Keys().new_job_id(),
        kind=kind,
        request=dict(body),
        request_sha256=hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest(),
        label=None,
        priority=priority,
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


def write_spec(path: Path, *faults: Mapping[str, Any]) -> None:
    """The fake worker's fault spec (``narration_worker.fake.faults``); the fake reads it again on change."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"faults": [dict(f) for f in faults]}), encoding="utf-8")
    os.replace(tmp, path)
    stamp = time.time_ns() + 1_000_000  # a new mtime even on a coarse clock, so the change is seen
    os.utime(path, ns=(stamp, stamp))


def drive(runner: Any, host: Host, *, max_steps: int = 400) -> int:
    """Step the runner until it has nothing to do; returns the steps taken."""
    for n in range(1, max_steps + 1):
        if not runner.step(host):
            return n
    raise AssertionError(f"the runner was still busy after {max_steps} steps")
