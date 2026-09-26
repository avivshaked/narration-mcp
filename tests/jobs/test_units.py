"""The job engine's pure parts: DC-2's numbers, the GPU residency and its wait, reading a request, the pinned
models, and the clip's working copy (design sections 4, 7.3, 7.4, 8, 10.1, 10.3, 17; plan.md DC-2, DC-4)."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from narration.config import Config, DefaultsConfig, GpuConfig
from narration.contracts import codes, names
from narration.contracts.errors import NarrationError
from narration.contracts.models import DaemonStatus, GpuStatus, Hint, JobRecord, Progress, SegmentIn
from narration.contracts.names import GpuHolder, JobPhase
from narration.jobs import admission
from narration.jobs.gpu import GroupNeed, NoProbe, NvmlProbe, Residency
from narration.jobs.pins import ModelPin, ProfileError, QaPins, call_cap, ceiling, qwen_load_payload
from narration.jobs.plan import (
    GenerateRequest,
    RequestError,
    estimated_audio_s,
    hints_used,
    next_attempt,
    requested_attempts,
)
from narration.jobs.voice import clip_path, stage_clip
from narration.store import NarrationStore
from narration.text import TextPipeline
from tests.store.standin import StandInPlatform

from .support import (
    GENERATION,
    KETTLE,
    LAMPS,
    FixedProbe,
    Host,
    MonotonicClock,
    engine_profile,
    measurement,
    write_clip,
)

# ======================================================================== DC-2: poll_after_s and retry_after_s


@pytest.mark.parametrize(
    ("status", "phase", "eta", "wait", "expected"),
    [
        ("completed", None, None, None, 0.0),
        ("failed", None, 100.0, None, 0.0),
        ("running", "waiting_for_gpu", 500.0, None, admission.GPU_RECHECK_S),
        ("queued", None, None, None, admission.POLL_DEFAULT_QUEUED_S),
        ("queued", None, None, 4.0, admission.POLL_QUEUED_MIN_S),
        ("queued", None, None, 40.0, 10.0),
        ("queued", None, None, 4000.0, admission.POLL_QUEUED_MAX_S),
        ("running", "rendering", None, None, admission.POLL_DEFAULT_RUNNING_S),
        ("running", "rendering", 3.0, None, admission.POLL_MIN_S),
        ("running", "scoring", 50.0, None, 5.0),
        ("cancelling", None, 5000.0, None, admission.POLL_RUNNING_MAX_S),
    ],
)
def test_poll_after_s_follows_the_jobs_state_dc2(
    status: Any, phase: JobPhase | None, eta: float | None, wait: float | None, expected: float
) -> None:
    assert admission.poll_after_s(status, phase, eta=eta, wait_to_start_s=wait) == expected


def test_queue_full_retry_after_shares_the_drain_over_the_queue_dc2() -> None:
    assert admission.queue_full_retry_after_s(None, 5) == admission.QUEUE_FULL_DEFAULT_S
    assert admission.queue_full_retry_after_s(600.0, 0) == admission.QUEUE_FULL_DEFAULT_S
    assert admission.queue_full_retry_after_s(600.0, 3) == 200.0
    assert admission.queue_full_retry_after_s(1e9, 1) == admission.QUEUE_FULL_MAX_S
    assert admission.queue_full_retry_after_s(0.1, 10) == admission.RETRY_MIN_S


def test_the_rate_window_says_when_the_oldest_submission_leaves_it_dc2() -> None:
    now = 10_000.0
    created = [now - 50.0, now - 40.0, now - 10.0]

    def count_since(t: float) -> int:
        return sum(1 for c in created if c >= t)

    assert admission.rate_window(count_since, now=now, max_per_window=5) == admission.RateWindow(2, 0.0)
    full = admission.rate_window(count_since, now=now, max_per_window=3)
    assert full.remaining == 0 and full.resets_in_s == pytest.approx(10.0, abs=0.11)
    closed = admission.rate_window(count_since, now=now, max_per_window=0)
    assert closed == admission.RateWindow(0, admission.RATE_WINDOW_S)


def _job(
    job_id: str, *, priority: Any = "batch", status: Any = "queued", body: Mapping[str, Any] | None = None
) -> JobRecord:
    return JobRecord(
        job_id=job_id,
        kind="generate",
        request=dict(body or {"segments": [{"segment_id": "a", "text": "x" * 150}]}),
        request_sha256="0" * 64,
        label=None,
        priority=priority,
        status=status,
        phase=None,
        round=0,
        progress=Progress(done_s=0.0, total_s=0.0, fraction=0.0, segments_done=0, segments_total=1),
        outcome=None,
        error=None,
        idempotency_key=None,
        created_at="2026-01-01T00:00:00.000Z",
        updated_at="2026-01-01T00:00:00.000Z",
    )


def test_the_queue_order_is_priority_then_first_come_s4() -> None:
    queued = [_job("b1"), _job("i1", priority="interactive"), _job("b2"), _job("i2", priority="interactive")]
    assert [j.job_id for j in admission.scheduling_order(queued)] == ["i1", "i2", "b1", "b2"]
    assert admission.queue_position("b2", queued) == 3
    assert admission.queue_position("i1", queued) == 0
    running = [_job("r", status="running"), *queued]
    assert admission.queue_position("r", running) is None


def test_the_drain_estimate_counts_every_active_job_dc2() -> None:
    body = {"segments": [{"segment_id": "a", "text": "x" * 150}], "options": {"takes": 2}}
    assert admission.request_audio_s("generate", body) == pytest.approx(20.0)  # 150 chars x 2 takes / 15
    queued = [_job("a", body=body), _job("b", status="running"), _job("c", status="completed")]
    rate = 2.0
    drain = admission.est_drain_s(queued, wall_per_audio_s=rate, running_remaining_s={"b": 5.0})
    expected = admission.job_wall_s(20.0, wall_per_audio_s=rate) + admission.job_wall_s(
        5.0, wall_per_audio_s=rate, loads=1
    )
    assert drain == pytest.approx(expected, abs=0.05)


def test_throughput_is_measured_on_this_machine_dc2() -> None:
    throughput = admission.Throughput(initial=3.0, weight=0.5)
    assert throughput.wall_per_audio_s == 3.0
    throughput.record(10.0, 5.0)
    assert throughput.wall_per_audio_s == 2.0  # the first measurement replaces the assumption
    throughput.record(4.0, 1.0)
    assert throughput.wall_per_audio_s == 3.0
    throughput.record(float("nan"), 1.0)
    throughput.record(1.0, 0.0)
    assert throughput.samples == 2


def test_admission_reports_room_rate_and_the_gpu_dc2() -> None:
    rate = admission.RateWindow(remaining=3, resets_in_s=0.0)
    status = DaemonStatus(
        state="busy",
        pid=1,
        started_at=None,
        workers=(),
        current_job=None,
        gpu=GpuStatus(
            name="Test GPU",
            total_mb=24000,
            free_mb=9000,
            in_use=True,
            holder="qwen",
            unload_in_s=None,
            need_mb={"qwen": 7000, "qa": 5000},
            waiting_since=None,
        ),
        est_drain_s=120.0,
        updated_at="2026-01-01T00:00:00.000Z",
    )
    facts = admission.admission(queue_length=2, max_queued=20, est_drain=120.0, rate=rate, daemon=status)
    assert facts == {
        "accepting": True,
        "queue": {"length": 2, "max": 20, "est_drain_s": 120.0},
        "rate": {"remaining": 3, "resets_in_s": 0.0},
        "gpu": {
            "in_use": True,
            "holder": "qwen",
            "free_mb": 9000,
            "need_mb": {"qwen": 7000, "qa": 5000},
            "waiting_since": None,
        },
    }
    full = admission.admission(queue_length=20, max_queued=20, est_drain=None, rate=rate, daemon=None)
    assert full["accepting"] is False and full["gpu"]["holder"] is None


# ======================================================================== section 4: one resident group, the wait


class _Client:
    def __init__(self, pid: int) -> None:
        self.pid = pid


@dataclasses.dataclass
class _Pool:
    """A ``WorkerPool`` with no processes: it records loads and unloads."""

    gpu_holder: GpuHolder | None = None
    groups: set[GpuHolder] = dataclasses.field(default_factory=set)
    pids: dict[GpuHolder, int] = dataclasses.field(default_factory=lambda: {"qwen": 101, "qa": 202})
    log: list[str] = dataclasses.field(default_factory=list)

    def loaded(self) -> frozenset[GpuHolder]:
        return frozenset(self.groups)

    def client(self, group: GpuHolder, *, cublas_workspace_config: str | None = None) -> Any:
        return _Client(self.pids[group])

    def load(
        self,
        group: GpuHolder,
        payload: Mapping[str, Any],
        *,
        gpu: bool = True,
        timeout_s: float,
        cublas_workspace_config: str | None = None,
    ) -> dict[str, Any]:
        assert not (gpu and self.gpu_holder not in (None, group))
        self.log.append(f"load {group}")
        self.groups.add(group)
        if gpu:
            self.gpu_holder = group
        return {}

    def unload(self, group: GpuHolder, *, timeout_s: float) -> None:
        self.log.append(f"unload {group}")
        self.groups.discard(group)
        if self.gpu_holder == group:
            self.gpu_holder = None

    def stop(self, group: GpuHolder) -> None:
        self.unload(group, timeout_s=1.0)


@pytest.fixture
def gpu_host() -> Iterator[Host]:
    yield Host(store=cast(NarrationStore, None), config=cast(Config, None), workers=_Pool(), clock=MonotonicClock())


def _need(group: GpuHolder, key: str = "k1", mb: int = 7000) -> GroupNeed:
    return GroupNeed(group=group, key=key, label=f"the {group} models", payload={"device": "cuda:0"}, need_mb=mb)


def _residency(host: Host, probe: Any, **gpu: Any) -> Residency:
    return Residency(gpu=GpuConfig(**gpu), probe=probe, clock=host.clock, wall=lambda: "2026-01-01T00:00:00.000Z")


def test_a_loaded_group_is_used_as_it_is_across_jobs_s4(gpu_host: Host) -> None:
    phases: list[JobPhase] = []
    residency = _residency(gpu_host, FixedProbe(free_mb=20_000))
    assert residency.ensure(gpu_host, _need("qwen"), phase=phases.append) == "loaded"
    assert residency.ensure(gpu_host, _need("qwen"), phase=phases.append) == "ready"
    assert cast(_Pool, gpu_host.workers).log == ["load qwen"]
    assert phases == ["loading_model"]


def test_the_other_group_is_unloaded_before_a_load_s4(gpu_host: Host) -> None:
    residency = _residency(gpu_host, NoProbe())
    residency.ensure(gpu_host, _need("qwen"), phase=lambda p: None)
    residency.ensure(gpu_host, _need("qa", key="qa"), phase=lambda p: None)
    residency.ensure(gpu_host, _need("qwen", key="k2"), phase=lambda p: None)  # another engine: load again
    assert cast(_Pool, gpu_host.workers).log == ["load qwen", "unload qwen", "load qa", "unload qa", "load qwen"]


def test_a_restarted_worker_is_loaded_again_s4(gpu_host: Host) -> None:
    residency = _residency(gpu_host, NoProbe())
    residency.ensure(gpu_host, _need("qwen"), phase=lambda p: None)
    pool = cast(_Pool, gpu_host.workers)
    pool.pids["qwen"] = 999  # the daemon started the worker again
    assert residency.ensure(gpu_host, _need("qwen"), phase=lambda p: None) == "loaded"


def test_too_little_free_vram_waits_in_steps_then_is_gpu_unavailable_s4(gpu_host: Host) -> None:
    phases: list[JobPhase] = []
    probe = FixedProbe(free_mb=7500)  # the need (7000) plus the margin (1024) is not free
    residency = _residency(gpu_host, probe, wait_timeout_min=1)
    for _ in range(4):
        assert residency.ensure(gpu_host, _need("qwen"), phase=phases.append) == "waiting"
    assert gpu_host.sleeps == [admission.GPU_RECHECK_S] * 4
    assert set(phases) == {"waiting_for_gpu"}
    assert gpu_host.gpu_facts[-1].waiting_since == "2026-01-01T00:00:00.000Z"
    assert gpu_host.gpu_facts[-1].free_mb == 7500
    with pytest.raises(NarrationError) as caught:
        residency.ensure(gpu_host, _need("qwen"), phase=phases.append)
    error = caught.value
    assert error.code == codes.GPU_UNAVAILABLE and error.retryable
    assert error.retry_after_s == admission.GPU_UNAVAILABLE_RETRY_S
    assert error.details == {"free_mb": 7500, "need_mb": 7000, "waited_s": 60.0}
    assert gpu_host.gpu_facts[-1].waiting_since is None
    assert cast(_Pool, gpu_host.workers).log == []


def test_enough_free_vram_and_the_margin_loads_at_once_s4(gpu_host: Host) -> None:
    residency = _residency(gpu_host, FixedProbe(free_mb=7000 + 1024))
    assert residency.ensure(gpu_host, _need("qwen"), phase=lambda p: None) == "loaded"
    assert gpu_host.sleeps == []


def test_nvml_is_asked_only_for_a_cuda_device_s4() -> None:
    assert NvmlProbe("cpu").read() is None
    assert NvmlProbe("vulkan:0").read() is None


# ======================================================================== reading a request (sections 7.3, 8, 10.3)


def _body(**extra: Any) -> dict[str, Any]:
    return {
        "voice": {"path": "clip.wav", "sha256": "a" * 64, "transcript": "Some words."},
        "segments": [{"segment_id": "p01", "text": LAMPS}],
        **extra,
    }


def test_a_request_is_read_with_the_services_defaults_s7_3() -> None:
    request = GenerateRequest.parse(_body(), DefaultsConfig(takes=2, max_retakes=1))
    assert (request.takes, request.max_retakes, request.priority, request.strict_text) == (2, 1, "batch", False)
    options = GenerateRequest.parse(
        _body(options={"takes": 3, "max_retakes": 0, "priority": "interactive"}), DefaultsConfig()
    )
    assert (options.takes, options.max_retakes, options.priority) == (3, 0, "interactive")


@pytest.mark.parametrize(
    "body",
    [
        {"segments": [{"segment_id": "p01", "text": "x"}]},
        _body(segments=[]),
        _body(options={"takes": "two"}),
        _body(options={"priority": "urgent"}),
        _body(expect_engine_profile=5),
    ],
)
def test_a_malformed_stored_request_is_refused_s7_3(body: dict[str, Any]) -> None:
    with pytest.raises(RequestError):
        GenerateRequest.parse(body, DefaultsConfig())


def test_attempts_are_the_ones_named_or_zero_to_takes_s10_3() -> None:
    assert requested_attempts(SegmentIn(segment_id="a", text="x"), 3) == (0, 1, 2)
    assert requested_attempts(SegmentIn(segment_id="a", text="x", attempts=(4, 7)), 3) == (4, 7)
    assert next_attempt([]) == 0
    assert next_attempt([4, 7, 5]) == 8  # above every attempt of the segment so far


def test_a_segment_uses_only_the_hints_applied_in_it_s9_1() -> None:
    pipeline = TextPipeline()
    hints = (
        Hint(term="kettle", respell="KET-ul", asr_aliases=("kettel",), note="informational"),
        Hint(term="lamplighter", asr_aliases=("lamp lighter",)),
        Hint(term="stove"),
    )
    (text,) = pipeline.plan_request([SegmentIn(segment_id="p01", text=KETTLE)], hints, strict_text=False)
    used = hints_used(text, hints)
    assert used == (
        Hint(term="kettle", respell="KET-ul", asr_aliases=("kettel",)),
        Hint(term="stove"),
    )


def test_the_audio_estimate_follows_the_voices_pace_s7_4(tmp_path: Path) -> None:
    (text,) = TextPipeline().plan_request([SegmentIn(segment_id="p01", text=LAMPS)], (), strict_text=False)
    record = measurement("b" * 64, (1.0, 0.0), intercept_wpm=120.0)
    assert estimated_audio_s(text, record) == pytest.approx(len(LAMPS.split()) / 120.0 * 60.0)
    assert estimated_audio_s(text, None) == pytest.approx(len(LAMPS) / admission.CHARS_PER_AUDIO_S)


# ======================================================================== the pinned models (sections 4, 10.1; DC-4)


def test_qwen_loads_with_every_audio_changing_setting_explicit_s10_1(tmp_path: Path) -> None:
    profile = engine_profile(tmp_path)
    payload = qwen_load_payload(profile, "cuda:0")
    assert payload["model"] == {
        "repo": names.MODEL_QWEN_BASE,
        "revision": profile.model_revision,
        "snapshot_dir": profile.snapshot_dir,
    }
    assert payload["settings"] == {"non_streaming_mode": False, "generation": GENERATION}
    assert set(payload["determinism"]) == {"tf32", "cudnn_deterministic", "cudnn_benchmark", "deterministic_algorithms"}
    assert payload["engine_profile_id"] == profile.engine_profile_id
    assert ceiling(profile) == names.MAX_NEW_TOKENS_CEILING


def test_each_call_gets_its_own_cap_from_the_profile_dc4(tmp_path: Path) -> None:
    profile = engine_profile(tmp_path, max_new_tokens_per_char=3.0, max_new_tokens_floor=64)
    assert call_cap(profile, "x" * 10) == 64
    assert call_cap(profile, "x" * 100) == 300
    assert call_cap(profile, "x" * 10_000) == names.MAX_NEW_TOKENS_CEILING


@pytest.mark.parametrize(
    "missing", ["generation", "max_new_tokens_per_char", "max_new_tokens_floor", "non_streaming_mode"]
)
def test_a_profile_missing_a_setting_is_refused_not_defaulted_s10_1(tmp_path: Path, missing: str) -> None:
    profile = engine_profile(tmp_path)
    settings = {k: v for k, v in profile.settings.items() if k != missing}
    broken = dataclasses.replace(profile, settings=settings)
    with pytest.raises(ProfileError):
        call_cap(broken, "text")
        qwen_load_payload(broken, "cuda:0")


def test_the_qa_pins_name_their_models_by_repo_and_revision_s10_2() -> None:
    pin = ModelPin(repo="org/model", revision="c" * 40, snapshot_dir="<models_root>/x")
    pins = QaPins(asr=pin, sv=pin, aligner=pin, vram_need_mb=5000)
    assert pin.name == "org/model@" + "c" * 40
    payload = pins.load_payload("cuda:0")
    assert payload["device"] == "cuda:0" and set(payload["models"]) == {"asr", "sv", "aligner"}
    assert pins.key.startswith("qa:")


# ======================================================================== the clip's working copy (section 17)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[NarrationStore]:
    with NarrationStore(tmp_path / "store", StandInPlatform()) as s:
        yield s


def test_the_clip_is_copied_into_the_store_and_reused_s17(store: NarrationStore, tmp_path: Path) -> None:
    source = tmp_path / "caller" / "voice.wav"
    sha = write_clip(source)
    request = GenerateRequest.parse(
        _body(voice={"path": str(source), "sha256": sha, "transcript": "Words."}), DefaultsConfig()
    )
    checked: list[str] = []

    def check(path: str) -> Path:
        checked.append(path)
        return Path(path)

    staged = stage_clip(store, request.voice, check_path=check)
    assert staged == clip_path(store, sha) and hashlib.sha256(staged.read_bytes()).hexdigest() == sha
    assert checked == [str(source)]
    source.unlink()  # the caller's file may go; the working copy serves
    assert stage_clip(store, request.voice, check_path=check) == staged
    assert checked == [str(source)]


def test_a_clip_that_is_not_the_one_sent_is_voice_file_mismatch_s17(store: NarrationStore, tmp_path: Path) -> None:
    source = tmp_path / "voice.wav"
    write_clip(source)
    wrong = GenerateRequest.parse(
        _body(voice={"path": str(source), "sha256": "d" * 64, "transcript": "Words."}), DefaultsConfig()
    )
    with pytest.raises(NarrationError) as caught:
        stage_clip(store, wrong.voice)
    assert caught.value.code == codes.VOICE_FILE_MISMATCH and caught.value.field == "voice.sha256"
    assert not clip_path(store, "d" * 64).exists()
    assert not list(clip_path(store, "d" * 64).parent.glob("*.tmp"))

    gone = GenerateRequest.parse(
        _body(voice={"path": str(tmp_path / "nowhere.wav"), "sha256": "d" * 64, "transcript": "W."}), DefaultsConfig()
    )
    with pytest.raises(NarrationError) as missing:
        stage_clip(store, gone.voice)
    assert missing.value.code == codes.VOICE_FILE_MISMATCH and missing.value.field == "voice.path"
