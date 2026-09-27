"""The canary gate as the job engine meets it after a Qwen load (design section 10.1; plan.md DC-3, WP32): the
fingerprint check, then the gate render, a hash match or WavLM's similarity on the QA worker's CPU, and
``ENGINE_DRIFT`` before anything renders on an engine that cannot be vouched for.

The profiles are pinned by ``engine pin`` itself, on fake workers, so each gate here checks what a pin
recorded. The workers are ``narration_worker``'s fake role; nothing needs a GPU or a model.
"""

from __future__ import annotations

import dataclasses
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from narration import keys
from narration.config import MeasurementConfig
from narration.contracts import codes, names
from narration.contracts.errors import NarrationError, WorkerFailure
from narration.contracts.models import EngineProfile, EngineRef, ProvenanceEntry
from narration.contracts.names import CanaryStatus, EngineKind
from narration.contracts.worker import HelloReply
from narration.engine.canary import (
    THRESHOLD_FLOOR,
    CanaryGuard,
    calibrate,
    calibration_seeds,
    find_canary,
    floored,
    gate_facts,
)
from narration.engine.models import QWEN_BASE
from narration.engine.pinning import pin
from narration.engine.qa import qa_pins
from narration.jobs.core import EngineParts
from narration.jobs.engine import JobEngine
from narration.jobs.pins import qwen_load_payload
from narration.jobs.runner import EngineRunner
from narration.platform.testing import StandInPlatform
from narration.post import DeliveryPipeline
from narration.qa import Scorer
from narration.store import NarrationStore
from narration.store.store import utc_iso
from narration.text import TextPipeline
from narration.workers import SubprocessWorkerClient, worker_command
from tests.jobs import support as jobs

from .support import FAKE_WORKER_PACKAGES, CountingStarter, Install, fake_install, hello, write_lock

LOAD_TIMEOUT_S = 60.0


@dataclasses.dataclass
class Gate:
    """A pinned installation, the daemon's host over a pool of fake workers, and the guard the daemon builds."""

    install: Install
    store: NarrationStore
    pool: jobs.FakePool
    host: jobs.Host
    guard: CanaryGuard
    spec: Path

    def profile(self, kind: EngineKind = "base") -> EngineProfile:
        profile = self.store.current_engine_profile(kind)
        assert profile is not None and profile.canary is not None
        return profile

    def load(self, profile: EngineProfile) -> HelloReply | None:
        """Load Qwen with the profile as the job engine does, and return the worker's ``hello``."""
        cublas = profile.determinism.cublas_workspace_config
        self.pool.load(
            "qwen", qwen_load_payload(profile, "cpu"), timeout_s=LOAD_TIMEOUT_S, cublas_workspace_config=cublas
        )
        return self.pool.client("qwen", cublas_workspace_config=cublas).hello

    def check(self, profile: EngineProfile) -> CanaryStatus:
        """Load, then run the gate on what loaded."""
        return self.guard.after_load(self.host, profile, self.load(profile))

    def ops(self, group: str) -> list[str]:
        return [op for g, op, _ in self.pool.requests if g == group]

    def sent(self, group: str, op: str) -> list[dict[str, Any]]:
        return [payload for g, o, payload in self.pool.requests if g == group and o == op]

    def faults(self, *faults: dict[str, Any]) -> None:
        jobs.write_spec(self.spec, *faults)

    def scratch_wavs(self) -> list[Path]:
        return sorted((self.store.root / "scratch").rglob("*.wav"))


@pytest.fixture
def gate(tmp_path: Path) -> Iterator[Gate]:
    install = fake_install(tmp_path)
    store = NarrationStore(install.store_root, StandInPlatform(), alignment_method_id=jobs.METHOD_ID)
    spec = tmp_path / "fake-spec.json"
    jobs.write_spec(spec)
    pool = jobs.FakePool(install.config, spec)
    try:
        with CountingStarter(install.config) as starter:
            pin(install.config, store, mode="pin", starter=starter, packages=FAKE_WORKER_PACKAGES, device="cpu")
        host = jobs.Host(store=store, config=install.config, workers=pool, clock=jobs.MonotonicClock())
        yield Gate(
            install=install,
            store=store,
            pool=pool,
            host=host,
            guard=CanaryGuard(sv=qa_pins(install.config).sv),
            spec=spec,
        )
    finally:
        pool.close()
        store.close()


def _with_canary(profile: EngineProfile, **changes: Any) -> EngineProfile:
    assert profile.canary is not None
    return dataclasses.replace(profile, canary=dataclasses.replace(profile.canary, **changes))


def _drift(caught: pytest.ExceptionInfo[NarrationError]) -> dict[str, Any]:
    error = caught.value
    assert error.code == codes.ENGINE_DRIFT and error.retryable is False
    assert error.details is not None
    return dict(error.details)


# ======================================================================== the gate passes


def test_a_canary_that_repeats_its_pinned_bytes_passes_on_the_hash_alone_s10_1(gate: Gate) -> None:
    profile = gate.profile("base")
    material = find_canary()
    assert profile.canary is not None

    assert gate.check(profile) == "hash_match"
    assert gate.ops("qwen") == ["load", "prepare_voice", "synthesize"]
    (synth,) = gate.sent("qwen", "synthesize")
    assert synth["engine_text"] == material.gate_text and synth["seed"] == profile.canary.seed
    assert isinstance(synth["max_new_tokens"], int)  # every call passes its own cap (DC-4)
    (prepare,) = gate.sent("qwen", "prepare_voice")
    assert prepare["ref_wav"] == profile.canary.clip.path and prepare["ref_text"] == profile.canary.transcript
    assert gate.ops("qa") == []  # a matching hash needs no QA and no model swap
    assert gate.scratch_wavs() == []  # the gate render is removed


def test_the_voicedesign_gate_render_is_the_canary_design_itself_s10_1(gate: Gate) -> None:
    profile = gate.profile("design")
    material = find_canary()

    assert gate.check(profile) == "hash_match"
    assert gate.ops("qwen") == ["load", "design"]
    (design,) = gate.sent("qwen", "design")
    assert (design["description"], design["design_text"], design["seed"]) == (
        material.description,
        material.design_text,
        material.design_seed,
    )


def test_a_render_that_differs_like_another_seed_passes_on_similarity_on_the_cpu_s10_1(gate: Gate) -> None:
    """The threshold is calibrated with the next seeds: a render that diverges no more than one of them passes."""
    pinned = gate.profile("base")
    assert pinned.canary is not None
    another_seed = _with_canary(pinned, seed=calibration_seeds(pinned.canary.seed)[0])

    assert gate.check(another_seed) == "similarity_pass"
    assert gate.ops("qa") == ["load", "embed", "unload"]
    (load,) = gate.sent("qa", "load")
    assert load == {"device": "cpu", "models": {"sv": qa_pins(gate.install.config).sv.ref()}}  # WavLM alone
    (embed,) = gate.sent("qa", "embed")
    assert embed["device"] == "cpu"
    assert gate.pool.gpu_holder == "qwen"  # Qwen stays loaded: the canary never swaps it out
    assert "qa" not in gate.pool.loaded()  # and the QA worker's CPU copy is unloaded again
    assert gate.scratch_wavs() == []


def test_a_canary_below_its_threshold_is_engine_drift_s10_1(gate: Gate) -> None:
    pinned = gate.profile("base")
    assert pinned.canary is not None
    drifted = _with_canary(pinned, seed=calibration_seeds(pinned.canary.seed)[0], threshold=1.5)

    with pytest.raises(NarrationError) as caught:
        gate.check(drifted)
    details = _drift(caught)
    assert details["canary"] == "below_threshold" and details["similarity"] < details["threshold"] == 1.5
    assert details["engine_profile_id"] == pinned.engine_profile_id and details["tier"] == "bit_exact"
    assert "qa" not in gate.pool.loaded()
    assert gate.scratch_wavs() == []


def test_a_calibration_below_the_floor_is_clamped_to_it_s10_1() -> None:
    """A canary whose calibration renders share nothing with it cannot calibrate the gate to pass everything."""
    threshold, sims = calibrate((1.0, 0.0), [(0.0, 1.0), (-1.0, 0.0)])
    assert sims == (0.0, -1.0) and threshold == THRESHOLD_FLOOR and floored(threshold)
    threshold, _ = calibrate((1.0, 0.0), [(1.0, 0.1)])
    assert threshold > THRESHOLD_FLOOR and not floored(threshold)


def test_engine_show_says_when_a_threshold_is_the_floor_s10_1(gate: Gate) -> None:
    pinned = gate.profile("base")
    assert pinned.canary is not None
    assert gate_facts(pinned.canary)["threshold_floored"] is False
    assert gate_facts(dataclasses.replace(pinned.canary, threshold=THRESHOLD_FLOOR))["threshold_floored"] is True


@pytest.mark.parametrize(
    ("embedding", "reason"),
    [((), "pinned_embedding_empty"), ((0.0, 0.0, 0.0), "pinned_embedding_all_zero")],
)
def test_a_pinned_embedding_that_cannot_be_compared_is_drift_before_rendering_s10_1(
    gate: Gate, embedding: tuple[float, ...], reason: str
) -> None:
    broken = _with_canary(gate.profile("base"), embedding=embedding)
    with pytest.raises(NarrationError) as caught:
        gate.check(broken)
    assert _drift(caught)["canary"] == reason and "engine repin" in caught.value.hint
    assert gate.ops("qwen") == ["load"]  # nothing rendered


def test_a_pinned_embedding_of_another_length_is_drift_not_a_similarity_of_zero_s10_1(gate: Gate) -> None:
    pinned = gate.profile("base")
    assert pinned.canary is not None
    shorter = _with_canary(
        pinned, seed=calibration_seeds(pinned.canary.seed)[0], embedding=pinned.canary.embedding[:-1]
    )
    with pytest.raises(NarrationError) as caught:
        gate.check(shorter)
    details = _drift(caught)
    assert details["canary"] == "embedding_length"
    assert details["pinned_dim"] == details["dim"] - 1
    assert "qa" not in gate.pool.loaded()


# ======================================================================== drift before the gate renders


def test_a_changed_uv_lock_is_drift_before_the_canary_renders_s10_1(gate: Gate) -> None:
    lock = gate.install.project / "uv.lock"
    lock.write_text(lock.read_text(encoding="utf-8") + "\n# synced again\n", encoding="utf-8")

    with pytest.raises(NarrationError) as caught:
        gate.check(gate.profile("base"))
    details = _drift(caught)
    assert [(d["what"], d["name"]) for d in details["drift"]] == [("uv_lock", "uv.lock")]
    assert gate.ops("qwen") == ["load"]  # nothing rendered


def test_a_changed_or_added_weight_file_is_drift_before_the_canary_renders_s10_1(gate: Gate) -> None:
    snapshot = gate.install.snapshot(QWEN_BASE)
    (snapshot / "model.safetensors").write_bytes(b"other weights")
    (snapshot / "speech_tokenizer" / "extra.safetensors").write_bytes(b"one more file")

    with pytest.raises(NarrationError) as caught:
        gate.check(gate.profile("base"))
    drift = {d["name"]: d for d in _drift(caught)["drift"]}
    assert set(drift) == {"model.safetensors", "speech_tokenizer/extra.safetensors"}
    assert drift["speech_tokenizer/extra.safetensors"]["pinned"] is None
    assert gate.ops("qwen") == ["load"]


def test_a_venv_synced_to_another_lock_is_drift_s10_1(gate: Gate) -> None:
    """The lock and the running worker agree with each other but not with the pin: the worker's hello says so."""
    profile = gate.profile("base")
    write_lock(gate.install.project, {"narration-worker": "0.0.1"})
    with pytest.raises(NarrationError) as caught:
        gate.guard.after_load(gate.host, profile, cast(HelloReply, hello({"narration-worker": "0.0.1"})))
    whats = [(d["what"], d["name"]) for d in _drift(caught)["drift"]]
    assert ("package", "narration-worker") in whats and ("uv_lock", "uv.lock") in whats


@pytest.mark.parametrize(
    ("reply", "what"),
    [
        (None, ("fingerprint", "hello")),
        ("no_cublas", ("env", "CUBLAS_WORKSPACE_CONFIG")),
    ],
)
def test_a_worker_whose_hello_does_not_show_the_pin_is_drift_s10_1(
    gate: Gate, reply: str | None, what: tuple[str, str]
) -> None:
    profile = gate.profile("base")
    sent = None if reply is None else cast(HelloReply, hello(dict(profile.packages), cublas=None))
    with pytest.raises(NarrationError) as caught:
        gate.guard.after_load(gate.host, profile, sent)
    assert [(d["what"], d["name"]) for d in _drift(caught)["drift"]] == [what]
    assert gate.pool.requests == []


# ======================================================================== a gate that cannot run


def test_a_profile_pinned_without_a_canary_cannot_run_s10_1(gate: Gate) -> None:
    profile = dataclasses.replace(gate.profile("base"), canary=None)
    with pytest.raises(NarrationError) as caught:
        gate.guard.after_load(gate.host, profile, gate.load(profile))
    assert caught.value.code == codes.BACKEND_NOT_INSTALLED and "engine repin" in caught.value.hint
    assert gate.ops("qwen") == ["load"]


def test_a_canary_text_changed_since_the_pin_is_drift_dc3(gate: Gate) -> None:
    changed = dataclasses.replace(find_canary(), sha256="0" * 64)
    guard = CanaryGuard(sv=qa_pins(gate.install.config).sv, material=changed)
    with pytest.raises(NarrationError) as caught:
        guard.after_load(gate.host, gate.profile("base"), gate.load(gate.profile("base")))
    details = _drift(caught)
    assert details["canary"] == "material" and "engine repin" in caught.value.hint
    assert gate.ops("qwen") == ["load"]


def test_a_canary_the_worker_cannot_render_is_drift_s10_1(gate: Gate) -> None:
    gate.faults({"kind": "error", "op": "synthesize", "code": "INTERNAL", "message": "the render broke"})
    with pytest.raises(NarrationError) as caught:
        gate.check(gate.profile("base"))
    details = _drift(caught)
    assert details["canary"] == "not_rendered" and details["worker_code"] == "INTERNAL"
    assert gate.scratch_wavs() == []


def test_out_of_memory_during_the_canary_is_left_to_the_job_engine_s4(gate: Gate) -> None:
    """Running out of GPU memory is not drift: the job engine waits and loads again (section 4 item 5)."""
    gate.faults({"kind": "gpu_oom", "op": "synthesize"})
    with pytest.raises(WorkerFailure) as caught:
        gate.check(gate.profile("base"))
    assert caught.value.code == codes.GPU_OOM


def test_a_canary_that_cannot_be_measured_is_drift_and_leaves_qa_unloaded_s10_1(gate: Gate) -> None:
    pinned = gate.profile("base")
    assert pinned.canary is not None
    gate.faults({"kind": "error", "op": "embed", "code": "INTERNAL", "message": "no embedding"})
    with pytest.raises(NarrationError) as caught:
        gate.check(_with_canary(pinned, seed=calibration_seeds(pinned.canary.seed)[0]))
    assert _drift(caught)["canary"] == "not_measured"
    assert "qa" not in gate.pool.loaded() and gate.ops("qa")[-1] == "unload"


# ======================================================================== a job on the pinned engine


def _anchor(gate: Gate, clip: Path) -> tuple[float, ...]:
    """The fake's speaker embedding of the test voice's clip, which its takes sit near (a worker reads only
    files in the store, so it embeds a copy there)."""
    pins = jobs.qa_pins(gate.install.models_root)
    copy = gate.store.scratch_path("anchor", "clip.wav")
    shutil.copyfile(clip, copy)
    try:
        with SubprocessWorkerClient(worker_command(gate.install.config, "fake")) as client:
            client.start()
            client.request("load", pins.load_payload("cpu"), timeout_s=LOAD_TIMEOUT_S)
            reply = client.request("embed", {"wav": str(copy), "device": "cpu"}, timeout_s=LOAD_TIMEOUT_S)
    finally:
        copy.unlink(missing_ok=True)
    return tuple(float(v) for v in reply["embedding"])


def _job_runner(gate: Gate, tmp_path: Path) -> tuple[EngineRunner, JobEngine, dict[str, Any]]:
    """The job engine with the installed guard, and a measured test voice under the pinned Base profile."""
    config, store = gate.install.config, gate.store
    profile = gate.profile("base")
    clip = tmp_path / "voice" / "clip.wav"
    clip_sha256 = jobs.write_clip(clip)
    store.add_provenance(ProvenanceEntry(clip_sha256=clip_sha256, design_id=jobs.DESIGN_ID, date=utc_iso(time.time())))
    base = jobs.measurement(clip_sha256, _anchor(gate, clip))
    store.put_measurement(
        dataclasses.replace(
            base,
            engine_profile=EngineRef(id=profile.engine_profile_id, hash=profile.hash),
            measurement_key=keys.measurement_key(
                voice_hash=base.voice_hash,
                engine_profile_hash=profile.hash,
                corpus_version=f"{names.CORPUS}@sha256:{'0' * 64}",
                settings=MeasurementConfig(),
            ),
        )
    )
    parts = EngineParts(
        text=TextPipeline(config.text),
        delivery=DeliveryPipeline(fade_s=config.delivery.fade_s),
        scorer=Scorer(config.measurement),
        aligner=jobs.TestAligner(),
        qa_pins=jobs.qa_pins(gate.install.models_root),
        guard=gate.guard,
        clock=gate.host.clock,
        defer_s=0.05,
    )
    engine = JobEngine(config, parts)
    return EngineRunner(engine), engine, jobs.request(clip, clip_sha256, jobs.LAMPS)


def test_a_job_on_the_pinned_engine_records_the_canarys_hash_match_s10_1(gate: Gate, tmp_path: Path) -> None:
    runner, engine, body = _job_runner(gate, tmp_path)
    try:
        job = jobs.submit(gate.store, body)
        jobs.drive(runner, gate.host)
    finally:
        engine.close()
    done = gate.store.get_job(job.job_id)
    assert done is not None and done.status == "completed", done
    render = gate.store.get_render(done.items[0].attempts[0].render_key)
    assert render is not None and render.canary.batch_status == "hash_match"
    assert not any(f.code == codes.CANARY_MISMATCH for f in done.items[0].flags)
    texts = gate.pool.texts()
    assert texts == [find_canary().gate_text, jobs.LAMPS]  # the gate first, then the job


def test_a_job_on_a_drifted_engine_fails_before_it_renders_s10_1(gate: Gate, tmp_path: Path) -> None:
    runner, engine, body = _job_runner(gate, tmp_path)
    (gate.install.snapshot(QWEN_BASE) / "model.safetensors").write_bytes(b"other weights")
    try:
        job = jobs.submit(gate.store, body)
        jobs.drive(runner, gate.host)
    finally:
        engine.close()
    failed = gate.store.get_job(job.job_id)
    assert failed is not None and failed.status == "failed"
    assert failed.error is not None and failed.error.code == codes.ENGINE_DRIFT
    assert gate.pool.texts() == []  # neither the canary nor the job rendered
    assert "qwen" not in gate.pool.loaded()  # the refused load is unloaded
