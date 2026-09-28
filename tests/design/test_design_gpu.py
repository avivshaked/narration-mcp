"""``design_voice`` on this machine's GPU (markers ``gpu``, ``model`` and ``slow``; design sections 3.1, 7.6, 17.4):
the real VoiceDesign and QA workers design two candidates from an invented, positive-only description, and each
one is checked the way a caller would use it.

It pins both engines in a fresh store under the test's temporary folder (``engine pin``: the canary gate needs
the design engine's canary), submits the design through the backend, so the service's own design text is used,
and runs the job with the daemon's job engine (``installed_engine``) over the real worker supervisor. Then, for
each candidate: the clip renders and its bytes hash to the sha256 reported; the exact transcript passes the
ASR check; the hash is on the provenance list; and ``measure_voice`` admits the clip with no allowlist edit.
Last, ``profile_voice`` profiles a tone on the real QA worker, loaded with no model. Nothing it renders leaves
the temporary folder.

Run it (the Qwen models need about 6 GB of VRAM; the pin alone took 8 minutes on a 24 GB GPU)::

    uv run python -m pytest -m "gpu and model" tests/design/test_design_gpu.py -s

It needs what ``tests/engine/test_canary_gpu.py`` needs: ``NARRATION_MODELS_ROOT`` with the pinned models, the
Qwen and QA worker venvs synced (under ``NARRATION_WORKERS_ROOT``, else this checkout's ``workers/``), an NVIDIA
GPU with NVML, and the developers' GPU lock (holder ``$NARRATION_GPU_LOCK_HOLDER``, default
``design-gpu-tests``). It skips with the reason when one is missing.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import soundfile

from narration.config import Config
from narration.contracts.models import JobRecord
from narration.contracts.names import PROFILE_VERSION
from narration.daemon.supervisor import WorkerSupervisor
from narration.engine.pinning import SupervisedStarter, pin
from narration.jobs.runner import EngineRunner, installed_engine
from narration.jobs.voice import require_synthetic
from narration.platform import ProcessPlatform, get_platform
from narration.store import NarrationStore
from tests.backend.support import make_backend, sha256_of, write_wav
from tests.engine import test_canary_gpu as canary_gpu
from tests.jobs.support import Host, MonotonicClock

DESCRIPTION = (
    "A warm, unhurried storyteller in her forties, with a low, even voice, clear diction and a gentle smile in "
    "every sentence, like a friend reading aloud by a quiet fire."
)
"""Invented for this test, and positive only: it says what the voice is, never what it is not (section 3.5)."""
TAKES = 2
MAX_STEPS = 400
HOLDER = "design-gpu-tests"
TONE_HZ = 220.0

pytestmark = [pytest.mark.gpu, pytest.mark.model, pytest.mark.slow, pytest.mark.timeout(canary_gpu.RUN_TIMEOUT_S)]


def _lock(*args: str) -> tuple[int, str]:
    done = canary_gpu._lock(*args)  # pyright: ignore[reportPrivateUsage]
    return done.returncode, (done.stderr.strip() or done.stdout.strip())


@pytest.fixture(scope="module")
def gpu_lock() -> Iterator[None]:
    """The developers' GPU lock (AGENTS.md section 5), taken as the canary test takes it, and released on every
    exit path."""
    if not canary_gpu.LOCK_TOOL.is_file():
        pytest.skip(f"the GPU lock tool is not at {canary_gpu.LOCK_TOOL}")
    holder = os.environ.get(canary_gpu.LOCK_HOLDER_ENV) or HOLDER
    code, said = _lock("status")
    if code == 0 and said != "free" and (json.loads(said).get("owner") or {}).get("holder") == holder:
        yield  # a wrapper holds it for us, and releases it itself
        return
    minutes, need, wait = str(canary_gpu.LOCK_MINUTES), str(canary_gpu.NEED_MB), str(canary_gpu.WAIT_MINUTES)
    code, said = _lock("acquire", "--holder", holder, "--minutes", minutes, "--need-mb", need, "--wait-min", wait)
    if code != 0:
        pytest.skip(f"not starting a GPU run: {said}")
    try:
        yield
    finally:
        _lock("release", "--holder", holder)


def _run_alone(config: Config, platform: ProcessPlatform, store: NarrationStore, job_id: str) -> JobRecord:
    """Run the daemon's job engine over a fresh worker pool until ``job_id`` ends; the pool stops after."""
    with WorkerSupervisor(config, platform) as pool:
        host = Host(store=store, config=config, workers=pool, clock=MonotonicClock(), real_sleep_s=5.0)
        runner = EngineRunner(installed_engine)
        try:
            return _run(runner, host, job_id, store)
        finally:
            runner.shutdown(host, "now")


def _run(runner: EngineRunner, host: Host, job_id: str, store: NarrationStore) -> JobRecord:
    for _ in range(MAX_STEPS):
        job = store.get_job(job_id)
        if job is not None and job.status in ("completed", "failed", "cancelled"):
            return job
        if not runner.step(host):
            break
    job = store.get_job(job_id)
    assert job is not None
    return job


@pytest.mark.usefixtures("gpu_lock")
def test_two_designed_candidates_are_measurable_voices_s3_1_s17_4(tmp_path: Path) -> None:
    config = canary_gpu._config(tmp_path / "store")  # pyright: ignore[reportPrivateUsage]
    platform = get_platform()
    with NarrationStore(config.server.store_root, platform) as store:
        started = time.monotonic()
        with SupervisedStarter(config, platform) as starter:
            reports = {r.kind: r for r in pin(config, store, mode="pin", starter=starter)}
        print(f"pinned in {time.monotonic() - started:.0f} s: design {reports['design'].as_dict()}")

        backend, _, _ = make_backend(SimpleNamespace(store=store, config=config))
        request = {"name": "fireside narrator", "description": DESCRIPTION, "takes": TAKES}
        submitted = backend.design_voice_sync(request)
        assert submitted["lint"]["findings"] == [], "the description is positive only"

        started = time.monotonic()
        job = _run_alone(config, platform, store, submitted["job_id"])
        print(f"the design job: {job.status} {job.outcome} in {time.monotonic() - started:.0f} s; {job.result}")
        assert job.status == "completed", job.error

        # profile_voice on the real QA worker, with no model loaded (a fresh worker pool: on the CPU).
        tone = tmp_path / "elsewhere" / "tone.wav"
        tone_sha = write_wav(tone, seconds=3.0, freq=TONE_HZ)
        profiling = backend.profile_voice_sync({"audio": {"path": str(tone), "sha256": tone_sha}})
        started = time.monotonic()
        job = _run_alone(config, platform, store, profiling["job_id"])
        print(f"the profile job: {job.status} in {time.monotonic() - started:.0f} s")
        assert job.status == "completed", job.error
        record = store.get_profile(tone_sha, PROFILE_VERSION)
        assert record is not None
        print(f"the tone's profile: {record.measurements}")
        assert record.measurements.pitch_median_hz == pytest.approx(TONE_HZ, rel=0.05)
        assert record.measurements.speaking_rate_wpm is None
        assert Path(record.pictures.spectrogram).is_file() and Path(record.pictures.pitch).is_file()

        candidates = store.get_design(submitted["design_id"])
        assert [c.index for c in candidates] == list(range(TAKES))
        assert len({c.clip.sha256 for c in candidates}) == TAKES, "each seed designs its own voice"
        max_clip_s = config.limits.max_clip_seconds
        for candidate in candidates:
            clip = Path(candidate.clip.path)
            info = soundfile.info(str(clip))
            check, profile = candidate.transcript_check, candidate.profile
            assert check is not None and profile is not None
            m = profile.measurements
            print(
                f"candidate {candidate.index}: seed {candidate.seed}, {info.duration:.2f} s at {info.samplerate} Hz;"
                f" transcript ok {check.ok}, wer {check.wer:.3f}; pitch median {m.pitch_median_hz} Hz,"
                f" rate {m.speaking_rate_wpm} wpm, centroid {m.spectral_centroid_hz} Hz, {m.loudness_lufs} LUFS"
            )
            assert sha256_of(clip) == candidate.clip.sha256, "the clip renders to the bytes reported"
            assert info.duration > 1.0
            assert check.ok, check
            assert store.is_provenance(candidate.clip.sha256)
            require_synthetic(store, (), candidate.clip.sha256)
            assert info.duration <= max_clip_s, f"measure_voice admits clips up to {max_clip_s} s"
            voice = {"path": str(clip), "sha256": candidate.clip.sha256, "transcript": candidate.transcript}
            admitted = backend.measure_voice_sync({"voice": voice})
            assert admitted["status"] in ("queued", "completed"), admitted
