"""Opt-in tests that run the QA worker with the real models on the GPU (markers ``gpu``, ``model`` and ``slow``).

- ``test_worker_runs_every_op_on_the_gpu_s4``: through the protocol, with synthetic audio only: load Whisper,
  WavLM and the aligner, transcribe, embed on the GPU and on the CPU (the canary's path), f0, profile, unload.
- ``test_acceptance_matches_the_bakeoff_eval_s11_1`` (also ``evidence``): WP22's acceptance, which runs
  ``spikes/acceptance-wp22/run.py`` against the bakeoff's takes.
- ``TestQaWorkerContractOnTheGpu``: WP16's shared worker contract with the real models loaded on the GPU. The
  default suite runs the same contract with tiny models on the CPU.

This test process sees no GPU (``conftest.py``); each test starts the worker as a process of its own and gives
it the GPU back.

**The GPU lock.** The GPU on a development machine is shared (AGENTS.md section 5), so the first GPU test takes
the lock through ``tools/gpu_lock.py`` for the whole session, after checking that free VRAM is at least
``NEED_MB`` plus 1 GB, and releases it at the end, on every exit path. The holder is
``$NARRATION_GPU_LOCK_HOLDER`` (default ``qa-gpu-tests``). If that holder already has the lock (a wrapper script
took it for a longer run), the tests use it and leave it to the wrapper; if anyone else holds it, or VRAM is
short, the tests skip.

Each skips with its reason when its resource is absent: an NVIDIA GPU, the lock tool, the lock,
``NARRATION_MODELS_ROOT``, or ``NARRATION_BAKEOFF_ROOT``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

CHECKOUT = Path(__file__).resolve().parents[3]
LOCK_TOOL = CHECKOUT / "tools" / "gpu_lock.py"
LOCK_HOLDER_ENV = "NARRATION_GPU_LOCK_HOLDER"
DEFAULT_HOLDER = "qa-gpu-tests"
GPU_ENV = "NARRATION_QA_TESTS_CUDA_VISIBLE_DEVICES"
NEED_MB = 11_500
"""VRAM the QA group needs at its peak, for a take of any length: the proposed ``vram_need_mb`` (spike h, QA half:
``spikes/h-i-qa-load``)."""
RESIDENT_MB = (3000, 4500)
"""The bounds of the reserved VRAM a load of Whisper and WavLM reports (3606 MB measured, spike h)."""
LOCK_MINUTES = 30
ACCEPTANCE_TIMEOUT_S = 20 * 60
REPOS = {
    "asr": ("openai/whisper-large-v3", ""),
    "sv": ("microsoft/wavlm-base-plus-sv", "feb593a6"),
    "aligner": ("facebook/wav2vec2-large-960h-lv60-self", ""),
}
"""The pinned models by use, with the prefix of the revision to pick where the models root holds two."""


def _run_lock(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(LOCK_TOOL), *args], cwd=CHECKOUT, capture_output=True, text=True, check=False
    )


@pytest.fixture(scope="session")
def gpu_lock() -> Iterator[None]:
    """Hold the developers' GPU lock for the session's GPU tests (see the module docstring)."""
    if not LOCK_TOOL.is_file():
        pytest.skip(f"the GPU lock tool is not at {LOCK_TOOL}")
    holder = os.environ.get(LOCK_HOLDER_ENV) or DEFAULT_HOLDER
    status = _run_lock("status")
    if status.returncode == 0 and status.stdout.strip() != "free":
        owner = json.loads(status.stdout).get("owner") or {}
        if owner.get("holder") == holder:
            yield  # a wrapper holds it for us, and releases it itself
            return
        pytest.skip(f"the GPU lock is held by someone else: {status.stdout.strip()}")
    taken = _run_lock(
        "acquire", "--holder", holder, "--minutes", str(LOCK_MINUTES), "--need-mb", str(NEED_MB)
    )  # fmt: skip
    if taken.returncode == 1:
        pytest.skip(f"the GPU lock is held by someone else: {taken.stderr.strip()}")
    if taken.returncode != 0:
        pytest.skip(f"not starting a GPU test: {taken.stderr.strip() or taken.stdout.strip()}")
    try:
        yield
    finally:
        _run_lock("release", "--holder", holder)


def _gpu_env() -> dict[str, str]:
    """This process's environment with the GPU given back (``conftest.py`` hid it)."""
    env = dict(os.environ)
    original = env.get(GPU_ENV, "<unset>")
    if original == "<unset>":
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = original
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return env


def _need_gpu() -> None:
    """Skip unless NVML sees an NVIDIA GPU (torch cannot tell here: this process hides the GPU)."""
    pynvml = pytest.importorskip("pynvml")
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:
        pytest.skip(f"needs an NVIDIA GPU with a working driver: {exc}")
    try:
        if pynvml.nvmlDeviceGetCount() < 1:
            pytest.skip("needs an NVIDIA GPU")
    finally:
        pynvml.nvmlShutdown()


def _ref(use: str) -> dict[str, str]:
    root = os.environ.get("NARRATION_MODELS_ROOT")
    if not root:
        pytest.skip("set NARRATION_MODELS_ROOT to the models root")
    manifest = Path(root) / "manifest.json"
    if not manifest.is_file():
        pytest.skip(f"no models manifest at {manifest}")
    repo, prefix = REPOS[use]
    for entry in json.loads(manifest.read_text(encoding="utf-8")).values():
        if entry["repo"] == repo and entry["revision"].startswith(prefix):
            return {
                "repo": repo,
                "revision": entry["revision"],
                "snapshot_dir": str(Path(root) / entry["snapshot_dir"]),
            }
    pytest.skip(f"{repo} is not installed under NARRATION_MODELS_ROOT")


def _load_body(*uses: str) -> dict[str, Any]:
    return {"device": "cuda:0", "models": {use: _ref(use) for use in uses}}


def _worker(store: Path) -> Any:
    client = pytest.importorskip("narration_worker.testing.client", reason="needs WP16's narration_worker.testing")
    argv = [sys.executable, "-m", "narration_worker", "--role", "qa", "--store", str(store), "--cpu-threads", "8"]
    return client.WorkerProcess(argv, env=_gpu_env())


def _tone_wav(path: Path, seconds: float, rate: int = 24_000) -> Path:
    """A harmonic tone with a slow vibrato, in noise: sound, but not speech."""
    rng = np.random.default_rng(11)
    t = np.arange(int(seconds * rate)) / rate
    phase = 2 * np.pi * np.cumsum(140 * (1 + 0.05 * np.sin(2 * np.pi * 4 * t))) / rate
    audio = sum(np.sin(k * phase) / k for k in range(1, 12)) * 0.2 + 0.005 * rng.standard_normal(t.shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), rate, subtype="FLOAT")
    return path


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.slow
@pytest.mark.usefixtures("gpu_lock")
def test_worker_runs_every_op_on_the_gpu_s4(tmp_path: Path) -> None:
    _need_gpu()
    body = _load_body("asr", "sv", "aligner")
    store = tmp_path / "store"
    wav = _tone_wav(store / "scratch" / "tone.wav", 35.0)
    with _worker(store) as worker:
        loaded = worker.request("load", timeout_s=600, **body)
        assert loaded["ok"], loaded
        assert isinstance(loaded["vram_mb"], int) and RESIDENT_MB[0] < loaded["vram_mb"] < RESIDENT_MB[1], loaded

        heard = worker.request(
            "transcribe", timeout_s=600, wav=str(wav), language="English", word_timestamps=True, long_form=True
        )
        assert heard["ok"], heard
        assert (heard["model"], heard["revision"]) == (body["models"]["asr"]["repo"], body["models"]["asr"]["revision"])
        for word in heard["words"]:
            assert word["start_s"] is None or word["end_s"] is None or word["start_s"] <= word["end_s"] <= 35.5

        on_gpu = worker.request("embed", timeout_s=300, wav=str(wav), device="cuda")
        again = worker.request("embed", timeout_s=300, wav=str(wav), device="cuda")
        on_cpu = worker.request("embed", timeout_s=300, wav=str(wav), device="cpu")
        assert on_gpu["ok"] and on_cpu["ok"] and on_gpu["dim"] == 512
        assert again["embedding"] == on_gpu["embedding"]  # the determinism switches: the same bits twice
        assert float(np.dot(on_gpu["embedding"], on_cpu["embedding"])) > 0.999

        track = worker.request("f0", timeout_s=300, wav=str(wav), fmin_hz=50.0, fmax_hz=400.0)
        voiced = [f for f in track["f0_hz"] if f is not None]
        assert voiced and abs(float(np.median(voiced)) - 140.0) < 3.0

        aligned = worker.request("align", timeout_s=300, wav=str(wav), tokens=list("RAIN|CAME"))
        assert aligned["ok"] and aligned["device"] == "cpu", aligned

        out_dir = store / "scratch" / "profile"
        profile = worker.request("profile", timeout_s=300, wav=str(wav), out_dir=str(out_dir), transcript=None)
        assert profile["ok"] and profile["measurements"]["pitch_median_hz"] == pytest.approx(140.0, abs=3.0)

        assert worker.request("unload", timeout_s=120)["ok"]
        after = worker.request("embed", timeout_s=60, wav=str(wav), device="cuda")
        assert after["error"]["code"] == "NOT_LOADED"
        assert worker.request("shutdown", timeout_s=120)["ok"]


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.slow
@pytest.mark.evidence
@pytest.mark.usefixtures("gpu_lock")
def test_acceptance_matches_the_bakeoff_eval_s11_1(tmp_path: Path) -> None:
    """WP22's acceptance: through the protocol, the worker's transcripts and similarities of the bakeoff's clone
    takes match the bakeoff's ``eval/`` numbers within the script's tolerances."""
    _need_gpu()
    _ref("asr")
    _ref("sv")
    if not os.environ.get("NARRATION_BAKEOFF_ROOT"):
        pytest.skip("set NARRATION_BAKEOFF_ROOT to the bakeoff's folder (its takes are the evidence)")
    out = tmp_path / "acceptance.json"
    script = CHECKOUT / "spikes" / "acceptance-wp22" / "run.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--out", str(out)], env=_gpu_env(), check=False, timeout=ACCEPTANCE_TIMEOUT_S
    )
    results = json.loads(out.read_text(encoding="utf-8"))
    assert completed.returncode == 0 and results["accepted"], results["failures"]


def _contract_base() -> type:
    try:
        from narration_worker.testing.contract import WorkerContract
    except ImportError:  # WP16's contract suite is not in this checkout: a class with no tests
        return object
    return WorkerContract


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.slow
@pytest.mark.usefixtures("gpu_lock")
class TestQaWorkerContractOnTheGpu(_contract_base()):
    """WP16's worker contract with the real models loaded on the GPU."""

    role = "qa"
    timeout_s = 600.0

    @pytest.fixture
    def worker_env(self) -> dict[str, str]:
        return _gpu_env()

    @pytest.fixture
    def load_request(self) -> dict[str, Any]:
        _need_gpu()
        return _load_body("asr", "sv", "aligner")
