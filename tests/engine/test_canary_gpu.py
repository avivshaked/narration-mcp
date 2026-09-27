"""The real canary on this machine's GPU (markers ``gpu``, ``model`` and ``slow``; design section 10.1, DC-3,
ADR 0002): ``engine pin`` with the real Qwen and QA workers, then the canary gate after a fresh Qwen load.

It pins both engines in a fresh store under the test's temporary folder: it designs the canary with
VoiceDesign, clones it with Base, runs the repeat test (two renders in one process, one in a fresh one) and
calibrates each threshold with WavLM on the QA worker's CPU. Then it loads each profile again through the
daemon's worker supervisor and runs the gate, which must pass on the hash in the ``bit_exact`` tier. Nothing
it renders leaves the temporary folder.

Run it (under 25 minutes on a 24 GB GPU; the Qwen models need about 6 GB and the QA models run on the CPU)::

    uv run python -m pytest -m "gpu and model" tests/engine/test_canary_gpu.py -s

It needs ``NARRATION_MODELS_ROOT`` with the pinned models installed (``narration-admin install``), the Qwen and
QA worker venvs synced (under ``NARRATION_WORKERS_ROOT``, else this checkout's ``workers/``), the QA worker's
``embed`` op (WP22), an NVIDIA GPU with NVML, and the developers' GPU lock (``tools/gpu_lock.py``; holder
``$NARRATION_GPU_LOCK_HOLDER``, default ``engine-gpu-tests``). It skips with the reason when one is missing.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from narration.config import Config, GpuConfig, WorkerProject
from narration.contracts.names import EngineKind
from narration.daemon.supervisor import WorkerSupervisor
from narration.engine.canary import CanaryGuard
from narration.engine.installed import probe_for
from narration.engine.models import PINNED
from narration.engine.pinning import SupervisedStarter, pin
from narration.engine.profile import QWEN_VRAM_MB
from narration.engine.qa import qa_pins
from narration.jobs.gpu import LOAD_TIMEOUT_S
from narration.jobs.host import RunnerHost
from narration.jobs.pins import qwen_load_payload
from narration.platform import get_platform
from narration.store import NarrationStore
from narration.workers import venv_python

CHECKOUT = Path(__file__).resolve().parents[2]
LOCK_TOOL = CHECKOUT / "tools" / "gpu_lock.py"
LOCK_HOLDER_ENV = "NARRATION_GPU_LOCK_HOLDER"
DEFAULT_HOLDER = "engine-gpu-tests"
LOCK_MINUTES = 30
WAIT_MINUTES = 10
"""How long to wait for the lock when someone else holds it (``--wait-min``)."""
RUN_TIMEOUT_S = 25 * 60
"""Inside the lock's 30 minutes. The first run (2026-09-27) was still in the repeat test's fresh process when the
suite's default 300 s ran out, so the run gets its own limit. A run stopped by a timeout cannot release the
lock (pytest-timeout ends the process); the lock then lapses at its expected end, and the kill-on-close group
has already stopped the workers."""
NEED_MB = QWEN_VRAM_MB + 1000
DEVICE = "cuda:0"

pytestmark = [pytest.mark.gpu, pytest.mark.model, pytest.mark.slow, pytest.mark.timeout(RUN_TIMEOUT_S)]


def _lock(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(LOCK_TOOL), *args], cwd=CHECKOUT, capture_output=True, text=True, check=False
    )


@pytest.fixture(scope="module")
def gpu_lock() -> Iterator[None]:
    """The developers' GPU lock for this module's run (AGENTS.md section 5), released on every exit path."""
    if not LOCK_TOOL.is_file():
        pytest.skip(f"the GPU lock tool is not at {LOCK_TOOL}")
    holder = os.environ.get(LOCK_HOLDER_ENV) or DEFAULT_HOLDER
    status = _lock("status")
    if status.returncode == 0 and status.stdout.strip() != "free":
        owner = json.loads(status.stdout).get("owner") or {}
        if owner.get("holder") == holder:
            yield  # a wrapper holds it for us, and releases it itself
            return
    taken = _lock(
        "acquire",
        "--holder",
        holder,
        "--minutes",
        str(LOCK_MINUTES),
        "--need-mb",
        str(NEED_MB),
        "--wait-min",
        str(WAIT_MINUTES),
    )
    if taken.returncode != 0:  # held by someone else past the wait, or too little free VRAM
        pytest.skip(f"not starting a GPU run: {taken.stderr.strip() or taken.stdout.strip()}")
    try:
        yield
    finally:
        _lock("release", "--holder", holder)


def _config(store_root: Path) -> Config:
    """This machine's models and worker venvs, a fresh store, and the GPU; skips when any is missing."""
    models = os.environ.get("NARRATION_MODELS_ROOT")
    if not models:
        pytest.skip("set NARRATION_MODELS_ROOT to the models root")
    models_root = Path(models)
    for model in PINNED.values():
        if not model.snapshot_dir(models_root).is_dir():
            pytest.skip(f"{model.repo} at {model.revision} is not installed under NARRATION_MODELS_ROOT")
    workers = Path(os.environ.get("NARRATION_WORKERS_ROOT", CHECKOUT / "workers"))
    qwen3, qa = workers / "qwen3tts", workers / "qa"
    for project in (qwen3, qa):
        if not venv_python(project).is_file():
            pytest.skip(f"the worker venv of {project} is not synced")
    if not (qa / "src" / "narration_worker_qa" / "sv.py").is_file():
        pytest.skip("the QA worker has no embed op yet (WP22)")
    reading = probe_for(DEVICE).read()
    if reading is None:
        pytest.skip("no NVIDIA GPU readable through NVML")
    config = Config.for_tests(store_root, models_root)
    return dataclasses.replace(
        config,
        gpu=GpuConfig(device=DEVICE),
        workers=dataclasses.replace(config.workers, qwen3=WorkerProject(project=qwen3), qa=WorkerProject(project=qa)),
    )


@dataclasses.dataclass
class _Host:
    """What the gate reads of the daemon's host."""

    store: NarrationStore
    config: Config
    workers: WorkerSupervisor


@pytest.mark.usefixtures("gpu_lock")
def test_the_canary_pinned_on_this_gpu_passes_its_own_gate_s10_1(tmp_path: Path) -> None:
    config = _config(tmp_path / "store")
    platform = get_platform()
    with NarrationStore(config.server.store_root, platform) as store:
        with SupervisedStarter(config, platform) as starter:
            reports = {r.kind: r for r in pin(config, store, mode="pin", starter=starter)}
        for kind, report in reports.items():
            print(f"{kind}: {report.as_dict()}")  # the tier and threshold measured here (-s shows them)
            assert report.action == "new" and report.tier in ("bit_exact", "similar")
            assert report.threshold is not None and 0.0 < report.threshold < 1.0
            assert len(report.repeat_sha256) == 3
        # Another seed designs another voice; another seed of one voice's clone stays close to it.
        assert reports["base"].threshold is not None and reports["design"].threshold is not None
        assert reports["base"].threshold > reports["design"].threshold

        guard = CanaryGuard(sv=qa_pins(config).sv)
        with WorkerSupervisor(config, platform) as pool:
            host = cast(RunnerHost, _Host(store=store, config=config, workers=pool))
            kinds: tuple[EngineKind, ...] = ("design", "base")
            for kind in kinds:
                profile = store.current_engine_profile(kind)
                assert profile is not None
                cublas = profile.determinism.cublas_workspace_config
                pool.load(
                    "qwen",
                    qwen_load_payload(profile, DEVICE),
                    timeout_s=LOAD_TIMEOUT_S,
                    cublas_workspace_config=cublas,
                )
                hello: Any = pool.client("qwen", cublas_workspace_config=cublas).hello
                status = guard.after_load(host, profile, hello)
                print(f"{kind}: the gate says {status}")
                expected = ("hash_match",) if profile.tier == "bit_exact" else ("hash_match", "similarity_pass")
                assert status in expected
                pool.stop("qwen")
