"""Opt-in tests that load the Qwen models on the GPU (markers ``gpu``, ``model`` and ``slow``), and a pure check
of their helper.

- ``test_worker_designs_clones_and_repeats_the_canary_s10_1``: through the protocol, with only the service's
  own material (``material/canary/canary.v1``): design the canary voice with VoiceDesign, clone it with Base,
  render the gate text twice with the same seed (the same file bytes: the ``bit_exact`` tier, ADR 0002), and
  cut a render with a low call cap.
- ``test_acceptance_render_matches_the_bakeoff_take_s10_1`` (also ``evidence``): WP20's acceptance, which
  runs ``spikes/acceptance-wp20/run.py`` against the bakeoff's takes.
- ``TestQwen3WorkerContractOnTheGpu``: WP16's shared worker contract with a real model loaded (VoiceDesign),
  so its load round trip and its call-cap tests (DC-4: the cap cuts a take, an end token exactly at the cap
  is not a hit) run against the real worker. The default suite runs the same contract without a model.

**The GPU lock.** The GPU on a development machine is shared (AGENTS.md section 5), so the first GPU test
takes the lock through ``tools/gpu_lock.py`` for the whole session, after checking that free VRAM is at least
``NEED_MB`` plus 1 GB, and releases it at the end, on every exit path. The holder is
``$NARRATION_GPU_LOCK_HOLDER`` (default ``qwen3tts-gpu-tests``). If that holder already has the lock (a
wrapper script took it for a longer run), the tests use it and leave it to the wrapper; if anyone else holds
it, or VRAM is short, the tests skip.

Each skips with its reason when its resource is absent: an NVIDIA GPU, the lock tool, the lock,
``NARRATION_MODELS_ROOT``, the service's material, ``NARRATION_BAKEOFF_ROOT``, the QA worker's venv, or WP16's
worker loop.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

CHECKOUT = Path(__file__).resolve().parents[3]
LOCK_TOOL = CHECKOUT / "tools" / "gpu_lock.py"
LOCK_HOLDER_ENV = "NARRATION_GPU_LOCK_HOLDER"
DEFAULT_HOLDER = "qwen3tts-gpu-tests"
NEED_MB = 7000
"""VRAM the Qwen group needs, with room: about 5.8 GB measured for a 30 s render (spike h+i)."""
LOCK_MINUTES = 30
"""The lock covers the whole session: the longest bounded run AGENTS.md allows. The acceptance run's own
timeout (``ACCEPTANCE_TIMEOUT_S``) leaves room inside it for the other GPU tests."""
ACCEPTANCE_TIMEOUT_S = 15 * 60
DETERMINISM = {
    "tf32": False,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "deterministic_algorithms": "warn_only",
}
MODEL_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
PER_CHAR = 2.5
FLOOR = 128
"""The engines' shipped ``max_new_tokens_per_char`` and ``max_new_tokens_floor`` (``narration.example.toml``)."""


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


def _need_gpu() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs an NVIDIA GPU with CUDA")


def _cap_for(text: str) -> int:
    """The daemon's call cap for ``text``: ``narration.contracts.names.max_new_tokens_for`` with the shipped
    engine keys and the snapshots' ceiling, loaded from this checkout (the worker's venv has no server)."""
    name = "_narration_contract_names"
    names = sys.modules.get(name)
    if names is None:
        path = CHECKOUT / "src" / "narration" / "contracts" / "names.py"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            pytest.skip(f"the server's contracts are not at {path}")
        names = importlib.util.module_from_spec(spec)
        sys.modules[name] = names  # its dataclasses look their module up while the class is built
        spec.loader.exec_module(names)
    return int(names.max_new_tokens_for(text, per_char=PER_CHAR, floor=FLOOR, ceiling=8192))


def test_the_call_cap_helper_applies_the_contracts_rule_s10_1() -> None:
    """The GPU tests' call caps come from the server's own rule, loaded by path (pure: no GPU, no model).

    Regression: ``names.py`` defines dataclasses, which fail to build unless their module is registered in
    ``sys.modules`` while it runs."""
    assert _cap_for("") == FLOOR
    assert _cap_for("x" * 100) == 250
    assert _cap_for("x" * 10_000) == 8192
    assert _cap_for("x" * 100) == 250  # the second call reuses the loaded module


def _snapshot(repo: str) -> tuple[Path, str]:
    root = os.environ.get("NARRATION_MODELS_ROOT")
    if not root:
        pytest.skip("set NARRATION_MODELS_ROOT to the models root")
    manifest = Path(root) / "manifest.json"
    if not manifest.is_file():
        pytest.skip(f"no models manifest at {manifest}")
    for entry in json.loads(manifest.read_text(encoding="utf-8")).values():
        if entry["repo"] == repo:
            return Path(root) / entry["snapshot_dir"], entry["revision"]
    pytest.skip(f"{repo} is not installed under NARRATION_MODELS_ROOT")


def _material() -> Path:
    root = Path(os.environ.get("NARRATION_MATERIAL_ROOT", CHECKOUT / "material"))
    canary = root / "canary" / "canary.v1" / "canary.json"
    if not canary.is_file():
        pytest.skip(f"the service's canary material is not at {canary}")
    return canary


def _load_body(repo: str, non_streaming: bool) -> dict[str, Any]:
    """A ``load`` of ``repo`` from the models root, with every setting the snapshot pins passed explicitly."""
    from narration_qwen3tts.settings import effective_generation

    snapshot, revision = _snapshot(repo)
    return {
        "device": "cuda:0",
        "model": {"repo": repo, "revision": revision, "snapshot_dir": str(snapshot)},
        "engine_profile_id": "test",
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "determinism": dict(DETERMINISM),
        "settings": {"non_streaming_mode": non_streaming, "generation": dict(effective_generation(snapshot))},
    }


def _load(worker: Any, repo: str, non_streaming: bool) -> dict[str, Any]:
    return worker.request("load", timeout_s=600, **_load_body(repo, non_streaming))


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.slow
@pytest.mark.usefixtures("gpu_lock")
def test_worker_designs_clones_and_repeats_the_canary_s10_1(tmp_path: Path) -> None:
    _need_gpu()
    client = pytest.importorskip("narration_worker.testing.client", reason="needs WP16's narration_worker.testing")
    canary = json.loads(_material().read_text(encoding="utf-8"))
    _snapshot(MODEL_BASE)
    _snapshot(MODEL_DESIGN)
    store = tmp_path / "store"
    (store / "scratch").mkdir(parents=True)
    argv = [sys.executable, "-m", "narration_worker", "--role", "qwen3", "--store", str(store), "--cpu-threads", "8"]
    with client.WorkerProcess(argv) as worker:
        assert _load(worker, MODEL_DESIGN, non_streaming=True)["ok"]
        clip = store / "scratch" / "canary.wav"
        voice = canary["voice"]
        designed = worker.request(
            "design",
            timeout_s=900,
            description=voice["description"],
            design_text=voice["design_text"],
            language="English",
            seed=voice["seed"],
            max_new_tokens=_cap_for(voice["design_text"]),
            out_path=str(clip),
        )
        assert designed["ok"] and designed["hit_token_cap"] is False, designed
        assert designed["sample_rate"] == 24000 and designed["samples"] > 24000
        assert designed["max_new_tokens"] == _cap_for(voice["design_text"])

        assert _load(worker, MODEL_BASE, non_streaming=False)["ok"]
        prepared = worker.request(
            "prepare_voice",
            voice_hash="sha256:canary",
            ref_wav=str(clip),
            ref_text=voice["design_text"],
            x_vector_only_mode=False,
            timeout_s=300,
        )
        assert prepared["ok"], prepared
        gate = canary["gate"]
        cap = _cap_for(gate["text"])
        outs = [store / "scratch" / f"gate-{n}.wav" for n in (1, 2)]
        replies = [
            worker.request(
                "synthesize",
                timeout_s=900,
                voice_hash="sha256:canary",
                engine_text=gate["text"],
                language="English",
                seed=gate["seed"],
                max_new_tokens=cap,
                out_path=str(out),
            )
            for out in outs
        ]
        for reply in replies:
            assert reply["ok"] and reply["hit_token_cap"] is False and reply["max_new_tokens"] == cap, reply
            assert reply["new_tokens"] <= cap - 1
            # 12.5 codec frames per second at 24 kHz (spike h+i); the reference cut is proportional, hence the slack
            assert abs(reply["samples"] - reply["new_tokens"] * 1920) <= 1920, reply
        assert outs[0].read_bytes() == outs[1].read_bytes(), "the same request rendered different bytes (tier, s10.1)"

        capped = worker.request(
            "synthesize",
            timeout_s=300,
            voice_hash="sha256:canary",
            engine_text=gate["text"],
            language="English",
            seed=gate["seed"],
            max_new_tokens=24,
            out_path=str(store / "scratch" / "capped.wav"),
        )
        assert capped["ok"] and capped["hit_token_cap"] is True, capped
        assert (capped["new_tokens"], capped["max_new_tokens"]) == (23, 24), capped
        too_high = worker.request(
            "synthesize",
            timeout_s=300,
            voice_hash="sha256:canary",
            engine_text=gate["text"],
            language="English",
            seed=gate["seed"],
            max_new_tokens=8193,
            out_path=str(store / "scratch" / "never.wav"),
        )
        assert too_high["ok"] is False and too_high["error"]["code"] == "INVALID_REQUEST", too_high
        assert worker.request("unload")["ok"]
        assert worker.request("shutdown")["ok"]


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.slow
@pytest.mark.evidence
@pytest.mark.usefixtures("gpu_lock")
def test_acceptance_render_matches_the_bakeoff_take_s10_1(tmp_path: Path) -> None:
    """WP20's acceptance: WavLM similarity ≥ 0.98 to the bakeoff's take for the same text and seed."""
    _need_gpu()
    pytest.importorskip("narration_worker.testing.client", reason="needs WP16's narration_worker.testing")
    _snapshot(MODEL_BASE)
    if not os.environ.get("NARRATION_BAKEOFF_ROOT"):
        pytest.skip("set NARRATION_BAKEOFF_ROOT to the bakeoff's folder (its takes are the evidence)")
    if not os.environ.get("NARRATION_SPIKE_ALLOW_SHA256"):
        pytest.skip("set NARRATION_SPIKE_ALLOW_SHA256 to the sha256 of each clip the spikes may clone")
    qa = CHECKOUT / "workers" / "qa" / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not qa.is_file():
        pytest.skip("the QA worker's venv is not synced (it scores the similarity)")
    out = tmp_path / "acceptance.json"
    script = CHECKOUT / "spikes" / "acceptance-wp20" / "run.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--out", str(out)], check=False, timeout=ACCEPTANCE_TIMEOUT_S
    )
    results = json.loads(out.read_text(encoding="utf-8"))
    scores = {f"{r['voice']} {r['segment']}": r["similarity_to_bakeoff_take"] for r in results["renders"]}
    assert completed.returncode == 0 and results["accepted"], scores


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
class TestQwen3WorkerContractOnTheGpu(_contract_base()):
    """WP16's worker contract with VoiceDesign loaded on the GPU: every test, including the ones that need a
    model (the load round trip, and the call cap's cut and end-token cases, section 10.1 and DC-4)."""

    role = "qwen3"
    timeout_s = 600.0

    @pytest.fixture
    def load_request(self) -> dict[str, Any]:
        _need_gpu()
        return _load_body(MODEL_DESIGN, non_streaming=True)

    @pytest.fixture
    def render_requests(self, store_root: Path) -> list[tuple[str, dict[str, Any]]]:
        from narration_worker.testing.contract import sample_requests

        return [("design", sample_requests(store_root)["design"])]
