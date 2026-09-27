"""Support for the engine tests: an installation in miniature (plan.md WP32).

Snapshots at the service's pinned revisions hold a few small files in place of the weights, and the Qwen
worker's project holds a ``uv.lock`` naming the packages a profile pins. Nothing here needs a GPU, a model or
the network; the tests that run a worker use ``narration_worker``'s ``fake`` role.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from narration.config import Config, WorkerProject
from narration.contracts.interfaces import WorkerClient
from narration.engine.models import CTC_ALIGNER, QWEN_BASE, QWEN_DESIGN, WAVLM_SV, WHISPER, PinnedModel
from narration.engine.pinning import SubprocessStarter
from narration.engine.profile import QWEN_PACKAGES

GENERATION: Final[dict[str, Any]] = {
    "do_sample": True,
    "repetition_penalty": 1.05,
    "temperature": 0.9,
    "top_p": 1.0,
    "top_k": 50,
    "subtalker_dosample": True,
    "subtalker_temperature": 0.9,
    "subtalker_top_p": 1.0,
    "subtalker_top_k": 50,
    "max_new_tokens": 8192,
}
"""What the pinned Qwen snapshots' ``generation_config.json`` sets: all ten sampling values."""

VERSIONS: Final[dict[str, str]] = {name: f"1.{i}.0" for i, name in enumerate(QWEN_PACKAGES)}
"""Made-up locked versions for the packages a profile pins."""
FAKE_WORKER_PACKAGES: Final = ("narration-worker",)
"""The one package the fake worker reports in its fingerprint."""


def make_snapshot(models_root: Path, model: PinnedModel, *, generation: Mapping[str, Any] | None = None) -> Path:
    """A snapshot folder at the model's pinned revision with a config, a generation config and small weights."""
    folder = model.snapshot_dir(models_root)
    (folder / "speech_tokenizer").mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps({"model": model.repo}), encoding="utf-8")
    (folder / "generation_config.json").write_text(json.dumps(dict(generation or GENERATION)), encoding="utf-8")
    (folder / "model.safetensors").write_bytes(model.repo.encode("utf-8") * 64)
    (folder / "speech_tokenizer" / "model.safetensors").write_bytes(b"tokenizer" * 64)
    return folder


def write_lock(project: Path, versions: Mapping[str, str]) -> Path:
    """A ``uv.lock`` in the shape uv writes, naming each package and its version."""
    project.mkdir(parents=True, exist_ok=True)
    lines = ["version = 1", "revision = 3", 'requires-python = "==3.12.*"', ""]
    for name, version in versions.items():
        lines += ["[[package]]", f'name = "{name}"', f'version = "{version}"', ""]
    path = project / "uv.lock"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@dataclasses.dataclass
class Install:
    """A miniature installation: the config, and where its models, store and Qwen worker project are."""

    config: Config
    models_root: Path
    store_root: Path
    project: Path

    def snapshot(self, model: PinnedModel) -> Path:
        return model.snapshot_dir(self.models_root)


def make_install(root: Path, *, versions: Mapping[str, str] | None = None, qa: bool = True) -> Install:
    """Snapshots of both Qwen models (and the QA models, unless ``qa`` is False), a Qwen worker project with a
    lock, and a config naming them."""
    models_root = root / "models"
    store_root = root / "store"
    project = root / "workers" / "qwen3tts"
    (store_root / "scratch").mkdir(parents=True, exist_ok=True)
    for model in (QWEN_BASE, QWEN_DESIGN):
        make_snapshot(models_root, model)
    if qa:
        for model in (WHISPER, WAVLM_SV, CTC_ALIGNER):
            make_snapshot(models_root, model)
    write_lock(project, versions if versions is not None else VERSIONS)
    config = Config.for_tests(store_root, models_root)
    config = dataclasses.replace(
        config, workers=dataclasses.replace(config.workers, qwen3=WorkerProject(project=project))
    )
    return Install(config=config, models_root=models_root, store_root=store_root, project=project)


def fake_install(root: Path) -> Install:
    """An installation whose Qwen lock names the fake worker's package at the version it reports, so a profile
    pinned with ``FAKE_WORKER_PACKAGES`` matches the fake's ``hello``."""
    return make_install(root, versions={"narration-worker": importlib.metadata.version("narration-worker")})


class CountingStarter:
    """Starts fake workers in place of the real ones (``engine pin``'s ``Starter``), and counts them."""

    def __init__(self, config: Config) -> None:
        base = {k: v for k, v in os.environ.items() if k != "NARRATION_FAKE_SPEC"}
        self.inner = SubprocessStarter(config, role="fake", base_env=base)
        self.started: list[str] = []

    def start(self, role: Any, *, cublas_workspace_config: str | None = None) -> WorkerClient:
        self.started.append(role)
        return self.inner.start(role, cublas_workspace_config=cublas_workspace_config)


def hello(packages: Mapping[str, str], *, cublas: str | None = ":4096:8") -> dict[str, Any]:
    """A ``hello`` reply as a worker sends it, with these package versions and this cuBLAS pin."""
    env = {"CUBLAS_WORKSPACE_CONFIG": cublas} if cublas is not None else {}
    return {
        "id": 1,
        "ok": True,
        "role": "qwen3",
        "protocol": 1,
        "capabilities": {"ops": []},
        "fingerprint": {
            "python": "3.12.0",
            "platform": "test",
            "packages": dict(packages),
            "cuda": "12.8",
            "cudnn": "91900",
            "gpu": "Test GPU",
            "driver": "1.0",
            "cpu_threads": 8,
            "env": env,
        },
    }
