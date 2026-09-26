"""WavLM-SV embeddings and their devices (design sections 10.1 and 11.1; plan.md WP22), on a tiny, random WavLM
(``conftest.tiny_snapshot``). The embedding recipe itself is checked in ``test_loading.py``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from narration_worker_qa import align as qa
from narration_worker_qa.sv import WavLmSv, min_samples

torch = pytest.importorskip("torch", reason="needs torch and transformers: the QA worker's full venv")
pytest.importorskip("transformers", reason="needs torch and transformers: the QA worker's full venv")

torch.set_num_threads(4)


@pytest.fixture
def sv(tiny_snapshot: Callable[..., Path]) -> WavLmSv:
    model = WavLmSv(torch)
    model.load("example/sv", "b" * 40, tiny_snapshot("sv"), "cpu")
    return model


def _audio(seconds: float = 1.0, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(int(16_000 * seconds)) * 0.1).astype(np.float32)


def test_an_embedding_is_unit_length_and_repeatable_s11_1(sv: WavLmSv) -> None:
    first = sv.embed(_audio(), "cpu")
    assert (first["model"], first["revision"], first["dim"]) == ("example/sv", "b" * 40, 8)
    assert abs(float(np.linalg.norm(first["embedding"])) - 1.0) < 1e-6
    assert sv.embed(_audio(), "cpu") == first
    assert sv.embed(_audio(seed=1), "cpu")["embedding"] != first["embedding"]


def test_embedding_on_cuda_needs_the_model_on_a_gpu_s10_1(sv: WavLmSv) -> None:
    with pytest.raises(qa.InvalidRequest) as caught:
        sv.embed(_audio(), "cuda")
    assert caught.value.details["field"] == "device"


def test_a_device_other_than_cuda_or_cpu_is_an_invalid_request_s11_1(sv: WavLmSv) -> None:
    with pytest.raises(qa.InvalidRequest):
        sv.embed(_audio(), "cuda:0")


def test_a_model_on_a_gpu_embeds_on_the_cpu_from_a_copy_s10_1(sv: WavLmSv) -> None:
    # The canary check embeds on the CPU while the QA group may be on the GPU: the worker loads a CPU copy from
    # the same snapshot on first use, and keeps it until unload. (No GPU here: the loaded model is relabelled.)
    expected = sv.embed(_audio(), "cpu")
    gpu_model = sv._models.pop("cpu")
    sv._models["cuda:0"] = gpu_model
    sv.device = "cuda:0"
    assert sv.embed(_audio(), "cpu") == expected
    copy = sv._models["cpu"]
    assert copy is not gpu_model
    assert sv.embed(_audio(), "cpu") == expected and sv._models["cpu"] is copy
    sv.unload()
    assert not sv.loaded and sv._models == {}


def test_audio_too_short_to_embed_is_unsupported_audio_s11_1(sv: WavLmSv) -> None:
    assert sv.min_samples == 5200  # the pinned model's layout, which the tiny one keeps
    with pytest.raises(qa.UnreadableAudio) as caught:
        sv.embed(np.full(sv.min_samples - 1, 0.01, dtype=np.float32), "cpu")
    assert caught.value.code == "UNSUPPORTED_AUDIO" and caught.value.details["reason"] == "too_short"
    shortest = sv.embed(_audio(sv.min_samples / 16_000), "cpu")
    assert shortest["dim"] == 8 and np.isfinite(shortest["embedding"]).all()


def test_min_samples_follows_the_models_layout_s11_1() -> None:
    from transformers import WavLMConfig

    assert min_samples(WavLMConfig()) == 400 + 15 * 320
    assert min_samples(WavLMConfig(tdnn_kernel=(1,), tdnn_dilation=(1,), tdnn_dim=(8,))) == 400 + 320


def test_embed_before_load_is_not_loaded_s11_1() -> None:
    with pytest.raises(qa.NotLoaded):
        WavLmSv(torch).embed(_audio(), "cpu")
