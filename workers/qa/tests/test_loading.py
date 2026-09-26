"""How the QA worker's three models load, and refuse to (design section 4; plan.md WP22 and WP15's follow-up F4).

Each loader (Whisper, WavLM-SV, the CTC aligner) is run on real, tiny snapshots (``conftest.tiny_snapshot``)
with their weights in ``pytorch_model.bin`` (as the pinned WavLM and wav2vec2 revisions have) or in
``model.safetensors`` (as Whisper's has), then damaged. A damaged or mismatched file is ``BACKEND_NOT_INSTALLED``
with the hint to install the models again; a failure that is not the install's (running out of memory, a bug)
is not reported as one.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from narration_worker_qa import align as qa
from narration_worker_qa.asr import WhisperAsr
from narration_worker_qa.sv import WavLmSv

torch = pytest.importorskip("torch", reason="needs torch and transformers: the QA worker's full venv")
pytest.importorskip("transformers", reason="needs torch and transformers: the QA worker's full venv")

torch.set_num_threads(4)

REVISIONS = {"asr": "a" * 40, "sv": "b" * 40, "aligner": "54074b1c16f4de6a5ad59affb4caa8f2ea03a119"}


# ---------------------------------------------------------------------- is_broken_file (WP15 F4)


class SafetensorError(Exception):
    """Named as safetensors' Rust error class is (it is not importable as a Python class)."""


@pytest.mark.parametrize(
    "exc",
    [
        OSError("no file named pytorch_model.bin"),
        FileNotFoundError("config.json"),
        EOFError("Ran out of input"),
        pickle.UnpicklingError("invalid load key, '\\x00'."),
        SafetensorError("Error while deserializing header: HeaderTooLarge"),
        RuntimeError("PytorchStreamReader failed reading zip archive: failed finding central directory"),
        RuntimeError("Trying to resize storage that is not resizable"),
        RuntimeError("Error(s) in loading state_dict: size mismatch for lm_head.weight: copying a param ..."),
        RuntimeError("You set `ignore_mismatched_sizes` to `False`, thus raising an error."),
        OSError("The process cannot access the file because it is being used by another process"),
    ],
    ids=lambda exc: type(exc).__name__,
)
def test_a_missing_damaged_or_mismatched_file_is_broken_s4(exc: BaseException) -> None:
    assert qa.is_broken_file(exc)


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB"),
        RuntimeError("DefaultCPUAllocator: not enough memory: you tried to allocate 1073741824 bytes"),
        RuntimeError("expected scalar type Half but found Float"),
        ValueError("a bug in a library"),
        TypeError("a bug in the handler"),
        KeyError("model.encoder"),
    ],
    ids=["cuda_oom", "cpu_oom", "dtype_bug", "value_error", "type_error", "key_error"],
)
def test_a_failure_that_is_not_the_installs_is_not_broken_s4(exc: BaseException) -> None:
    # Regression (WP15 F4): any RuntimeError was taken for a broken file, so running out of memory told the
    # user to reinstall the models.
    assert not qa.is_broken_file(exc)


# ---------------------------------------------------------------------- loaders on tiny snapshots


def _load(use: str, path: Path) -> Any:
    if use == "asr":
        model: Any = WhisperAsr(torch)
        model.load("example/asr", REVISIONS["asr"], path, "cpu")
    elif use == "sv":
        model = WavLmSv(torch)
        model.load("example/sv", REVISIONS["sv"], path, "cpu")
    else:
        model = qa.Wav2Vec2Aligner(torch)
        model.load("example/aligner", REVISIONS["aligner"], path)
    return model


def _weights(path: Path) -> Path:
    [weights] = [p for p in path.iterdir() if p.name in {"model.safetensors", "pytorch_model.bin"}]
    return weights


def _remove_weights(path: Path) -> None:
    _weights(path).unlink()


def _garble_weights(path: Path) -> None:
    _weights(path).write_bytes(bytes(range(256)) * 20)


def _truncate_weights(path: Path) -> None:
    weights = _weights(path)
    weights.write_bytes(weights.read_bytes()[: weights.stat().st_size // 2])


def _empty_weights(path: Path) -> None:
    _weights(path).write_bytes(b"")


def _break_config(path: Path) -> None:
    (path / "config.json").write_text("{not json", encoding="utf-8")


def _remove_config(path: Path) -> None:
    (path / "config.json").unlink()


def _remove_preprocessor(path: Path) -> None:
    """Remove the feature extractor's settings (transformers 5 saves a processor's in ``processor_config.json``)."""
    name = "preprocessor_config.json" if (path / "preprocessor_config.json").is_file() else "processor_config.json"
    (path / name).unlink()


def _resize(path: Path) -> None:
    """Change a width in ``config.json`` so the weights no longer fit it."""
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    key = {"whisper": "d_model", "wavlm": "xvector_output_dim", "wav2vec2": "hidden_size"}[config["model_type"]]
    config[key] = config[key] * 2
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


DAMAGE = [
    _remove_weights,
    _garble_weights,
    _truncate_weights,
    _empty_weights,
    _break_config,
    _remove_config,
    _remove_preprocessor,
    _resize,
]


@pytest.mark.parametrize("safetensors", [False, True], ids=["bin", "safetensors"])
@pytest.mark.parametrize("use", ["asr", "sv", "aligner"])
@pytest.mark.parametrize("damage", DAMAGE, ids=lambda f: f.__name__.lstrip("_"))
def test_a_damaged_snapshot_is_backend_not_installed_s4(
    tiny_snapshot: Callable[..., Path], use: str, safetensors: bool, damage: Callable[[Path], None]
) -> None:
    path = tiny_snapshot(use, safetensors=safetensors)
    assert _weights(path).name == ("model.safetensors" if safetensors else "pytorch_model.bin")
    damage(path)
    with pytest.raises(qa.BackendMissing, match="install the models again") as caught:
        _load(use, path)
    assert caught.value.code == "BACKEND_NOT_INSTALLED"


@pytest.mark.parametrize("safetensors", [False, True], ids=["bin", "safetensors"])
@pytest.mark.parametrize("use", ["asr", "sv", "aligner"])
def test_a_tiny_snapshot_loads_offline_with_weights_only_s4(
    tiny_snapshot: Callable[..., Path], use: str, safetensors: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transformers

    cls = {"asr": "WhisperForConditionalGeneration", "sv": "WavLMForXVector", "aligner": "Wav2Vec2ForCTC"}[use]
    seen: dict[str, Any] = {}
    real = getattr(transformers, cls).from_pretrained

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(getattr(transformers, cls), "from_pretrained", spy)
    model = _load(use, tiny_snapshot(use, safetensors=safetensors))
    assert model.loaded
    assert seen["weights_only"] is True and seen["local_files_only"] is True
    assert seen["dtype"] is torch.float32  # on the CPU; Whisper is float16 on a GPU
    if use == "asr":
        assert seen["attn_implementation"] == "eager"


def test_whisper_loads_with_eager_attention_s11_1(tiny_snapshot: Callable[..., Path]) -> None:
    # Word timestamps switch the attention to eager inside generate; loading it so keeps both paths alike.
    asr = _load("asr", tiny_snapshot("asr"))
    assert asr._pipeline.model.config._attn_implementation == "eager"


@pytest.mark.parametrize(
    ("use", "other"),
    [("asr", "sv"), ("sv", "asr"), ("sv", "aligner"), ("asr", "aligner")],
)
def test_another_models_snapshot_is_refused_s4(tiny_snapshot: Callable[..., Path], use: str, other: str) -> None:
    path = tiny_snapshot(other)
    with pytest.raises(qa.BackendMissing, match="is not a") as caught:
        _load(use, path)
    assert caught.value.code == "BACKEND_NOT_INSTALLED"


@pytest.mark.parametrize("use", ["asr", "sv"])
def test_a_failure_that_is_not_the_installs_propagates_s4(
    tiny_snapshot: Callable[..., Path], use: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transformers

    cls = {"asr": "WhisperForConditionalGeneration", "sv": "WavLMForXVector"}[use]
    path = tiny_snapshot(use)

    def out_of_memory(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(getattr(transformers, cls), "from_pretrained", out_of_memory)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        _load(use, path)


@pytest.mark.parametrize("use", ["asr", "sv"])
def test_a_file_another_process_holds_is_a_transient_internal_error_s4(
    tiny_snapshot: Callable[..., Path], use: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transformers

    cls = {"asr": "WhisperForConditionalGeneration", "sv": "WavLMForXVector"}[use]
    path = tiny_snapshot(use)

    def held(*args: Any, **kwargs: Any) -> Any:
        raise OSError("The process cannot access the file because it is being used by another process")

    monkeypatch.setattr(getattr(transformers, cls), "from_pretrained", held)
    with pytest.raises(qa.QaError) as caught:
        _load(use, path)
    assert caught.value.code == "INTERNAL" and caught.value.details["transient"] is True


def test_the_tiny_sv_model_embeds_as_the_bakeoff_did_s11_1(tiny_snapshot: Callable[..., Path]) -> None:
    """The embedding is the bake-off's recipe (``eval/evaluate.py``, ``Scorer.embed``): the snapshot's feature
    extractor, ``embeddings[0]``, L2-normalised."""
    from transformers import AutoFeatureExtractor, WavLMForXVector

    path = tiny_snapshot("sv")
    audio = np.random.default_rng(3).standard_normal(16_000).astype(np.float32) * 0.1
    reply = _load("sv", path).embed(audio, "cpu")
    extractor = AutoFeatureExtractor.from_pretrained(str(path))
    model = WavLMForXVector.from_pretrained(str(path)).eval()
    with torch.no_grad():
        expected = torch.nn.functional.normalize(
            model(**extractor(audio, sampling_rate=16_000, return_tensors="pt")).embeddings[0], dim=-1
        )
    assert reply["dim"] == 8 and len(reply["embedding"]) == 8
    assert np.allclose(reply["embedding"], expected.numpy(), atol=1e-6)
    assert abs(float(np.linalg.norm(reply["embedding"])) - 1.0) < 1e-6
