"""The ``qa`` role's handler: its requests, their order of checks, and its models' lifecycle (design section 4 and
Appendix A; plan.md WP22), on tiny, random snapshots of the three models (``conftest.tiny_snapshot``), on the CPU.

The shared worker contract (``test_worker_contract.py``) runs the same handler as a process.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from narration_worker.errors import OpError
from narration_worker.handler import WorkerContext
from narration_worker_qa import worker as qa_worker
from narration_worker_qa.asr import WhisperAsr
from narration_worker_qa.sv import WavLmSv
from narration_worker_qa.voice import PICTURES, PROFILE_KEYS
from narration_worker_qa.worker import QaHandler

torch = pytest.importorskip("torch", reason="needs torch and transformers: the QA worker's full venv")
pytest.importorskip("transformers", reason="needs torch and transformers: the QA worker's full venv")

torch.set_num_threads(4)

REVISIONS = {"asr": "a" * 40, "sv": "b" * 40, "aligner": "54074b1c16f4de6a5ad59affb4caa8f2ea03a119"}
MODEL_OPS = ("transcribe", "embed", "f0", "align", "profile")


@pytest.fixture
def store(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    (root / "scratch").mkdir(parents=True)
    return root


@pytest.fixture
def handler(store: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[QaHandler]:
    """A handler over ``store``; the environment and torch's determinism switches are restored afterwards."""
    for name in ("MPLCONFIGDIR", "NUMBA_CACHE_DIR", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # as the worker's start-up sets it
    saved = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
    )
    worker = QaHandler(WorkerContext(role="qa", store_root=store.resolve(), cpu_threads=2))
    try:
        yield worker
    finally:
        worker.shutdown()
        torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = saved[2], saved[3]
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved[4], saved[5]


def _ref(use: str, path: Path) -> dict[str, str]:
    return {"repo": f"example/{use}", "revision": REVISIONS[use], "snapshot_dir": str(path)}


@pytest.fixture
def snapshots(tiny_snapshot: Callable[..., Path]) -> dict[str, Path]:
    return {
        "asr": tiny_snapshot("asr", safetensors=True),
        "sv": tiny_snapshot("sv"),
        "aligner": tiny_snapshot("aligner"),
    }


def _load(handler: QaHandler, snapshots: dict[str, Path], *uses: str, device: str = "cpu") -> dict[str, Any]:
    return handler.handle("load", {"device": device, "models": {u: _ref(u, snapshots[u]) for u in uses}})


def _error(call: Callable[[], Any]) -> OpError:
    with pytest.raises(OpError) as caught:
        call()
    return caught.value


def _wav(store: Path, name: str = "take.wav", seconds: float = 2.0, rate: int = 24_000) -> Path:
    t = np.arange(int(seconds * rate)) / rate
    audio = 0.3 * np.sin(2 * np.pi * 140 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    path = store / "scratch" / name
    sf.write(str(path), audio.astype(np.float32), rate, subtype="FLOAT")
    return path


def _requests(store: Path, wav: Path) -> dict[str, dict[str, Any]]:
    return {
        "transcribe": {"wav": str(wav), "language": "English", "word_timestamps": False, "long_form": True},
        "embed": {"wav": str(wav), "device": "cpu"},
        "f0": {"wav": str(wav), "fmin_hz": 50.0, "fmax_hz": 400.0},
        "align": {"wav": str(wav), "tokens": list("RAIN|CAME")},
        "profile": {"wav": str(wav), "out_dir": str(store / "scratch" / "profile"), "transcript": "rain came"},
    }


# ---------------------------------------------------------------------- start-up


def test_the_handler_keeps_its_caches_in_the_store_and_loads_offline_s17_2(handler: QaHandler, store: Path) -> None:
    cache = store.resolve().joinpath(*qa_worker.CACHE_DIR)
    assert os.environ["MPLCONFIGDIR"] == str(cache / "matplotlib")
    assert os.environ["NUMBA_CACHE_DIR"] == str(cache / "numba")
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert handler.missing_ops() == []


# ---------------------------------------------------------------------- load


def test_a_missing_snapshot_is_backend_not_installed_before_any_other_check_s4(handler: QaHandler, store: Path) -> None:
    missing = {"repo": "example/asr", "revision": "not-hex", "snapshot_dir": str(store / "none")}
    error = _error(lambda: handler.handle("load", {"device": "no-such-device", "models": {"asr": missing}}))
    assert error.code == "BACKEND_NOT_INSTALLED" and "install the models" in error.message
    assert error.details == {"field": "models.asr", "repo": "example/asr", "snapshot_dir": str(store / "none")}


@pytest.mark.parametrize(
    ("models", "field"),
    [
        (None, "models"),
        ("asr", "models"),
        ({"asr": "x"}, "models.asr"),
        ({"asr": {"repo": "r", "revision": "0" * 40}}, "models.asr"),
        ({"sv": {"repo": 1, "revision": "0" * 40, "snapshot_dir": "x"}}, "models.sv"),
        ({"sv": {"repo": "r", "revision": "0" * 40, "snapshot_dir": "relative/" + "0" * 40}}, "models.sv.snapshot_dir"),
    ],
)
def test_a_malformed_models_member_is_an_invalid_request_s4(handler: QaHandler, models: Any, field: str) -> None:
    request: dict[str, Any] = {"device": "no-such-device"}
    if models is not None:
        request["models"] = models
    error = _error(lambda: handler.handle("load", request))
    assert (error.code, (error.details or {}).get("field")) == ("INVALID_REQUEST", field)


def test_snapshot_references_are_checked_as_the_other_workers_check_them_s4(
    handler: QaHandler, snapshots: dict[str, Path], store: Path
) -> None:
    """The fake's and the ``qwen3`` worker's order, codes and fields (WP16's review): shape and an absolute path,
    then every folder present, then each revision a 40-hex SHA naming its folder, then the use, then the device."""
    folder = store / "snapshots" / ("0" * 40)
    folder.mkdir(parents=True)
    present = {"repo": "example/asr", "revision": "0" * 40, "snapshot_dir": str(folder)}
    missing = present | {"snapshot_dir": str(store / "none")}
    cases: list[tuple[dict[str, Any], str, str]] = [
        ({"asr": present | {"revision": "1" * 40}, "sv": missing}, "BACKEND_NOT_INSTALLED", "models.sv"),
        ({"asr": present | {"revision": "1" * 40}}, "INVALID_REQUEST", "models.asr.snapshot_dir"),
        ({"asr": present | {"revision": "0" * 39}}, "INVALID_REQUEST", "models.asr.revision"),
        ({"asr": present | {"revision": "0" * 40 + "\n"}}, "INVALID_REQUEST", "models.asr.revision"),
        ({"tts": present}, "INVALID_REQUEST", "models.tts"),
        ({"asr": present, "sv": present | {"revision": "A" * 40}}, "INVALID_REQUEST", "models.sv.revision"),
    ]
    for models, code, field in cases:
        error = _error(lambda models=models: handler.handle("load", {"device": "no-such-device", "models": models}))
        assert (error.code, (error.details or {}).get("field")) == (code, field), models
    named = handler.handle
    error = _error(
        lambda: named("load", {"device": "gpu", "models": {"aligner": _ref("aligner", snapshots["aligner"])}})
    )
    assert (error.code, (error.details or {}).get("field")) == ("INVALID_REQUEST", "device")
    error = _error(lambda: named("load", {"device": "cpu", "models": {"asr": present}}))
    assert error.code == "BACKEND_NOT_INSTALLED" and "config.json" in error.message  # an empty folder


def test_a_load_of_no_model_runs_f0_and_profile_only_s4(handler: QaHandler, store: Path) -> None:
    assert handler.handle("load", {"device": "cpu", "models": {}}) == {"load_s": 0.0, "vram_mb": None}
    requests = _requests(store, _wav(store))
    assert handler.handle("f0", requests["f0"])["f0_hz"]
    assert handler.handle("profile", requests["profile"])["measurements"]
    for op in ("transcribe", "embed", "align"):
        assert _error(lambda op=op: handler.handle(op, requests[op])).code == "NOT_LOADED", op


@pytest.mark.parametrize("device", ["gpu", "cuda0", "CPU", ""])
def test_a_malformed_device_is_an_invalid_request_s4(
    handler: QaHandler, snapshots: dict[str, Path], device: str
) -> None:
    error = _error(lambda: _load(handler, snapshots, "sv", device=device))
    assert (error.code, (error.details or {}).get("field")) == ("INVALID_REQUEST", "device")


@pytest.mark.parametrize(("use", "other"), [("asr", "sv"), ("sv", "aligner"), ("aligner", "asr")])
def test_another_models_snapshot_is_refused_and_changes_nothing_s4(
    handler: QaHandler, snapshots: dict[str, Path], store: Path, use: str, other: str
) -> None:
    _load(handler, snapshots, "aligner")
    wrong = snapshots[other].rename(snapshots[other].with_name(REVISIONS[use]).with_suffix(".wrong"))
    target = wrong.parent / "moved" / REVISIONS[use]
    target.parent.mkdir()
    wrong.rename(target)
    error = _error(lambda: handler.handle("load", {"device": "cpu", "models": {use: _ref(use, target)}}))
    assert error.code == "BACKEND_NOT_INSTALLED" and "install the models again" in error.message
    wav = _wav(store)
    assert handler.handle("align", _requests(store, wav)["align"])["spans"]  # the earlier load still serves


@pytest.mark.skipif(torch.cuda.device_count() > 0, reason="checks a machine without a visible GPU")
def test_a_gpu_load_without_cuda_is_backend_not_installed_s4(handler: QaHandler, snapshots: dict[str, Path]) -> None:
    error = _error(lambda: _load(handler, snapshots, "asr", "sv", device="cuda"))
    assert (error.code, (error.details or {})["field"]) == ("BACKEND_NOT_INSTALLED", "device")


def test_the_aligner_alone_loads_for_a_gpu_device_and_runs_on_the_cpu_s11_2(
    handler: QaHandler, snapshots: dict[str, Path], store: Path
) -> None:
    # The aligner ignores the load's device (section 11.2): a QA load for the GPU that names only the aligner
    # needs no CUDA.
    assert _load(handler, snapshots, "aligner", device="cuda")["vram_mb"] is None  # CUDA never started
    assert handler.handle("align", _requests(store, _wav(store))["align"])["device"] == "cpu"


def test_a_load_that_fails_part_way_leaves_nothing_loaded_s4(
    handler: QaHandler, snapshots: dict[str, Path], store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _load(handler, snapshots, "aligner")

    def broken(self: WavLmSv, *args: Any) -> float:
        raise ValueError("a bug")

    monkeypatch.setattr(WavLmSv, "load", broken)
    with pytest.raises(ValueError, match="a bug"):
        _load(handler, snapshots, "asr", "sv", "aligner")
    wav = _wav(store)
    for op, body in _requests(store, wav).items():
        assert _error(lambda op=op, body=body: handler.handle(op, body)).code == "NOT_LOADED", op


def test_running_out_of_gpu_memory_while_loading_is_gpu_oom_s4(
    handler: QaHandler, snapshots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def out_of_memory(self: WhisperAsr, *args: Any) -> float:
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(WhisperAsr, "load", out_of_memory)
    assert _error(lambda: _load(handler, snapshots, "asr")).code == "GPU_OOM"


def test_load_applies_the_determinism_switches_s10_1(handler: QaHandler, snapshots: dict[str, Path]) -> None:
    torch.use_deterministic_algorithms(False)
    _load(handler, snapshots, "sv")
    assert torch.are_deterministic_algorithms_enabled() and torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    switches = {"tf32": True, "cudnn_deterministic": False, "cudnn_benchmark": False, "deterministic_algorithms": "off"}
    handler.handle("load", {"device": "cpu", "models": {"sv": _ref("sv", snapshots["sv"])}, "determinism": switches})
    assert not torch.are_deterministic_algorithms_enabled() and torch.backends.cuda.matmul.allow_tf32
    error = _error(
        lambda: handler.handle(
            "load", {"device": "cpu", "models": {"sv": _ref("sv", snapshots["sv"])}, "determinism": {}}
        )
    )
    assert (error.code, (error.details or {})["field"]) == ("INVALID_REQUEST", "determinism")


# ---------------------------------------------------------------------- the ops, loaded


def test_every_op_runs_on_the_loaded_models_s4(handler: QaHandler, snapshots: dict[str, Path], store: Path) -> None:
    loaded = _load(handler, snapshots, "asr", "sv", "aligner")
    assert loaded["load_s"] >= 0 and loaded["vram_mb"] is None
    requests = _requests(store, _wav(store))

    transcript = handler.handle("transcribe", requests["transcribe"])
    assert set(transcript) == {"text", "words", "model", "revision"}
    assert (transcript["model"], transcript["revision"]) == ("example/asr", REVISIONS["asr"])

    embedding = handler.handle("embed", requests["embed"])
    assert embedding["dim"] == len(embedding["embedding"]) == 8

    track = handler.handle("f0", requests["f0"])
    assert track["hop_s"] == 0.01 and len(track["f0_hz"]) == len(track["voiced_probability"]) == 201
    voiced = [f for f in track["f0_hz"] if f is not None]
    assert voiced and abs(float(np.median(voiced)) - 140.0) < 2.0
    assert track["method"].startswith("librosa.pyin")

    aligned = handler.handle("align", requests["align"])
    assert [s["token_index"] for s in aligned["spans"]] == list(range(9))

    profile = handler.handle("profile", requests["profile"])
    assert tuple(profile["measurements"]) == PROFILE_KEYS
    assert set(profile["pictures"]) == set(PICTURES)
    out_dir = (store / "scratch" / "profile").resolve()
    for picture in profile["pictures"].values():
        assert Path(picture).parent == out_dir and Path(picture).is_file()
    assert profile["method"]["version"]


def test_an_op_whose_model_the_load_did_not_name_is_not_loaded_s4(
    handler: QaHandler, snapshots: dict[str, Path], store: Path
) -> None:
    _load(handler, snapshots, "aligner")
    requests = _requests(store, _wav(store))
    for op, use in (("transcribe", "asr"), ("embed", "sv")):
        error = _error(lambda op=op: handler.handle(op, requests[op]))
        assert (error.code, (error.details or {})["model"]) == ("NOT_LOADED", use)
        assert f"models.{use}" in error.message
    assert handler.handle("f0", requests["f0"])["f0_hz"]
    assert handler.handle("profile", requests["profile"])["measurements"]


def test_unload_makes_every_model_op_not_loaded_s4(handler: QaHandler, snapshots: dict[str, Path], store: Path) -> None:
    _load(handler, snapshots, "asr", "sv", "aligner")
    assert handler.handle("unload", {}) == {}
    assert handler.handle("unload", {}) == {}
    for op, body in _requests(store, store / "scratch" / "missing.wav").items():
        assert _error(lambda op=op, body=body: handler.handle(op, body)).code == "NOT_LOADED", op


@pytest.mark.parametrize(
    ("op", "change", "field"),
    [
        ("transcribe", {"language": "French"}, "language"),
        ("transcribe", {"language": 3}, "language"),
        ("transcribe", {"word_timestamps": "yes"}, "word_timestamps"),
        ("transcribe", {"long_form": None}, "long_form"),
        ("embed", {"device": "cuda:0"}, "device"),
        ("f0", {"fmin_hz": 20.0}, "fmin_hz"),
        ("f0", {"fmin_hz": "50"}, "fmin_hz"),
        ("f0", {"fmax_hz": 50.0}, "fmax_hz"),
        ("f0", {"fmax_hz": 9000.0}, "fmax_hz"),
        ("align", {"tokens": "RAIN"}, "tokens"),
        ("profile", {"transcript": 5}, "transcript"),
    ],
)
def test_a_bad_member_is_an_invalid_request_before_the_file_is_read_s4(
    handler: QaHandler, snapshots: dict[str, Path], store: Path, op: str, change: dict[str, Any], field: str
) -> None:
    _load(handler, snapshots, "asr", "sv", "aligner")
    body = _requests(store, store / "scratch" / "missing.wav")[op] | change
    error = _error(lambda: handler.handle(op, body))
    assert (error.code, (error.details or {}).get("field")) == ("INVALID_REQUEST", field)


def test_a_file_outside_the_store_or_unreadable_is_refused_s17_2(
    handler: QaHandler, snapshots: dict[str, Path], store: Path, tmp_path: Path
) -> None:
    _load(handler, snapshots, "asr", "sv", "aligner")
    outside = _wav(store).rename(tmp_path / "outside.wav")
    garbage = store / "scratch" / "garbage.wav"
    garbage.write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunk")
    for op in MODEL_OPS:
        for wav, code in (
            (outside, "INVALID_REQUEST"),
            (store / "scratch" / "none.wav", "UNSUPPORTED_AUDIO"),
            (garbage, "UNSUPPORTED_AUDIO"),
        ):
            body = _requests(store, wav)[op]
            assert _error(lambda op=op, body=body: handler.handle(op, body)).code == code, (op, wav.name)
    body = _requests(store, _wav(store))["profile"] | {"out_dir": str(tmp_path / "pictures")}
    assert _error(lambda: handler.handle("profile", body)).code == "INVALID_REQUEST"
    assert not (tmp_path / "pictures").exists()


def test_profile_without_a_transcript_has_no_speaking_rate_s3_6(
    handler: QaHandler, snapshots: dict[str, Path], store: Path
) -> None:
    _load(handler, snapshots, "aligner")
    body = _requests(store, _wav(store))["profile"]
    assert handler.handle("profile", body)["measurements"]["speaking_rate_wpm"] is not None
    for transcript in (None, ""):
        profile = handler.handle("profile", body | {"transcript": transcript})
        assert profile["measurements"]["speaking_rate_wpm"] is None
    del body["transcript"]
    assert handler.handle("profile", body)["measurements"]["speaking_rate_wpm"] is None
