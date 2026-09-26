"""The ``qwen3`` handler's request handling, in process, with a fake engine and a fake torch: no model, no GPU,
no torch.

Skipped until WP16's worker loop (``narration_worker.handler``) is in this checkout.
"""

from __future__ import annotations

import json
import os
import struct
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("narration_worker.handler", reason="needs WP16's narration_worker.handler (the worker loop)")

import narration_qwen3tts.worker as worker_module
from narration_qwen3tts.engine import EngineError, Rendered
from narration_qwen3tts.worker import Qwen3Handler
from narration_worker.errors import OpError
from narration_worker.handler import WorkerContext
from narration_worker.protocol import AudioReply

REVISION = "ab" * 20
CEILING = 8192
GENERATION: dict[str, Any] = {
    "do_sample": True,
    "top_k": 50,
    "top_p": 1.0,
    "temperature": 0.9,
    "repetition_penalty": 1.05,
    "subtalker_dosample": True,
    "subtalker_top_k": 50,
    "subtalker_top_p": 1.0,
    "subtalker_temperature": 0.9,
    "max_new_tokens": CEILING,
}
DETERMINISM = {"tf32": False, "cudnn_deterministic": True, "cudnn_benchmark": False, "deterministic_algorithms": "off"}


class FakeTorch:
    """What the handler and ``narration_worker.determinism`` touch: the switches, the seeds, CUDA's presence."""

    def __init__(self) -> None:
        self.backends = types.SimpleNamespace(
            cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)),
            cudnn=types.SimpleNamespace(allow_tf32=True, deterministic=False, benchmark=True),
        )
        self.cuda = types.SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda seed: None)
        self.switch_calls: list[tuple[bool, bool]] = []
        self.seed: int | None = None

    def use_deterministic_algorithms(self, mode: bool, *, warn_only: bool = False) -> None:
        self.switch_calls.append((mode, warn_only))

    def manual_seed(self, seed: int) -> None:
        self.seed = seed

    def initial_seed(self) -> int:
        assert self.seed is not None
        return self.seed


class FakeEngine:
    """Stands in for ``QwenEngine``: records calls and the torch seed in force when a render starts."""

    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.loaded = False
        self.ceiling: int | None = None
        self.voices: set[str] = set()
        self.calls: list[tuple[str, Any]] = []
        self.seed_at_render: int | None = None
        self.fail_with: Exception | None = None
        self.audio = np.linspace(-0.5, 0.5, 2400, dtype=np.float32)

    def load(self, **kwargs: Any) -> Any:
        self.calls.append(("load", kwargs))
        if self.fail_with is not None:
            raise self.fail_with
        self.loaded = True
        self.ceiling = kwargs["settings"]["generation"]["max_new_tokens"]
        self.voices.clear()
        return type("R", (), {"load_s": 1.23456, "vram_mb": None, "model_type": "base", "reused": False})()

    def unload(self) -> None:
        self.calls.append(("unload", None))
        self.loaded = False
        self.ceiling = None
        self.voices.clear()

    def prepared(self, voice_hash: str) -> bool:
        return voice_hash in self.voices

    def prepare_voice(self, voice_hash: str, ref_wav: Path, ref_text: str, x_vector_only_mode: bool) -> None:
        self.calls.append(("prepare_voice", (voice_hash, ref_wav, ref_text, x_vector_only_mode)))
        self.voices.add(voice_hash)

    def _render(self, op: str, args: Any, cap: int) -> Rendered:
        self.calls.append((op, args))
        self.seed_at_render = int(self.torch.initial_seed())
        if self.fail_with is not None:
            raise self.fail_with
        return Rendered(
            audio=self.audio, sample_rate=24000, new_tokens=24, hit_token_cap=False, talker_steps=25, gen_s=0.5,
            max_new_tokens=cap,
        )  # fmt: skip

    def synthesize(self, voice_hash: str, text: str, language: str, max_new_tokens: int) -> Rendered:
        return self._render("synthesize", (voice_hash, text, language, max_new_tokens), max_new_tokens)

    def design(self, description: str, text: str, language: str, max_new_tokens: int) -> Rendered:
        return self._render("design", (description, text, language, max_new_tokens), max_new_tokens)


@pytest.fixture
def store(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    (root / "scratch").mkdir(parents=True)
    return root


@pytest.fixture
def torch() -> FakeTorch:
    return FakeTorch()


@pytest.fixture
def handler(store: Path, torch: FakeTorch, monkeypatch: pytest.MonkeyPatch) -> Qwen3Handler:
    monkeypatch.setattr(worker_module, "QwenEngine", FakeEngine)
    handler = Qwen3Handler(WorkerContext(role="qwen3", store_root=store, cpu_threads=2))
    handler._torch = torch  # what self.torch() returns from now on: no torch import
    return handler


def snapshot(store: Path, name: str = REVISION, config: dict[str, Any] | None = None) -> Path:
    folder = store.parent / "models" / name
    folder.mkdir(parents=True, exist_ok=True)
    body = {"tts_model_type": "base"} if config is None else config
    (folder / "config.json").write_text(json.dumps(body), encoding="utf-8")
    return folder


def load_request(folder: Path, **overrides: Any) -> dict[str, Any]:
    request: dict[str, Any] = {
        "id": 1,
        "op": "load",
        "device": "cpu",
        "model": {"repo": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "revision": REVISION, "snapshot_dir": str(folder)},
        "engine_profile_id": "qwen3-base-1.7b.p1",
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "determinism": dict(DETERMINISM),
        "settings": {"non_streaming_mode": False, "generation": dict(GENERATION)},
    }
    request.update(overrides)
    return {k: v for k, v in request.items() if v is not None}


def error_of(call: Callable[[], object]) -> OpError:
    with pytest.raises(OpError) as caught:
        call()
    return caught.value


def fake_engine(handler: Qwen3Handler) -> FakeEngine:
    engine = handler._engine
    assert isinstance(engine, FakeEngine)
    return engine


# ---------------------------------------------------------------------------- load
def test_the_worker_forces_offline_loading_s17_7(handler: Qwen3Handler) -> None:
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_the_fingerprint_names_the_libraries_that_shape_the_audio_s6() -> None:
    for name in ("qwen-tts", "transformers", "accelerate", "tokenizers", "torch", "librosa", "soxr", "soundfile"):
        assert name in Qwen3Handler.fingerprint_packages


def test_load_checks_the_snapshot_before_anything_else_s14(handler: Qwen3Handler, store: Path) -> None:
    request = load_request(store / "missing", device="tpu", determinism=None, settings=None)
    assert error_of(lambda: handler.op_load(request)).code == "BACKEND_NOT_INSTALLED"


def test_load_needs_an_absolute_snapshot_dir_appA(handler: Qwen3Handler, store: Path) -> None:
    request = load_request(Path("models") / REVISION)
    error = error_of(lambda: handler.op_load(request))
    assert (error.code, (error.details or {})["field"]) == ("INVALID_REQUEST", "model.snapshot_dir")


def test_load_needs_a_snapshot_folder_named_by_its_revision_s4(handler: Qwen3Handler, store: Path) -> None:
    error = error_of(lambda: handler.op_load(load_request(snapshot(store, "main"))))
    assert (error.code, (error.details or {})["field"]) == ("INVALID_REQUEST", "model.snapshot_dir")
    bad_revision = load_request(snapshot(store, "main"))
    bad_revision["model"]["revision"] = "main"
    error = error_of(lambda: handler.op_load(bad_revision))
    assert (error.details or {})["field"] == "model.revision"


def test_another_models_snapshot_is_invalid_request_and_switches_nothing_s14(
    handler: Qwen3Handler, store: Path, torch: FakeTorch
) -> None:
    """A wav2vec2-shaped snapshot is refused before the determinism switches are touched."""
    folder = snapshot(store, config={"model_type": "wav2vec2", "architectures": ["Wav2Vec2ForCTC"]})
    error = error_of(lambda: handler.op_load(load_request(folder)))
    assert (error.code, (error.details or {})["field"]) == ("INVALID_REQUEST", "model.snapshot_dir")
    assert torch.switch_calls == [] and torch.backends.cudnn.benchmark is True
    assert handler._engine is None


def test_a_snapshot_without_a_readable_config_is_backend_not_installed_s14(
    handler: Qwen3Handler, store: Path, torch: FakeTorch
) -> None:
    folder = snapshot(store)
    (folder / "config.json").unlink()
    error = error_of(lambda: handler.op_load(load_request(folder)))
    assert error.code == "BACKEND_NOT_INSTALLED" and "narration-admin install" in error.message
    assert torch.switch_calls == []


def test_load_needs_the_determinism_switches_s10_1(handler: Qwen3Handler, store: Path) -> None:
    error = error_of(lambda: handler.op_load(load_request(snapshot(store), determinism=None)))
    assert (error.code, (error.details or {})["field"]) == ("INVALID_REQUEST", "determinism")


@pytest.mark.parametrize("key", ["max_new_tokens", "subtalker_temperature"])
def test_load_needs_every_audio_changing_setting_s10_1(handler: Qwen3Handler, store: Path, key: str) -> None:
    generation = {k: v for k, v in GENERATION.items() if k != key}
    request = load_request(snapshot(store), settings={"non_streaming_mode": False, "generation": generation})
    error = error_of(lambda: handler.op_load(request))
    assert (error.code, (error.details or {})["field"]) == ("INVALID_REQUEST", f"settings.generation.{key}")
    error = error_of(lambda: handler.op_load(load_request(snapshot(store), settings=None)))
    assert (error.details or {})["field"] == "settings"


def test_a_ceiling_below_two_is_refused_s10_1(handler: Qwen3Handler, store: Path) -> None:
    settings = {"non_streaming_mode": False, "generation": {**GENERATION, "max_new_tokens": 1}}
    error = error_of(lambda: handler.op_load(load_request(snapshot(store), settings=settings)))
    assert (error.code, (error.details or {})["field"]) == ("INVALID_REQUEST", "settings.generation.max_new_tokens")


@pytest.mark.parametrize(("field", "value"), [("device", "tpu"), ("dtype", "int8"), ("attn_implementation", "flash")])
def test_load_refuses_unknown_devices_dtypes_and_attention(
    handler: Qwen3Handler, store: Path, field: str, value: str
) -> None:
    error = error_of(lambda: handler.op_load(load_request(snapshot(store), **{field: value})))
    assert (error.code, (error.details or {})["field"]) == ("INVALID_REQUEST", field)


def test_load_on_cuda_without_a_gpu_is_backend_not_installed(
    handler: Qwen3Handler, store: Path, torch: FakeTorch
) -> None:
    error = error_of(lambda: handler.op_load(load_request(snapshot(store), device="cuda:0")))
    assert error.code == "BACKEND_NOT_INSTALLED"
    assert torch.switch_calls == []


def test_load_applies_the_switches_then_loads_and_replies_load_facts_appA(
    handler: Qwen3Handler, store: Path, torch: FakeTorch
) -> None:
    folder = snapshot(store)
    reply = handler.op_load(load_request(folder))
    assert reply == {"load_s": 1.235, "vram_mb": None}
    assert torch.switch_calls == [(False, False)]
    assert torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False
    (_, kwargs), *_ = fake_engine(handler).calls
    assert kwargs == {
        "snapshot_dir": folder,
        "device": "cpu",
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "settings": {"non_streaming_mode": False, "generation": GENERATION},
    }


def test_an_engine_load_error_keeps_its_code(handler: Qwen3Handler, store: Path) -> None:
    handler.op_load(load_request(snapshot(store)))
    fake_engine(handler).fail_with = EngineError("BACKEND_NOT_INSTALLED", "the snapshot cannot be loaded")
    assert error_of(lambda: handler.op_load(load_request(snapshot(store)))).code == "BACKEND_NOT_INSTALLED"


def test_running_out_of_gpu_memory_while_loading_is_gpu_oom_s4(handler: Qwen3Handler, store: Path) -> None:
    handler.op_load(load_request(snapshot(store)))
    fake_engine(handler).fail_with = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    assert error_of(lambda: handler.op_load(load_request(snapshot(store)))).code == "GPU_OOM"


# ---------------------------------------------------------------------------- synthesize and design
def prepared(handler: Qwen3Handler, store: Path) -> None:
    handler.op_load(load_request(snapshot(store)))
    clip = store / "scratch" / "voices" / "v.wav"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"RIFF")  # the fake engine never reads it
    prepare = {"id": 2, "op": "prepare_voice", "voice_hash": "sha256:v", "ref_wav": str(clip)}
    handler.op_prepare_voice(prepare | {"ref_text": "Far below.", "x_vector_only_mode": False})


def synthesize_request(store: Path, **overrides: Any) -> dict[str, Any]:
    request: dict[str, Any] = {
        "id": 3,
        "op": "synthesize",
        "voice_hash": "sha256:v",
        "engine_text": "Rain moved across the hills.",
        "language": "English",
        "seed": 1834112093,
        "max_new_tokens": 128,
        "out_path": str(store / "scratch" / "job" / "p00_a0.wav"),
    }
    request.update(overrides)
    return {k: v for k, v in request.items() if v is not None}


def design_request(store: Path, **overrides: Any) -> dict[str, Any]:
    request: dict[str, Any] = {
        "id": 4,
        "op": "design",
        "description": "A warm, clear adult voice.",
        "design_text": "Good bread asks for patience.",
        "language": "English",
        "seed": 271828,
        "max_new_tokens": 128,
        "out_path": str(store / "scratch" / "design" / "c0.wav"),
    }
    request.update(overrides)
    return {k: v for k, v in request.items() if v is not None}


def test_synthesize_seeds_then_writes_a_float_wav_and_the_generation_facts_s10_3(
    handler: Qwen3Handler, store: Path
) -> None:
    prepared(handler, store)
    reply = handler.op_synthesize(synthesize_request(store))
    assert reply == {
        "sample_rate": 24000,
        "samples": 2400,
        "gen_s": 0.5,
        "hit_token_cap": False,
        "new_tokens": 24,
        "max_new_tokens": 128,
    }
    engine = fake_engine(handler)
    assert engine.seed_at_render == 1834112093
    assert engine.calls[-1] == ("synthesize", ("sha256:v", "Rain moved across the hills.", "English", 128))
    out = store / "scratch" / "job" / "p00_a0.wav"
    data = out.read_bytes()
    audio_format, channels, rate = struct.unpack_from("<HHI", data, 20)
    assert (audio_format, channels, rate) == (3, 1, 24000)  # WAVE_FORMAT_IEEE_FLOAT, mono
    assert np.array_equal(np.frombuffer(data[-2400 * 4 :], dtype="<f4"), engine.audio)
    assert [p.name for p in out.parent.iterdir()] == ["p00_a0.wav"]


@pytest.mark.parametrize("op", ["synthesize", "design"])
def test_the_audio_reply_has_exactly_the_protocols_members_appA(handler: Qwen3Handler, store: Path, op: str) -> None:
    """Every member of ``protocol.AudioReply`` but the loop's own (``id``, ``ok``, ``error``), and nothing else."""
    if op == "synthesize":
        prepared(handler, store)
        reply = handler.op_synthesize(synthesize_request(store))
    else:
        handler.op_load(load_request(snapshot(store)))  # the fake engine designs with any model
        reply = handler.op_design(design_request(store))
    members = (AudioReply.__required_keys__ | AudioReply.__optional_keys__) - {"id", "ok", "error"}
    assert set(reply) == members


@pytest.mark.parametrize("op", ["synthesize", "design"])
@pytest.mark.parametrize("cap", [None, 0, 1, CEILING + 1, 2.5, True, "128"])
def test_a_call_without_a_valid_cap_is_invalid_request_before_voice_not_prepared_s10_1(
    handler: Qwen3Handler, store: Path, op: str, cap: Any
) -> None:
    """The cap is required (never a default), from 2 to the loaded ceiling, refused rather than clamped, and
    checked with the other arguments, before the voice is looked up."""
    handler.op_load(load_request(snapshot(store)))
    if op == "synthesize":
        call = lambda: handler.op_synthesize(synthesize_request(store, voice_hash="sha256:nobody", max_new_tokens=cap))  # noqa: E731
    else:
        call = lambda: handler.op_design(design_request(store, max_new_tokens=cap))  # noqa: E731
    error = error_of(call)
    assert (error.code, (error.details or {})["field"]) == ("INVALID_REQUEST", "max_new_tokens")
    assert [name for name, _ in fake_engine(handler).calls] == ["load"]


def test_the_cap_may_be_anything_from_two_to_the_loaded_ceiling_s10_1(handler: Qwen3Handler, store: Path) -> None:
    prepared(handler, store)
    for cap in (2, CEILING):
        assert handler.op_synthesize(synthesize_request(store, max_new_tokens=cap))["max_new_tokens"] == cap
    low = {"non_streaming_mode": False, "generation": {**GENERATION, "max_new_tokens": 300}}
    handler.op_load(load_request(snapshot(store), settings=low))
    error = error_of(lambda: handler.op_design(design_request(store, max_new_tokens=301)))
    assert (error.details or {})["ceiling"] == 300


def test_synthesize_for_an_unprepared_voice_appA(handler: Qwen3Handler, store: Path) -> None:
    handler.op_load(load_request(snapshot(store)))
    error = error_of(lambda: handler.op_synthesize(synthesize_request(store, voice_hash="sha256:nobody")))
    assert error.code == "VOICE_NOT_PREPARED" and "retry" in error.message


def test_every_load_clears_the_prepared_voices(handler: Qwen3Handler, store: Path) -> None:
    prepared(handler, store)
    handler.op_load(load_request(snapshot(store)))
    assert error_of(lambda: handler.op_synthesize(synthesize_request(store))).code == "VOICE_NOT_PREPARED"


def test_nan_and_infinity_reach_the_raw_file_unchanged_s11_1(handler: Qwen3Handler, store: Path) -> None:
    """raw.wav keeps what the model generated: SIGNAL_INVALID (DC-5) is the server's verdict on the raw take."""
    prepared(handler, store)
    engine = fake_engine(handler)
    engine.audio = np.array([np.nan, np.inf, -np.inf, 0.25], dtype=np.float32)
    handler.op_synthesize(synthesize_request(store))
    samples = np.frombuffer((store / "scratch" / "job" / "p00_a0.wav").read_bytes()[-16:], dtype="<f4")
    assert np.isnan(samples[0]) and samples[1] == np.inf and samples[2] == -np.inf and samples[3] == 0.25


def test_design_seeds_and_writes_the_candidate_s10_3(handler: Qwen3Handler, store: Path) -> None:
    handler.op_load(load_request(snapshot(store)))
    reply = handler.op_design(design_request(store))
    engine = fake_engine(handler)
    assert engine.seed_at_render == 271828 and reply["max_new_tokens"] == 128
    assert reply["samples"] == 2400 and (store / "scratch" / "design" / "c0.wav").is_file()


def test_a_render_failure_is_render_failed(handler: Qwen3Handler, store: Path) -> None:
    handler.op_load(load_request(snapshot(store)))
    fake_engine(handler).fail_with = ValueError("something inside generate")
    error = error_of(lambda: handler.op_design(design_request(store)))
    assert error.code == "RENDER_FAILED" and (error.details or {})["type"] == "builtins.ValueError"
    assert not (store / "scratch" / "design" / "c0.wav").exists()


def test_running_out_of_gpu_memory_is_gpu_oom_s4(handler: Qwen3Handler, store: Path) -> None:
    handler.op_load(load_request(snapshot(store)))
    fake_engine(handler).fail_with = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    assert error_of(lambda: handler.op_design(design_request(store))).code == "GPU_OOM"


def test_an_engine_error_keeps_its_code(handler: Qwen3Handler, store: Path) -> None:
    handler.op_load(load_request(snapshot(store)))
    fake_engine(handler).fail_with = EngineError(
        "INVALID_REQUEST", "language 'Klingon' is not supported", {"field": "language"}
    )
    error = error_of(lambda: handler.op_design(design_request(store)))
    assert (error.code, error.details) == ("INVALID_REQUEST", {"field": "language"})


def test_an_output_outside_the_store_is_refused_appA(handler: Qwen3Handler, store: Path) -> None:
    handler.op_load(load_request(snapshot(store)))
    request = design_request(store, out_path=str(store.parent / "elsewhere.wav"))
    assert error_of(lambda: handler.op_design(request)).code == "INVALID_REQUEST"


def test_unload_and_shutdown_drop_the_model(handler: Qwen3Handler, store: Path) -> None:
    handler.op_load(load_request(snapshot(store)))
    assert handler.op_unload({"id": 5, "op": "unload"}) == {}
    assert error_of(lambda: handler.op_design(design_request(store))).code == "NOT_LOADED"
    handler.shutdown()
