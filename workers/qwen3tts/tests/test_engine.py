"""The engine's own logic, with a fake qwen-tts model and a fake torch: no model, no GPU.

What is pinned here: every audio-changing setting reaches qwen-tts explicitly, with each call's own
``max_new_tokens`` (section 10.1, DC-4); the token cap is detected from how the talker stopped (section 11.1,
``TOKEN_CAP_HIT``); the voice-prompt encode is the only place deterministic algorithms are suspended; a bad
snapshot is reported with the right code; and the model and voice bookkeeping.
"""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
import soundfile as sf
from narration_qwen3tts.engine import MAX_PREPARED_VOICES, EngineError, QwenEngine, check_snapshot
from narration_qwen3tts.settings import GENERATION_KEYS

EOS = 2150
SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 2000
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


class FakeCuda:
    def __init__(self) -> None:
        self.initialized = True
        self.emptied = 0

    def synchronize(self, device: str | None = None) -> None:
        return None

    def memory_reserved(self, device: str | None = None) -> int:
        return 6 * 1024 * 1024 * 1024

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return self.initialized

    def empty_cache(self) -> None:
        self.emptied += 1


class FakeTorch:
    """The parts of torch the engine touches."""

    bfloat16 = "bfloat16"
    float16 = "float16"
    float32 = "float32"

    def __init__(self) -> None:
        self.deterministic = True
        self.warn_only = True
        self.cuda = FakeCuda()

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self.deterministic

    def is_deterministic_algorithms_warn_only_enabled(self) -> bool:
        return self.warn_only

    def use_deterministic_algorithms(self, mode: bool, *, warn_only: bool = False) -> None:
        self.deterministic = mode
        self.warn_only = warn_only


class FakeTalker:
    def __init__(self, model: FakeInnerModel) -> None:
        self.model = model

    def generate(self, **kwargs: Any) -> Any:
        steps, ends_on_eos = self.model.plan(kwargs["max_new_tokens"])
        tokens = [7] * steps
        if ends_on_eos:
            tokens[-1] = EOS
        return types.SimpleNamespace(sequences=np.array([tokens]))


class FakeInnerModel:
    """``Qwen3TTSForConditionalGeneration``: ``generate`` calls the talker and returns the decoded codes."""

    def __init__(self, tts_model_type: str) -> None:
        self.tts_model_type = tts_model_type
        self.config = types.SimpleNamespace(talker_config=types.SimpleNamespace(codec_eos_token_id=EOS))
        self.talker = FakeTalker(self)
        self.natural_steps = 40
        self.talker_cap_offset = 0  # a qwen-tts that changed its call path would not pass the cap through

    def plan(self, cap: int) -> tuple[int, bool]:
        """The fake speaks ``natural_steps`` tokens, the last one EOS, unless the cap cuts it short."""
        if self.natural_steps <= cap:
            return self.natural_steps, True
        return cap, False

    def generate(self, **kwargs: Any) -> tuple[list[np.ndarray], None]:
        result = self.talker.generate(max_new_tokens=kwargs["max_new_tokens"] + self.talker_cap_offset)
        steps = int(result.sequences.shape[1])
        return [np.zeros((steps - 1, 16), dtype=np.int64)], None


class FakeQwen:
    """``qwen_tts.Qwen3TTSModel``."""

    loads: ClassVar[list[dict[str, Any]]] = []
    fail_with: ClassVar[BaseException | None] = None

    def __init__(self, tts_model_type: str, torch: FakeTorch) -> None:
        self.model = FakeInnerModel(tts_model_type)
        self.torch = torch
        self.calls: list[dict[str, Any]] = []
        self.prompt_deterministic: list[bool] = []

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> FakeQwen:
        cls.loads.append({"path": path, **kwargs})
        if cls.fail_with is not None:
            raise cls.fail_with
        config = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
        return cls(config["tts_model_type"], CURRENT_TORCH[0])

    def get_supported_languages(self) -> list[str]:
        return ["auto", "english", "german"]

    def create_voice_clone_prompt(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.prompt_deterministic.append(self.torch.are_deterministic_algorithms_enabled())
        self.calls.append({"op": "prompt", **kwargs})
        return [{"prompt": kwargs["ref_text"]}]

    def _speak(self, op: str, kwargs: dict[str, Any]) -> tuple[list[np.ndarray], int]:
        self.calls.append({"op": op, **kwargs})
        codes, _ = self.model.generate(max_new_tokens=kwargs["max_new_tokens"])
        return [np.ones(codes[0].shape[0] * SAMPLES_PER_FRAME, dtype=np.float32) * 0.1], SAMPLE_RATE

    def generate_voice_clone(self, **kwargs: Any) -> tuple[list[np.ndarray], int]:
        return self._speak("clone", kwargs)

    def generate_voice_design(self, **kwargs: Any) -> tuple[list[np.ndarray], int]:
        return self._speak("design", kwargs)


CURRENT_TORCH: list[FakeTorch] = []


@pytest.fixture
def torch() -> Iterator[FakeTorch]:
    fake = FakeTorch()
    CURRENT_TORCH[:] = [fake]
    module = types.ModuleType("qwen_tts")
    module.Qwen3TTSModel = FakeQwen  # type: ignore[attr-defined]
    saved = sys.modules.get("qwen_tts")
    sys.modules["qwen_tts"] = module
    FakeQwen.loads = []
    FakeQwen.fail_with = None
    try:
        yield fake
    finally:
        FakeQwen.fail_with = None
        if saved is None:
            sys.modules.pop("qwen_tts", None)
        else:
            sys.modules["qwen_tts"] = saved


def snapshot(tmp_path: Path, tts_model_type: str) -> Path:
    folder = tmp_path / tts_model_type / ("ab" * 20)
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps({"tts_model_type": tts_model_type}), encoding="utf-8")
    return folder


def settings(non_streaming_mode: bool = False, **generation: Any) -> Any:
    return {"non_streaming_mode": non_streaming_mode, "generation": {**GENERATION, **generation}}


def load(engine: QwenEngine, folder: Path, **kwargs: Any) -> Any:
    return engine.load(
        snapshot_dir=folder,
        device="cuda:0",
        dtype="bfloat16",
        attn_implementation="sdpa",
        settings=kwargs.pop("settings", settings()),
    )


def clip(tmp_path: Path) -> Path:
    path = tmp_path / "voice.wav"
    sf.write(str(path), np.zeros((SAMPLE_RATE, 2), dtype=np.float32), SAMPLE_RATE, subtype="PCM_16")
    return path


def wrapper(engine: QwenEngine) -> FakeQwen:
    return engine._loaded.wrapper  # type: ignore[union-attr]


def base_with_voice(tmp_path: Path, torch: FakeTorch, **load_kwargs: Any) -> QwenEngine:
    engine = QwenEngine(torch)
    load(engine, snapshot(tmp_path, "base"), **load_kwargs)
    engine.prepare_voice("v", clip(tmp_path), "Far below.", False)
    return engine


# ---------------------------------------------------------------------------- loading
def test_load_is_offline_from_the_snapshot_folder_s4(tmp_path: Path, torch: FakeTorch) -> None:
    engine = QwenEngine(torch)
    result = load(engine, snapshot(tmp_path, "base"))
    (call,) = FakeQwen.loads
    assert call["local_files_only"] is True
    assert (call["device_map"], call["dtype"], call["attn_implementation"]) == ("cuda:0", "bfloat16", "sdpa")
    assert (result.model_type, result.vram_mb, result.reused) == ("base", 6144, False)
    assert engine.ceiling == CEILING


def test_a_custom_voice_snapshot_is_refused(tmp_path: Path, torch: FakeTorch) -> None:
    with pytest.raises(EngineError) as caught:
        load(QwenEngine(torch), snapshot(tmp_path, "custom_voice"))
    assert (caught.value.code, caught.value.details["field"]) == ("INVALID_REQUEST", "model.snapshot_dir")
    assert FakeQwen.loads == []


def test_another_models_snapshot_is_invalid_request_s14(tmp_path: Path, torch: FakeTorch) -> None:
    """A snapshot of another model (wav2vec2's config.json has no tts_model_type) is a wrong request."""
    folder = tmp_path / ("cd" * 20)
    folder.mkdir()
    config = {"model_type": "wav2vec2", "architectures": ["Wav2Vec2ForCTC"], "vocab_size": 32}
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(EngineError) as caught:
        load(QwenEngine(torch), folder)
    assert (caught.value.code, caught.value.details["field"]) == ("INVALID_REQUEST", "model.snapshot_dir")
    assert FakeQwen.loads == []


@pytest.mark.parametrize("config", [None, "{truncated", "[]"])
def test_a_snapshot_without_a_readable_config_is_backend_not_installed_s14(
    tmp_path: Path, torch: FakeTorch, config: str | None
) -> None:
    """A missing or broken config.json is a broken install: BACKEND_NOT_INSTALLED, with what to do next."""
    folder = tmp_path / ("ef" * 20)
    folder.mkdir()
    if config is not None:
        (folder / "config.json").write_text(config, encoding="utf-8")
    with pytest.raises(EngineError) as caught:
        check_snapshot(folder)
    assert caught.value.code == "BACKEND_NOT_INSTALLED"
    assert "narration-admin install" in caught.value.message
    with pytest.raises(EngineError) as again:
        load(QwenEngine(torch), folder)
    assert again.value.code == "BACKEND_NOT_INSTALLED" and FakeQwen.loads == []


def test_qwen_tts_that_cannot_be_imported_is_backend_not_installed_s14(tmp_path: Path, torch: FakeTorch) -> None:
    sys.modules["qwen_tts"] = None  # type: ignore[assignment]  # makes "import qwen_tts" raise ImportError
    with pytest.raises(EngineError) as caught:
        load(QwenEngine(torch), snapshot(tmp_path, "base"))
    assert caught.value.code == "BACKEND_NOT_INSTALLED"
    assert caught.value.details["module"] == "qwen_tts"


def test_a_snapshot_whose_weights_cannot_be_read_is_backend_not_installed_s14(tmp_path: Path, torch: FakeTorch) -> None:
    FakeQwen.fail_with = OSError("model.safetensors not found")
    engine = QwenEngine(torch)
    emptied = torch.cuda.emptied
    with pytest.raises(EngineError) as caught:
        load(engine, snapshot(tmp_path, "base"))
    assert (caught.value.code, caught.value.details["field"]) == ("BACKEND_NOT_INSTALLED", "model.snapshot_dir")
    assert not engine.loaded
    assert torch.cuda.emptied > emptied  # what the failed load allocated is handed back at once


HELD = "The process cannot access the file because it is being used by another process"
"""Windows' text for a sharing violation, which ``narration_worker.errors.is_transient_load_error`` detects."""


def _held_import(name: str) -> object:
    raise ImportError(f"DLL load failed while importing _C: {HELD}")


@pytest.mark.parametrize("where", ["import", "weights"])
def test_a_file_another_process_holds_is_a_transient_internal_error_s14(
    tmp_path: Path, torch: FakeTorch, where: str
) -> None:
    """As ``narration_worker`` classifies it: a retry gets past it, so it is not ``BACKEND_NOT_INSTALLED``."""
    if where == "import":
        held = types.ModuleType("qwen_tts")
        held.__getattr__ = _held_import  # type: ignore[method-assign]  # PEP 562: "from qwen_tts import X" raises
        sys.modules["qwen_tts"] = held
    else:
        FakeQwen.fail_with = OSError(f"[WinError 32] {HELD}: 'model.safetensors'")
    engine = QwenEngine(torch)
    with pytest.raises(EngineError) as caught:
        load(engine, snapshot(tmp_path, "base"))
    assert caught.value.code == "INTERNAL" and caught.value.details["transient"] is True
    assert "try again" in caught.value.message and not engine.loaded


def test_an_unexpected_load_failure_propagates_and_releases_memory(tmp_path: Path, torch: FakeTorch) -> None:
    FakeQwen.fail_with = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    engine = QwenEngine(torch)
    emptied = torch.cuda.emptied
    with pytest.raises(RuntimeError):  # the worker classifies it (GPU_OOM)
        load(engine, snapshot(tmp_path, "base"))
    assert not engine.loaded and torch.cuda.emptied > emptied


def test_unload_releases_memory_even_with_nothing_loaded(torch: FakeTorch) -> None:
    """After a failed load nothing is resident, but the failed load may have left memory in torch's cache."""
    engine = QwenEngine(torch)
    engine.unload()
    assert torch.cuda.emptied == 1
    torch.cuda.initialized = False  # CUDA never started: nothing to empty, and it must not be started now
    engine.unload()
    assert torch.cuda.emptied == 1


def test_loading_the_same_model_again_keeps_it_takes_the_new_settings_and_clears_voices(
    tmp_path: Path, torch: FakeTorch
) -> None:
    engine = QwenEngine(torch)
    folder = snapshot(tmp_path, "base")
    load(engine, folder)
    engine.prepare_voice("v", clip(tmp_path), "Far below.", False)
    again = load(engine, folder, settings=settings(max_new_tokens=24))
    assert again.reused is True and len(FakeQwen.loads) == 1
    assert engine.ceiling == 24
    assert not engine.prepared("v"), "no state may carry from one load to the next"


def test_a_refused_load_keeps_what_was_loaded(tmp_path: Path, torch: FakeTorch) -> None:
    engine = base_with_voice(tmp_path, torch)
    with pytest.raises(EngineError):
        load(engine, snapshot(tmp_path, "custom_voice"))
    assert engine.model_type == "base" and engine.prepared("v")


def test_loading_another_model_replaces_the_first_and_its_voices_s4(tmp_path: Path, torch: FakeTorch) -> None:
    engine = base_with_voice(tmp_path, torch)
    load(engine, snapshot(tmp_path, "voice_design"))
    assert len(FakeQwen.loads) == 2
    assert engine.model_type == "voice_design" and not engine.prepared("v")


def test_unload_drops_the_model_and_the_voices(tmp_path: Path, torch: FakeTorch) -> None:
    engine = base_with_voice(tmp_path, torch)
    engine.unload()
    engine.unload()
    assert not engine.loaded and not engine.prepared("v") and engine.ceiling is None


# ---------------------------------------------------------------------------- explicit settings and the cap
def test_settings_passed_explicitly_s10_1(tmp_path: Path, torch: FakeTorch) -> None:
    engine = base_with_voice(tmp_path, torch, settings=settings(False, temperature=0.8))
    engine.synthesize("v", "Rain moved across the hills.", "English", 500)
    prompt, speak = wrapper(engine).calls
    assert prompt["x_vector_only_mode"] is False
    assert speak["non_streaming_mode"] is False
    assert {key: speak[key] for key in GENERATION_KEYS} == {**GENERATION, "temperature": 0.8, "max_new_tokens": 500}
    assert (speak["text"], speak["language"], speak["voice_clone_prompt"]) == (
        "Rain moved across the hills.",
        "English",
        [{"prompt": "Far below."}],
    )


def test_design_passes_its_streaming_mode_settings_and_cap_explicitly_s10_1(tmp_path: Path, torch: FakeTorch) -> None:
    engine = QwenEngine(torch)
    load(engine, snapshot(tmp_path, "voice_design"), settings=settings(True))
    rendered = engine.design("A warm, clear voice.", "Good bread asks for patience.", "English", 128)
    (speak,) = wrapper(engine).calls
    assert speak["non_streaming_mode"] is True
    assert (speak["instruct"], speak["text"]) == ("A warm, clear voice.", "Good bread asks for patience.")
    assert {key: speak[key] for key in GENERATION_KEYS} == {**GENERATION, "max_new_tokens": 128}
    assert rendered.max_new_tokens == 128


@pytest.mark.parametrize("cap", [0, 1, CEILING + 1, -5, 2.5, True, None, "128"])
def test_a_call_cap_outside_two_to_the_ceiling_is_refused_not_clamped_s10_1(
    tmp_path: Path, torch: FakeTorch, cap: Any
) -> None:
    engine = base_with_voice(tmp_path, torch)
    with pytest.raises(EngineError) as caught:
        engine.synthesize("v", "Rain.", "English", cap)
    assert (caught.value.code, caught.value.details["field"]) == ("INVALID_REQUEST", "max_new_tokens")
    assert [c["op"] for c in wrapper(engine).calls] == ["prompt"]


def test_the_cap_bounds_are_two_and_the_loaded_ceiling_s10_1(tmp_path: Path, torch: FakeTorch) -> None:
    engine = base_with_voice(tmp_path, torch, settings=settings(max_new_tokens=64))
    assert engine.synthesize("v", "Rain.", "English", 64).hit_token_cap is False
    cut = engine.synthesize("v", "Rain.", "English", 2)
    assert (cut.hit_token_cap, cut.talker_steps, cut.new_tokens, cut.max_new_tokens) == (True, 2, 1, 2)


def test_a_render_that_ends_on_the_end_token_did_not_hit_the_cap_s11_1(tmp_path: Path, torch: FakeTorch) -> None:
    engine = base_with_voice(tmp_path, torch)
    rendered = engine.synthesize("v", "Rain.", "English", CEILING)
    assert rendered.hit_token_cap is False
    assert (rendered.talker_steps, rendered.new_tokens, rendered.max_new_tokens) == (40, 39, CEILING)
    assert rendered.audio.dtype == np.float32 and rendered.audio.ndim == 1
    assert rendered.audio.size == 39 * SAMPLES_PER_FRAME and rendered.sample_rate == SAMPLE_RATE


def test_token_cap_detected_when_generation_stops_without_the_end_token_s11_1(tmp_path: Path, torch: FakeTorch) -> None:
    """new_tokens is the decoded frames: the talker's steps less one, so at most the cap less one."""
    engine = base_with_voice(tmp_path, torch)
    rendered = engine.synthesize("v", "Rain.", "English", 24)
    assert rendered.hit_token_cap is True
    assert (rendered.talker_steps, rendered.new_tokens, rendered.max_new_tokens) == (24, 23, 24)


def test_the_end_token_on_the_last_allowed_step_is_not_a_cap_hit_s11_1(tmp_path: Path, torch: FakeTorch) -> None:
    engine = base_with_voice(tmp_path, torch)
    assert engine.synthesize("v", "Rain.", "English", 40).hit_token_cap is False
    assert engine.synthesize("v", "Rain.", "English", 39).hit_token_cap is True


def test_a_cap_that_does_not_reach_the_talker_is_internal(tmp_path: Path, torch: FakeTorch) -> None:
    """The reply echoes the cap applied; if qwen-tts stopped passing it through, the engine must not claim it."""
    engine = base_with_voice(tmp_path, torch)
    wrapper(engine).model.talker_cap_offset = 1
    with pytest.raises(EngineError) as caught:
        engine.synthesize("v", "Rain.", "English", 100)
    assert caught.value.code == "INTERNAL"


def test_nan_and_infinity_from_the_model_are_kept_s11_1(tmp_path: Path, torch: FakeTorch) -> None:
    """The engine never cleans the raw take: SIGNAL_INVALID (DC-5) is the server's verdict on it."""
    engine = base_with_voice(tmp_path, torch)
    fake = wrapper(engine)
    real_speak = fake._speak

    def poisoned(op: str, kwargs: dict[str, Any]) -> tuple[list[np.ndarray], int]:
        wavs, rate = real_speak(op, kwargs)
        wavs[0][:3] = [np.nan, np.inf, -np.inf]
        return wavs, rate

    fake._speak = poisoned  # type: ignore[method-assign]
    audio = engine.synthesize("v", "Rain.", "English", CEILING).audio
    assert np.isnan(audio[0]) and audio[1] == np.inf and audio[2] == -np.inf


# ---------------------------------------------------------------------------- voices
def test_prepare_voice_suspends_deterministic_algorithms_only_for_the_encode_s10_1(
    tmp_path: Path, torch: FakeTorch
) -> None:
    engine = QwenEngine(torch)
    load(engine, snapshot(tmp_path, "base"))
    torch.use_deterministic_algorithms(True, warn_only=True)
    engine.prepare_voice("v", clip(tmp_path), "Far below.", False)
    assert wrapper(engine).prompt_deterministic == [False]
    assert (torch.deterministic, torch.warn_only) == (True, True)
    torch.use_deterministic_algorithms(False)
    engine.prepare_voice("v", clip(tmp_path), "Far below.", False)
    assert torch.deterministic is False


def test_the_reference_clip_is_passed_as_float32_mono_with_its_rate(tmp_path: Path, torch: FakeTorch) -> None:
    engine = base_with_voice(tmp_path, torch)
    audio, rate = wrapper(engine).calls[0]["ref_audio"]
    assert (audio.dtype, audio.ndim, audio.size, rate) == (np.float32, 1, SAMPLE_RATE, SAMPLE_RATE)


def test_icl_needs_the_reference_transcript_s10_1(tmp_path: Path, torch: FakeTorch) -> None:
    engine = QwenEngine(torch)
    load(engine, snapshot(tmp_path, "base"))
    with pytest.raises(EngineError) as caught:
        engine.prepare_voice("v", clip(tmp_path), "  ", False)
    assert (caught.value.code, caught.value.details["field"]) == ("INVALID_REQUEST", "ref_text")


def test_an_unreadable_clip_is_unsupported_audio(tmp_path: Path, torch: FakeTorch) -> None:
    engine = QwenEngine(torch)
    load(engine, snapshot(tmp_path, "base"))
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not a wav")
    with pytest.raises(EngineError) as caught:
        engine.prepare_voice("v", bad, "Far below.", False)
    assert caught.value.code == "UNSUPPORTED_AUDIO"


def test_synthesize_needs_a_prepared_voice_appA(tmp_path: Path, torch: FakeTorch) -> None:
    engine = QwenEngine(torch)
    load(engine, snapshot(tmp_path, "base"))
    with pytest.raises(EngineError) as caught:
        engine.synthesize("unknown", "Rain.", "English", CEILING)
    assert caught.value.code == "VOICE_NOT_PREPARED"


def test_the_least_recently_used_voice_is_evicted(tmp_path: Path, torch: FakeTorch) -> None:
    """Eviction is why VOICE_NOT_PREPARED means "prepare again and retry"."""
    engine = QwenEngine(torch)
    load(engine, snapshot(tmp_path, "base"))
    path = clip(tmp_path)
    for n in range(MAX_PREPARED_VOICES):
        engine.prepare_voice(f"v{n}", path, "Far below.", False)
    engine.synthesize("v0", "Rain.", "English", CEILING)  # using v0 makes v1 the least recently used
    engine.prepare_voice("new", path, "Far below.", False)
    assert engine.prepared("v0") and engine.prepared("new") and not engine.prepared("v1")
    with pytest.raises(EngineError) as caught:
        engine.synthesize("v1", "Rain.", "English", CEILING)
    assert caught.value.code == "VOICE_NOT_PREPARED"


# ---------------------------------------------------------------------------- which model for which op
def test_ops_need_a_model_appA(torch: FakeTorch, tmp_path: Path) -> None:
    engine = QwenEngine(torch)
    for call in (
        lambda: engine.prepare_voice("v", clip(tmp_path), "x", False),
        lambda: engine.synthesize("v", "Rain.", "English", CEILING),
        lambda: engine.design("A voice.", "Rain.", "English", CEILING),
    ):
        with pytest.raises(EngineError) as caught:
            call()
        assert caught.value.code == "NOT_LOADED"


def test_each_op_needs_its_own_model_s4(tmp_path: Path, torch: FakeTorch) -> None:
    engine = QwenEngine(torch)
    load(engine, snapshot(tmp_path, "base"))
    with pytest.raises(EngineError) as caught:
        engine.design("A voice.", "Rain.", "English", CEILING)
    assert caught.value.code == "INVALID_REQUEST"
    load(engine, snapshot(tmp_path, "voice_design"))
    for call in (
        lambda: engine.prepare_voice("v", clip(tmp_path), "x", False),
        lambda: engine.synthesize("v", "Rain.", "English", CEILING),
    ):
        with pytest.raises(EngineError) as caught:
            call()
        assert caught.value.code == "INVALID_REQUEST"


def test_an_unsupported_language_is_refused_before_generating(tmp_path: Path, torch: FakeTorch) -> None:
    engine = QwenEngine(torch)
    load(engine, snapshot(tmp_path, "voice_design"))
    with pytest.raises(EngineError) as caught:
        engine.design("A voice.", "Rain.", "Klingon", CEILING)
    assert (caught.value.code, caught.value.details["field"]) == ("INVALID_REQUEST", "language")
    assert wrapper(engine).calls == []
