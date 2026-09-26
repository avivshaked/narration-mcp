"""Whisper's decoding, pinned and passed explicitly (design section 11.1 step 2; plan.md WP22).

The pure tests check the settings and the reply's shape. The tiny-model tests run the real pipeline over a tiny,
random Whisper (``conftest.tiny_snapshot``) and look at what reaches ``generate``: every setting that changes
the transcript comes from the worker's own generation config, not from the pipeline's defaults.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from narration_worker_qa import align as qa
from narration_worker_qa import asr


@pytest.mark.parametrize("name", ["English", "english", "ENGLISH", " English ", "en", "EN"])
def test_the_engines_language_name_maps_to_whispers_token_s11_1(name: str) -> None:
    assert asr.whisper_language(name) == "en"


@pytest.mark.parametrize("name", ["French", "fr", "", "Englisch", "en-GB"])
def test_another_language_is_an_invalid_request_s11_1(name: str) -> None:
    with pytest.raises(qa.InvalidRequest) as caught:
        asr.whisper_language(name)
    assert caught.value.details["field"] == "language"


def test_decoding_is_the_evidences_five_beams_unconditioned_without_fallback_s11_1() -> None:
    # Section 11.1 says greedy and conditioned; that loops on two of the bake-off's six takes, so the worker pins
    # the evidence's decoding until the lead decides (status/WP22.md, spikes/acceptance-wp22/decoding.json).
    assert asr.DECODING["num_beams"] == 5 and asr.DECODING["do_sample"] is False
    assert asr.DECODING["temperature"] == 0.0 and asr.DECODING["condition_on_prev_tokens"] is False
    for key in ("compression_ratio_threshold", "logprob_threshold", "no_speech_threshold"):
        assert asr.DECODING[key] is None, key
    long_form = asr.decoding_settings("en", long_form=True)
    assert long_form == {
        "task": "transcribe",
        "temperature": 0.0,
        "condition_on_prev_tokens": False,
        "compression_ratio_threshold": None,
        "logprob_threshold": None,
        "no_speech_threshold": None,
        "language": "en",
        "force_unique_generate_call": False,
    }
    assert asr.decoding_settings("en", long_form=False)["force_unique_generate_call"] is True


def test_words_come_from_the_word_chunks_s11_1() -> None:
    output = {
        "text": " Rain came over the ridge.",
        "chunks": [
            {"text": " Rain", "timestamp": (0.0, 0.42)},
            {"text": " came", "timestamp": (0.42, 0.7000001)},
            {"text": " ", "timestamp": (0.7, 0.7)},
            {"text": " over", "timestamp": (0.7, None)},
        ],
    }
    text, words = asr.words_of(output, word_timestamps=True)
    assert text == "Rain came over the ridge."
    assert words == [
        {"text": "Rain", "start_s": 0.0, "end_s": 0.42, "probability": None},
        {"text": "came", "start_s": 0.42, "end_s": 0.7, "probability": None},
        {"text": "over", "start_s": 0.7, "end_s": None, "probability": None},
    ]


def test_without_word_times_the_words_have_null_times_s11_1() -> None:
    text, words = asr.words_of({"text": " Rain came  over."}, word_timestamps=False)
    assert text == "Rain came  over."
    assert [w["text"] for w in words] == ["Rain", "came", "over."]
    assert all(w["start_s"] is None and w["end_s"] is None for w in words)


# ---------------------------------------------------------------------- a tiny, random Whisper


@pytest.fixture
def torch() -> Any:
    """torch, for the tests that run a model (the pure tests above run without it)."""
    module = pytest.importorskip("torch", reason="needs torch and transformers: the QA worker's full venv")
    pytest.importorskip("transformers", reason="needs torch and transformers: the QA worker's full venv")
    module.set_num_threads(4)
    return module


@pytest.fixture
def tiny_asr(tiny_snapshot: Callable[..., Path], torch: Any) -> asr.WhisperAsr:
    model = asr.WhisperAsr(torch)
    model.load("example/asr", "a" * 40, tiny_snapshot("asr", safetensors=True), "cpu")
    return model


def _spy(model: asr.WhisperAsr) -> list[dict[str, Any]]:
    """Record every ``generate`` call's arguments."""
    calls: list[dict[str, Any]] = []
    pipeline = model._pipeline
    real = pipeline.model.generate

    def generate(*args: Any, **kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return real(*args, **kwargs)

    pipeline.model.generate = generate
    return calls


def _noise(seconds: float, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(int(16_000 * seconds)) * 0.1).astype(np.float32)


@pytest.mark.parametrize(("seconds", "long_form"), [(2.0, False), (2.0, True), (45.0, True)])
def test_every_decoding_setting_reaches_generate_explicitly_s11_1(
    tiny_asr: asr.WhisperAsr, seconds: float, long_form: bool
) -> None:
    calls = _spy(tiny_asr)
    reply = tiny_asr.transcribe(_noise(seconds), "English", word_timestamps=False, long_form=long_form)
    assert (reply["model"], reply["revision"]) == ("example/asr", "a" * 40)
    [call] = calls
    config = call["generation_config"]
    assert config.num_beams == 5 and config.do_sample is False
    assert call["language"] == "en" and call["task"] == "transcribe"
    assert call["temperature"] == 0.0 and call["condition_on_prev_tokens"] is False
    assert call["force_unique_generate_call"] is (not long_form)
    for key in ("compression_ratio_threshold", "logprob_threshold", "no_speech_threshold"):
        assert call[key] is None, key
    assert call["return_timestamps"] is True  # the pipeline's, from return_timestamps=True


def test_the_loaded_generation_config_is_not_changed_by_a_call_s11_1(tiny_asr: asr.WhisperAsr) -> None:
    before = tiny_asr._generation.to_json_string()
    tiny_asr.transcribe(_noise(2.0), "en", word_timestamps=False, long_form=True)
    assert tiny_asr._generation.to_json_string() == before


def test_the_same_audio_decodes_the_same_way_twice_s11_1(tiny_asr: asr.WhisperAsr) -> None:
    audio = _noise(3.0, seed=4)
    first = tiny_asr.transcribe(audio, "English", word_timestamps=False, long_form=True)
    assert tiny_asr.transcribe(audio, "English", word_timestamps=False, long_form=True) == first


def test_more_than_one_window_needs_long_form_s11_1(tiny_asr: asr.WhisperAsr) -> None:
    with pytest.raises(qa.InvalidRequest) as caught:
        tiny_asr.transcribe(_noise(31.0), "English", word_timestamps=True, long_form=False)
    assert caught.value.details["field"] == "long_form"


def test_no_audio_is_an_empty_transcript_s11_1(tiny_asr: asr.WhisperAsr) -> None:
    reply = tiny_asr.transcribe(np.zeros(0, dtype=np.float32), "English", word_timestamps=True, long_form=True)
    assert (reply["text"], reply["words"]) == ("", [])


def test_transcribe_before_load_is_not_loaded_s11_1(torch: Any) -> None:
    with pytest.raises(qa.NotLoaded):
        asr.WhisperAsr(torch).transcribe(_noise(1.0), "English", word_timestamps=True, long_form=True)


def test_unload_releases_the_model_s11_1(tiny_asr: asr.WhisperAsr) -> None:
    tiny_asr.unload()
    assert not tiny_asr.loaded
    with pytest.raises(qa.NotLoaded):
        tiny_asr.transcribe(_noise(1.0), "English", word_timestamps=True, long_form=True)
