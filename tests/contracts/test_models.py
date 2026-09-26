"""The sidecar records read and write the shapes of design Appendix B."""

from __future__ import annotations

import dataclasses

import pytest

from narration.contracts import models
from narration.contracts.serial import ContractError, from_json, to_json

RENDER_JSON = {
    "schema": "narration.render/v1",
    "render_id": "rn_77e0c4a1b2d93f08",
    "render_key": "sha256:" + "7" * 64,
    "voice": {"voice_hash": "sha256:" + "3" * 64, "clip_sha256": "5" * 64, "x_vector_only_mode": False},
    "engine": {
        "engine_profile_id": "qwen3-base-1.7b.p1",
        "engine_profile_hash": "sha256:" + "9" * 64,
        "model_repo": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "model_revision": "f" * 40,
        "non_streaming_mode": False,
        "generation": {"do_sample": True, "max_new_tokens": 8192},
        "observed": {"gpu": "a GPU", "driver": "1", "cuda": "12.8", "cudnn": "9"},
    },
    "engine_text": "Before dawn, the reef belongs to the Oss-a-veen shrimp.",
    "seed": 1834112093,
    "seed_scheme": "narration-seed/v1",
    "attempt": 0,
    "raw": {"path": "raw.wav", "sha256": "a" * 64, "sample_rate": 24000, "samples": 222240, "format": "WAV FLOAT mono"},
    "hit_token_cap": False,
    "gen_s": 21.3,
    "rtf": 2.30,
    "canary": {"batch_status": "hash_match"},
    "licence": {
        "generation_model": "Apache-2.0",
        "voice_clip": "synthetic (Qwen3-TTS VoiceDesign, Apache-2.0)",
    },
}

TAKE_JSON = {
    "schema": "narration.take/v1",
    "take_id": "tk_8c41d2e07a9b3f55",
    "delivery_key": "sha256:" + "8" * 64,
    "render_id": "rn_77e0c4a1b2d93f08",
    "delivery": {
        "path": "delivery.wav",
        "sha256": "b" * 64,
        "sample_rate": 48000,
        "samples": 420480,
        "duration_s": 8.76,
        "format": "WAV PCM_24 mono",
    },
    "trim": {"head_s": 0.27, "tail_s": 0.39, "pad_s": 0.08, "rule": "p95_frame_rms - 40 dB"},
    "loudness": {
        "target_lufs": -20.0,
        "measured_lufs": -20.0,
        "gain_db": 1.4,
        "true_peak_dbtp": -6.3,
        "ceiling_applied": False,
    },
    "tools": {
        "resampler": "scipy.signal.resample_poly 1.18",
        "loudness_meter": "pyloudnorm 0.2.0",
        "post": "narration.post/1",
    },
    "post_stretched": False,
    "flags": [],
}


@pytest.mark.parametrize(("cls", "data"), [(models.RenderRecord, RENDER_JSON), (models.TakeRecord, TAKE_JSON)])
def test_sidecars_round_trip_app_b(cls: type, data: dict[str, object]) -> None:
    record = from_json(cls, data)
    assert to_json(record) == data


def test_unknown_keys_are_refused() -> None:
    with pytest.raises(ContractError, match="unknown key"):
        from_json(models.TakeRecord, {**TAKE_JSON, "surprise": 1})


def test_missing_required_keys_are_refused() -> None:
    data = dict(TAKE_JSON)
    del data["trim"]
    with pytest.raises(ContractError, match="missing required key 'trim'"):
        from_json(models.TakeRecord, data)


def test_literal_values_are_checked() -> None:
    bad = {**RENDER_JSON, "canary": {"batch_status": "maybe"}}
    with pytest.raises(ContractError):
        from_json(models.RenderRecord, bad)


def test_unplaced_cue_keeps_null_times_s11_2() -> None:
    cue = models.CueTiming(index=1, start_s=None, end_s=None, confidence=None)
    assert to_json(cue) == {"index": 1, "start_s": None, "end_s": None, "confidence": None, "words": []}
    assert from_json(models.CueTiming, to_json(cue)) == cue


def test_analysis_alignment_block_of_app_b_loads() -> None:
    # App. B's analysis sidecar names the aligner in versions.aligner_method, not in the alignment block.
    block = {
        "method": "ctc-forced-align+silence-snap",
        "device": "cpu",
        "cross_check": {"model": "openai/whisper-large-v3", "max_disagreement_s": 0.06},
        "measured_error": {"p50_s": 0.02, "p95_s": 0.05, "n": 30, "benchmark": "alignment-en.v1"},
        "cues": [{"index": 0, "start_s": 0.08, "end_s": 3.02, "confidence": 0.93, "words": []}],
        "flags": [],
    }
    alignment = from_json(models.Alignment, block)
    assert alignment.model is None and alignment.revision is None
    assert to_json(alignment) == block
    named = to_json(dataclasses.replace(alignment, model="m", revision="r"))
    assert (named["model"], named["revision"]) == ("m", "r")


def test_exact_words_are_a_half_open_range_app_b() -> None:
    ew = from_json(models.ExactWords, {"start": 17, "end": 43, "words": [3, 7]})
    assert ew.words == (3, 7)


def test_error_record_carries_retry_after_dc2() -> None:
    err = models.Error(code="QUEUE_FULL", message="full", retryable=True, retry_after_s=12.5)
    assert to_json(err)["retry_after_s"] == 12.5
