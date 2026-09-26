"""The audio-changing settings are checked and always passed explicitly (design section 10.1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from narration_qwen3tts.settings import (
    GENERATION_KEYS,
    LIBRARY_FALLBACKS,
    SettingsError,
    SnapshotUnreadable,
    effective_generation,
    generation_kwargs,
    parse_generation,
    parse_settings,
    read_model_kind,
)
from narration_worker.handler import MIN_MAX_NEW_TOKENS

PINNED: dict[str, Any] = {
    "do_sample": True,
    "top_k": 50,
    "top_p": 1.0,
    "temperature": 0.9,
    "repetition_penalty": 1.05,
    "subtalker_dosample": True,
    "subtalker_top_k": 50,
    "subtalker_top_p": 1.0,
    "subtalker_temperature": 0.9,
    "max_new_tokens": 8192,
}
"""The pinned snapshots' generation_config.json (plan.md section 1.3 item 1)."""


@pytest.mark.parametrize("key", GENERATION_KEYS)
def test_every_sampling_value_must_be_explicit_s10_1(key: str) -> None:
    generation = {k: v for k, v in PINNED.items() if k != key}
    with pytest.raises(SettingsError) as caught:
        parse_generation(generation)
    assert caught.value.field == f"settings.generation.{key}"


def test_generation_keys_are_the_ten_qwen_tts_merges_s10_1() -> None:
    assert set(GENERATION_KEYS) == set(LIBRARY_FALLBACKS) == set(PINNED)
    assert len(GENERATION_KEYS) == 10


def test_an_unknown_sampling_value_is_refused_s10_1() -> None:
    with pytest.raises(SettingsError) as caught:
        parse_generation({**PINNED, "typical_p": 0.9})
    assert caught.value.field == "settings.generation.typical_p"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("do_sample", 1),
        ("subtalker_dosample", "yes"),
        ("top_k", 0),
        ("top_k", 50.0),
        ("top_k", True),
        ("max_new_tokens", 0),
        ("max_new_tokens", 1),
        ("top_p", 0.0),
        ("top_p", 1.5),
        ("subtalker_top_p", float("nan")),
        ("temperature", 0),
        ("repetition_penalty", -1.0),
        ("subtalker_temperature", "0.9"),
        ("temperature", True),
    ],
)
def test_a_sampling_value_of_the_wrong_type_or_range_is_refused_s10_1(key: str, value: object) -> None:
    with pytest.raises(SettingsError) as caught:
        parse_generation({**PINNED, key: value})
    assert caught.value.field == f"settings.generation.{key}"


def test_valid_generation_is_returned_in_qwen_tts_order_s10_1() -> None:
    reordered = dict(reversed(list(PINNED.items())))
    assert list(parse_generation(reordered)) == list(GENERATION_KEYS)
    assert parse_generation(reordered) == PINNED


def test_non_streaming_mode_must_be_explicit_s10_1() -> None:
    with pytest.raises(SettingsError) as caught:
        parse_settings({"generation": PINNED})
    assert caught.value.field == "settings.non_streaming_mode"
    with pytest.raises(SettingsError) as caught:
        parse_settings({"non_streaming_mode": 0, "generation": PINNED})
    assert caught.value.field == "settings.non_streaming_mode"


def test_settings_need_generation_and_nothing_else_s10_1() -> None:
    with pytest.raises(SettingsError) as caught:
        parse_settings({"non_streaming_mode": False})
    assert caught.value.field == "settings.generation"
    with pytest.raises(SettingsError) as caught:
        parse_settings({"non_streaming_mode": False, "generation": PINNED, "instruct": "calm"})
    assert caught.value.field == "settings.instruct"


@pytest.mark.parametrize("mode", [False, True])
def test_settings_keep_both_streaming_modes_s10_1(mode: bool) -> None:
    """Base uses false and VoiceDesign true; which one a profile pins is the server's decision."""
    settings = parse_settings({"non_streaming_mode": mode, "generation": PINNED})
    assert settings == {"non_streaming_mode": mode, "generation": PINNED}


def test_generation_kwargs_pass_all_ten_values_s10_1() -> None:
    kwargs = generation_kwargs(parse_generation(PINNED))
    assert kwargs == PINNED
    assert all(value is not None for value in kwargs.values())


def _snapshot(tmp_path: Path, generation: object | None, config: object | None = None) -> Path:
    snapshot = tmp_path / ("0" * 40)
    snapshot.mkdir()
    if generation is not None:
        (snapshot / "generation_config.json").write_text(json.dumps(generation), encoding="utf-8")
    if config is not None:
        (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return snapshot


def test_effective_generation_reads_the_snapshot_file_s10_1(tmp_path: Path) -> None:
    """The installed qwen-tts reads generation_config.json, where max_new_tokens is 8192 (plan.md 1.3)."""
    assert effective_generation(_snapshot(tmp_path, PINNED)) == PINNED


def test_effective_generation_falls_back_to_the_library_defaults_s10_1(tmp_path: Path) -> None:
    effective = effective_generation(_snapshot(tmp_path, {"temperature": 0.7}))
    assert effective.get("temperature") == 0.7
    assert effective.get("max_new_tokens") == LIBRARY_FALLBACKS["max_new_tokens"] == 2048
    no_file = tmp_path / "no-generation-config"
    no_file.mkdir()
    assert effective_generation(no_file) == dict(LIBRARY_FALLBACKS)


def test_effective_generation_refuses_a_bad_snapshot_value_s10_1(tmp_path: Path) -> None:
    with pytest.raises(SettingsError):
        effective_generation(_snapshot(tmp_path, {**PINNED, "top_k": "fifty"}))


def test_read_model_kind_reads_tts_model_type(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path, None, {"tts_model_type": "voice_design", "tts_model_size": "1b7"})
    kind = read_model_kind(snapshot)
    assert (kind.tts_model_type, kind.tts_model_size, kind.tokenizer_type) == ("voice_design", "1b7", None)


def test_read_model_kind_refuses_a_folder_that_is_not_qwen3_tts(tmp_path: Path) -> None:
    """Another model's snapshot (here shaped like wav2vec2's config.json) is a wrong request, not a broken
    install: a plain SettingsError, not SnapshotUnreadable."""
    with pytest.raises(SettingsError) as caught:
        read_model_kind(_snapshot(tmp_path, None, {"model_type": "wav2vec2", "architectures": ["Wav2Vec2ForCTC"]}))
    assert caught.value.field == "model.snapshot_dir"
    assert not isinstance(caught.value, SnapshotUnreadable)


def test_read_model_kind_reports_a_missing_or_broken_config_as_unreadable(tmp_path: Path) -> None:
    """A snapshot without a readable config.json object is a broken install (BACKEND_NOT_INSTALLED upstream)."""
    with pytest.raises(SnapshotUnreadable) as caught:
        read_model_kind(tmp_path)
    assert caught.value.field == "model.snapshot_dir"
    broken = tmp_path / "broken"
    broken.mkdir()
    for text in ("{not json", "[1, 2]", ""):
        (broken / "config.json").write_text(text, encoding="utf-8")
        with pytest.raises(SnapshotUnreadable):
            read_model_kind(broken)
    (broken / "config.json").write_bytes(b"\xff\xfe\x00garbage")
    with pytest.raises(SnapshotUnreadable):
        read_model_kind(broken)


def test_the_ceiling_is_at_least_qwen_tts_min_new_tokens_s10_1() -> None:
    """qwen-tts passes min_new_tokens=2 to the talker, so a lower max_new_tokens is refused, not clamped."""
    assert MIN_MAX_NEW_TOKENS == 2  # the protocol's floor for every cap (narration_worker.handler)
    assert parse_generation({**PINNED, "max_new_tokens": 2}).get("max_new_tokens") == 2
    with pytest.raises(SettingsError) as caught:
        parse_generation({**PINNED, "max_new_tokens": 1})
    assert caught.value.field == "settings.generation.max_new_tokens"
