"""What the qwen3 worker reads from a snapshot: its sampling defaults and its model kind (design section 10.1).

The checks on a ``load``'s ``settings`` are shared with the fake worker and tested with them
(``narration_worker.qwen_settings``, ``workers/common/tests/test_qwen_settings.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from narration_qwen3tts.settings import LIBRARY_FALLBACKS, SnapshotUnreadable, effective_generation, read_model_kind
from narration_worker.qwen_settings import GENERATION_KEYS, SettingsError

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


def test_the_library_fallbacks_cover_the_ten_values_qwen_tts_merges_s10_1() -> None:
    assert set(GENERATION_KEYS) == set(LIBRARY_FALLBACKS) == set(PINNED)


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
