"""What qwen-tts reads from a Qwen3-TTS snapshot: its sampling defaults and its model kind (section 10.1).

The checks on a ``load``'s ``settings`` are shared with the fake worker (``narration_worker.qwen_settings``:
``parse_settings``, ``SettingsError``, ``DTYPES``, ``ATTN_IMPLEMENTATIONS``), so both refuse the same loads.
This module holds what only the real worker needs:

- ``effective_generation`` computes the sampling values qwen-tts 0.1.1 would use for a snapshot if none were
  passed, from its ``generation_config.json`` and the library's fallbacks: that is what an engine profile
  pins (section 6). The worker itself never relies on them: every value is passed explicitly.
- ``read_model_kind`` reads ``config.json`` to tell Base from VoiceDesign.

Standard library only: the daemon's side can reuse it, and the tests run without torch.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from narration_worker.protocol import Generation
from narration_worker.qwen_settings import GENERATION_KEYS, SettingsError, parse_generation

LIBRARY_FALLBACKS: Final[Mapping[str, bool | int | float]] = {
    "do_sample": True,
    "top_k": 50,
    "top_p": 1.0,
    "temperature": 0.9,
    "repetition_penalty": 1.05,
    "subtalker_dosample": True,
    "subtalker_top_k": 50,
    "subtalker_top_p": 1.0,
    "subtalker_temperature": 0.9,
    "max_new_tokens": 2048,
}
"""qwen-tts 0.1.1's hard-coded fallbacks (``_merge_generate_kwargs``), used only for a key the snapshot's
``generation_config.json`` does not set. The pinned snapshots set all ten, with ``max_new_tokens`` 8192
(plan.md section 1.3 item 1)."""

GENERATION_CONFIG_FILE: Final = "generation_config.json"
"""The file qwen-tts 0.1.1 reads the sampling defaults from. The design calls it ``generate_config.json``
(section 10.1); the installed code reads ``generation_config.json`` (plan.md section 1.3 item 1)."""


class SnapshotUnreadable(SettingsError):
    """A snapshot folder whose ``config.json`` is missing, unreadable or not a JSON object: a broken or
    incomplete install, not a wrong request."""


@dataclass(frozen=True, slots=True)
class ModelKind:
    """What a snapshot holds, from its ``config.json``: ``tts_model_type`` is ``base`` or ``voice_design``."""

    tts_model_type: str
    tokenizer_type: str | None
    tts_model_size: str | None


def effective_generation(snapshot_dir: Path) -> Generation:
    """The sampling values qwen-tts 0.1.1 uses for a snapshot when none is passed: the snapshot's
    ``generation_config.json``, then the library's fallbacks. Raises ``SettingsError`` for a value there
    that ``parse_generation`` would refuse."""
    path = snapshot_dir / GENERATION_CONFIG_FILE
    config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if not isinstance(config, dict):
        raise SettingsError(GENERATION_CONFIG_FILE, "must hold a JSON object")
    config = cast(dict[str, object], config)
    merged = {key: config.get(key, LIBRARY_FALLBACKS[key]) for key in GENERATION_KEYS}
    return parse_generation(merged, GENERATION_CONFIG_FILE)


def read_model_kind(snapshot_dir: Path) -> ModelKind:
    """``tts_model_type`` (``base`` / ``voice_design`` / ``custom_voice``) and friends from ``config.json``.

    Raises ``SnapshotUnreadable`` when ``config.json`` is missing, unreadable or not a JSON object (a broken
    install), and ``SettingsError`` when it has no ``tts_model_type`` (another model's snapshot).
    """
    path = snapshot_dir / "config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotUnreadable("model.snapshot_dir", f"cannot read {path.name}: {exc}") from exc
    if not isinstance(config, dict):
        raise SnapshotUnreadable("model.snapshot_dir", f"{path.name} does not hold a JSON object")
    config = cast(dict[str, object], config)
    if not isinstance(config.get("tts_model_type"), str):
        raise SettingsError("model.snapshot_dir", f"{path.name} has no tts_model_type: not a Qwen3-TTS snapshot")
    return ModelKind(
        tts_model_type=cast(str, config["tts_model_type"]),
        tokenizer_type=_str_or_none(config.get("tokenizer_type")),
        tts_model_size=_str_or_none(config.get("tts_model_size")),
    )


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None
