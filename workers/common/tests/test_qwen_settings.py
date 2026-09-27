"""A Qwen load's audio-changing settings are checked and always passed explicitly (design section 10.1).

``narration_worker.qwen_settings`` is what both the ``qwen3`` worker and the fake check a Qwen ``load`` with,
so these tests pin the refusals both give: the code is the workers' ``INVALID_REQUEST``, and the member each
``SettingsError`` names becomes its ``details.field``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from narration_worker.handler import MIN_MAX_NEW_TOKENS
from narration_worker.qwen_settings import (
    CEILING_FIELD,
    GENERATION_KEYS,
    SettingsError,
    ceiling_of,
    generation_kwargs,
    parse_generation,
    parse_settings,
)

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


def refused_field(parse: Callable[[object], object], value: object) -> str:
    """The member ``parse`` names when it refuses ``value``."""
    with pytest.raises(SettingsError) as caught:
        parse(value)
    return caught.value.field


@pytest.mark.parametrize("key", GENERATION_KEYS)
def test_every_sampling_value_must_be_explicit_s10_1(key: str) -> None:
    generation = {k: v for k, v in PINNED.items() if k != key}
    assert refused_field(parse_generation, generation) == f"settings.generation.{key}"


def test_generation_keys_are_the_ten_qwen_tts_merges_s10_1() -> None:
    assert set(GENERATION_KEYS) == set(PINNED)
    assert len(GENERATION_KEYS) == 10


def test_a_missing_value_is_named_in_qwen_tts_order_s10_1() -> None:
    """An empty ``generation`` names ``do_sample``, the first key qwen-tts merges, not the ceiling."""
    assert refused_field(parse_generation, {}) == "settings.generation.do_sample"
    assert refused_field(parse_generation, {"max_new_tokens": 8192}) == "settings.generation.do_sample"


def test_an_unknown_sampling_value_is_refused_s10_1() -> None:
    assert refused_field(parse_generation, {**PINNED, "typical_p": 0.9}) == "settings.generation.typical_p"


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
        ("temperature", float("inf")),
        ("repetition_penalty", -1.0),
        ("subtalker_temperature", "0.9"),
        ("temperature", True),
    ],
)
def test_a_sampling_value_of_the_wrong_type_or_range_is_refused_s10_1(key: str, value: object) -> None:
    assert refused_field(parse_generation, {**PINNED, key: value}) == f"settings.generation.{key}"


def test_valid_generation_is_returned_in_qwen_tts_order_s10_1() -> None:
    reordered = dict(reversed(list(PINNED.items())))
    assert list(parse_generation(reordered)) == list(GENERATION_KEYS)
    assert parse_generation(reordered) == PINNED


def test_generation_must_be_an_object_s10_1() -> None:
    assert refused_field(parse_generation, "every value") == "settings.generation"
    assert refused_field(parse_generation, None) == "settings.generation"


def test_non_streaming_mode_must_be_explicit_s10_1() -> None:
    assert refused_field(parse_settings, {"generation": PINNED}) == "settings.non_streaming_mode"
    assert refused_field(parse_settings, {"non_streaming_mode": 0, "generation": PINNED}) == (
        "settings.non_streaming_mode"
    )
    assert refused_field(parse_settings, {}) == "settings.non_streaming_mode"


def test_settings_need_generation_and_nothing_else_s10_1() -> None:
    assert refused_field(parse_settings, {"non_streaming_mode": False}) == "settings.generation"
    unknown = {"non_streaming_mode": False, "generation": PINNED, "instruct": "calm"}
    assert refused_field(parse_settings, unknown) == "settings.instruct"
    assert refused_field(parse_settings, {"instruct": "calm"}) == "settings.instruct"  # before anything missing


@pytest.mark.parametrize("value", ["8192", [], None, 8192])
def test_settings_must_be_an_object_s10_1(value: object) -> None:
    assert refused_field(parse_settings, value) == "settings"


@pytest.mark.parametrize("mode", [False, True])
def test_settings_keep_both_streaming_modes_s10_1(mode: bool) -> None:
    """Base uses false and VoiceDesign true; which one a profile pins is the server's decision."""
    settings = parse_settings({"non_streaming_mode": mode, "generation": PINNED})
    assert settings == {"non_streaming_mode": mode, "generation": PINNED}


def test_generation_kwargs_pass_all_ten_values_s10_1() -> None:
    kwargs = generation_kwargs(parse_generation(PINNED))
    assert kwargs == PINNED
    assert all(value is not None for value in kwargs.values())


def test_the_ceiling_is_at_least_qwen_tts_min_new_tokens_s10_1() -> None:
    """qwen-tts passes min_new_tokens=2 to the talker, so a lower max_new_tokens is refused, not clamped."""
    assert MIN_MAX_NEW_TOKENS == 2  # the protocol's floor for every cap (narration_worker.handler)
    assert ceiling_of(parse_generation({**PINNED, "max_new_tokens": 2})) == 2
    settings = {"non_streaming_mode": False, "generation": {**PINNED, "max_new_tokens": 1}}
    assert refused_field(parse_settings, settings) == CEILING_FIELD == "settings.generation.max_new_tokens"
