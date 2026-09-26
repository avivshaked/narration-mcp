"""The relative trim (design section 13, step 1)."""

from __future__ import annotations

import numpy as np
import pytest

from narration.post.trim import find_trim, frame_length, frame_rms, rule_text, speech_frames

from .signals import frames_of, speech_like


def test_trim_frames_are_20ms_s13() -> None:
    assert frame_length(24000) == 480
    assert frame_length(48000) == 960
    assert frame_length(22050) == 441


def test_trim_measures_a_partial_last_frame_over_its_own_samples_s13() -> None:
    x = np.concatenate([np.full(480, 0.5), np.full(10, 0.25)])
    assert frame_rms(x, 480).tolist() == [0.5, 0.25]


def test_trim_threshold_is_p95_frame_rms_minus_40db_s13() -> None:
    # 100 frames at RMS 1.0 set the speech level (the 95th percentile); the threshold is then 0.01.
    head = [0.005] * 5 + [0.0099] * 3 + [0.0101] * 2
    x = frames_of([*head, *([1.0] * 100), *([0.0099] * 4)], 480)
    speech = speech_frames(x, 24000, -40.0)
    assert speech.tolist() == [False] * 8 + [True] * 102 + [False] * 4
    points = find_trim(x, 24000, rel_db=-40.0, pad_s=0.08)
    assert (points.head, points.tail) == (8 * 480, 4 * 480)


def test_trim_finds_the_silence_at_each_end_s13() -> None:
    points = find_trim(speech_like(24000, head_s=0.5, tail_s=0.7), 24000, rel_db=-40.0, pad_s=0.08)
    assert (points.head, points.tail) == (12000, 16800)
    assert (points.head_s, points.tail_s) == (0.5, 0.7)


@pytest.mark.parametrize("scale", [0.25, 4.0, 0.1, 3.7])
def test_trim_is_gain_independent_s13(scale: float) -> None:
    x = speech_like(24000, head_s=0.34, bursts=(1.5, 0.8), gaps=(0.6,), tail_s=0.52)
    assert find_trim(x * scale, 24000, rel_db=-40.0, pad_s=0.08) == find_trim(x, 24000, rel_db=-40.0, pad_s=0.08)


def test_trim_pad_is_80ms_s13() -> None:
    points = find_trim(speech_like(24000, head_s=0.5, tail_s=0.7), 24000, rel_db=-40.0, pad_s=0.08)
    assert points.pad == 1920
    assert (points.start, points.stop) == (12000 - 1920, points.samples - (16800 - 1920))
    assert points.kept == points.samples - (points.head - points.pad) - (points.tail - points.pad)


def test_trim_keeps_all_silence_shorter_than_the_pad_s13() -> None:
    x = speech_like(24000, head_s=0.04, tail_s=0.06)
    points = find_trim(x, 24000, rel_db=-40.0, pad_s=0.08)
    assert (points.head_s, points.tail_s) == (0.04, 0.06)
    assert (points.start, points.stop) == (0, points.samples)


def test_trim_leaves_digital_silence_untrimmed_s13() -> None:
    points = find_trim(np.zeros(24000), 24000, rel_db=-40.0, pad_s=0.08)
    assert (points.head, points.tail, points.start, points.stop) == (0, 0, 0, 24000)


def test_trim_frames_of_zero_are_never_speech_s13() -> None:
    # Mostly digital zero: the speech level (p95) is 0, yet the zero frames are silence, not speech.
    x = np.zeros(48000)
    x[24000:24480] = frames_of([0.3], 480)
    points = find_trim(x, 24000, rel_db=-40.0, pad_s=0.08)
    assert (points.head, points.tail) == (24000, 48000 - 24480)


def test_trim_rule_is_stated_as_in_app_b_s13() -> None:
    assert rule_text(-40.0) == "p95_frame_rms - 40 dB; head_s/tail_s found, pad_s kept"
    assert rule_text(-35.5) == "p95_frame_rms - 35.5 dB; head_s/tail_s found, pad_s kept"
