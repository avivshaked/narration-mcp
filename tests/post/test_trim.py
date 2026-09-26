"""The relative trim (design section 13, step 1, with DC-9 and DC-10)."""

from __future__ import annotations

import numpy as np
import pytest

from narration.post.trim import (
    TrimPoints,
    find_trim,
    frame_length,
    frame_rms,
    rule_text,
    speech_frames,
    speech_threshold,
)

from .signals import Audio, frames_of, speech_like

REL, FLOOR, PAD = -40.0, -70.0, 0.08


def _trim(x: Audio, rate: int = 24000) -> TrimPoints:
    return find_trim(x, rate, rel_db=REL, floor_dbfs=FLOOR, pad_s=PAD)


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
    speech = speech_frames(x, 24000, REL, FLOOR)
    assert speech.tolist() == [False] * 8 + [True] * 102 + [False] * 4
    points = _trim(x)
    assert (points.head, points.tail) == (8 * 480, 4 * 480)


def test_trim_threshold_never_falls_below_minus_70_dbfs_s13_dc10() -> None:
    floor = 10.0 ** (-70.0 / 20.0)
    assert speech_threshold(np.array([1e-3] * 20), REL, FLOOR) == floor  # p95 - 40 dB = -100 dBFS
    assert speech_threshold(np.array([1.0] * 20), REL, FLOOR) == 0.01
    x = frames_of([floor * 0.99] * 10 + [floor * 1.01] * 10 + [floor * 0.99] * 10, 480)
    assert speech_frames(x, 24000, REL, FLOOR).tolist() == [False] * 10 + [True] * 10 + [False] * 10


def test_trim_finds_the_silence_at_each_end_s13() -> None:
    points = _trim(speech_like(24000, head_s=0.5, tail_s=0.7))
    assert (points.head, points.tail) == (12000, 16800)
    assert (points.head_s, points.tail_s) == (0.5, 0.7)


@pytest.mark.parametrize("scale", [0.25, 0.5, 2.0, 3.7])
def test_trim_is_gain_independent_above_minus_30_dbfs_s13(scale: float) -> None:
    x = speech_like(24000, head_s=0.34, bursts=(1.5, 0.8), gaps=(0.6,), tail_s=0.52, level=0.2)
    assert _trim(x * scale) == _trim(x)


def test_trim_ignores_a_dc_offset_s13_dc10() -> None:
    # Without the mean removed, a 0.001 offset (-60 dBFS) would make every frame speech.
    x = speech_like(24000, head_s=0.5, tail_s=0.7)
    assert _trim(x + 0.001) == _trim(x)
    assert (_trim(x + 0.001).head, _trim(x - 0.001).tail) == (12000, 16800)


@pytest.mark.parametrize("kind", ["noise", "constant"])
def test_trim_cuts_a_near_silent_tail_under_5_percent_speech_s13_dc10(kind: str) -> None:
    # 1 s of speech, then 25 s at 2e-5: p95 lies in the tail, so only the -70 dBFS floor finds the speech.
    tail = 2e-5 * np.random.default_rng(3).standard_normal(24000 * 25) if kind == "noise" else np.full(24000 * 25, 2e-5)
    x = np.concatenate([speech_like(24000, head_s=0.2, bursts=(1.0,), tail_s=0.0, floor=0.0), tail])
    points = _trim(x)
    assert (points.head, points.tail) == (4800, 25 * 24000)
    assert points.kept == 1920 + 24000 + 1920


def test_speech_frames_leave_the_edges_unmeasured_s11_1() -> None:
    # A frame is measured over its samples outside the edges; one with none left is silence.
    x = frames_of([0.5] * 10, 480)
    x[:480] = 0.9  # a loud first frame, entirely inside the edge
    # The first and last frames each lie entirely inside an edge.
    assert speech_frames(x, 24000, REL, FLOOR, edge=480).tolist() == [False] + [True] * 8 + [False]
    assert speech_frames(x, 24000, REL, FLOOR, edge=240)[0]  # half of it is measured
    assert not speech_frames(x, 24000, REL, FLOOR, edge=2400).any()  # nothing is left to measure
    assert speech_frames(x, 24000, REL, FLOOR, edge=0).tolist() == speech_frames(x, 24000, REL, FLOOR).tolist()


def test_trim_does_not_change_the_audio_s13_dc10() -> None:
    x = speech_like(24000) + 0.001
    before = x.copy()
    _trim(x)
    assert np.array_equal(x, before)


def test_trim_pad_is_80ms_s13() -> None:
    points = _trim(speech_like(24000, head_s=0.5, tail_s=0.7))
    assert points.pad == 1920
    assert (points.start, points.stop) == (12000 - 1920, points.samples - (16800 - 1920))
    assert points.kept == points.samples - (points.head - points.pad) - (points.tail - points.pad)


def test_trim_keeps_all_silence_shorter_than_the_pad_s13_dc9() -> None:
    points = _trim(speech_like(24000, head_s=0.04, tail_s=0.06))
    assert (points.head_s, points.tail_s) == (0.04, 0.06)
    assert (points.start, points.stop) == (0, points.samples)


def test_trim_leaves_silence_untrimmed_s13() -> None:
    for x in (np.zeros(24000), np.full(24000, 1e-5)):
        points = _trim(x)
        assert (points.head, points.tail, points.start, points.stop) == (0, 0, 0, 24000)


def test_trim_frames_of_zero_are_never_speech_s13() -> None:
    x = np.zeros(48000)
    x[24000:24480] = frames_of([0.3], 480)
    points = _trim(x)
    assert (points.head, points.tail) == (24000, 48000 - 24480)


def test_trim_rule_is_stated_as_in_app_b_s13() -> None:
    assert rule_text(-40.0, -70.0) == (
        "max(p95_frame_rms - 40 dB, -70 dBFS), mean removed; head_s/tail_s found, up to pad_s kept"
    )
    assert rule_text(-35.5, -65.0).startswith("max(p95_frame_rms - 35.5 dB, -65 dBFS), mean removed;")
