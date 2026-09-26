"""The signal facts QA needs (design section 11.1 step 1), measured by the delivery processor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from narration.config import DeliveryConfig
from narration.post import DeliveryPipeline, deliver, measure_signal
from narration.post.pcm import decode_wav

from .signals import speech_like, write_raw


def _stats_of(raw: np.ndarray) -> tuple[float, bool, float]:
    stats = measure_signal(raw, 24000, np.zeros(4800), 48000)
    return stats.raw_clipping_fraction, stats.raw_nonfinite, stats.raw_dc_offset


def test_clipping_counts_raw_samples_at_full_scale_s11_1() -> None:
    raw = np.zeros(10000)
    raw[:3] = 1.0
    raw[3:5] = -1.0
    raw[5] = 32767 / 32768
    raw[6] = 0.99
    raw[7] = 1.5
    clipping, nonfinite, _ = _stats_of(raw)
    assert clipping == 7 / 10000
    assert not nonfinite


def test_nonfinite_raw_samples_are_reported_s11_1() -> None:
    raw = np.zeros(1000)
    raw[10], raw[11] = np.nan, np.inf
    clipping, nonfinite, dc = _stats_of(raw)
    assert nonfinite
    assert clipping == 1 / 1000  # the infinity is at full scale; NaN is not
    assert dc == 0.0


def test_dc_offset_is_the_raw_mean_s11_1() -> None:
    raw = 0.01 + 0.1 * np.sin(np.arange(24000) * 2 * np.pi / 240)
    _, _, dc = _stats_of(raw)
    assert dc == pytest.approx(0.01, abs=1e-9)


def test_voiced_span_and_longest_internal_silence_s11_1() -> None:
    raw = speech_like(24000, head_s=0.5, bursts=(1.0, 1.0), gaps=(1.5,), tail_s=0.7)
    made = deliver(raw, 24000, DeliveryConfig())
    delivery, rate = decode_wav(made.wav)
    stats = measure_signal(raw, 24000, delivery, rate)
    assert stats.raw_samples == raw.shape[0]
    assert stats.raw_sample_rate == 24000
    assert stats.delivery_duration_s == made.output.duration_s
    assert stats.voiced_start_s is not None and stats.voiced_end_s is not None
    assert 0.06 <= stats.voiced_start_s <= 0.08
    assert stats.delivery_duration_s - 0.10 <= stats.voiced_end_s <= stats.delivery_duration_s - 0.06
    assert stats.longest_internal_silence_s == pytest.approx(1.5, abs=0.04)


def test_a_dc_offset_does_not_hide_an_internal_silence_s11_1_dc10() -> None:
    # The reviewer's case: DC 0.001 with a 3 s gap. Without the mean removed every frame was speech.
    raw = speech_like(24000, head_s=0.5, bursts=(1.0, 1.0), gaps=(3.0,), tail_s=0.7) + 0.001
    delivery, rate = decode_wav(deliver(raw, 24000, DeliveryConfig()).wav)
    stats = measure_signal(raw, 24000, delivery, rate)
    assert stats.longest_internal_silence_s == pytest.approx(3.0, abs=0.04)
    assert stats.raw_dc_offset == pytest.approx(0.001, abs=1e-5)


def test_a_near_silent_delivery_has_no_voiced_span_s11_1_dc10() -> None:
    # Everything under -70 dBFS is silence, whatever its own p95.
    stats = measure_signal(np.zeros(2400), 24000, 1e-5 * np.sin(np.arange(48000) / 3.0), 48000)
    assert (stats.voiced_start_s, stats.voiced_end_s) == (None, None)


def test_continuous_speech_has_no_internal_silence_s11_1() -> None:
    raw = speech_like(24000)
    delivery, rate = decode_wav(deliver(raw, 24000, DeliveryConfig()).wav)
    assert measure_signal(raw, 24000, delivery, rate).longest_internal_silence_s == 0.0


def test_silent_delivery_has_no_voiced_span_s11_1() -> None:
    stats = measure_signal(np.zeros(2400), 24000, np.zeros(4800), 48000)
    assert (stats.voiced_start_s, stats.voiced_end_s, stats.longest_internal_silence_s) == (None, None, 0.0)
    assert stats.delivery_duration_s == 0.1


def test_signal_stats_reads_the_two_files_s11_1(tmp_path: Path) -> None:
    raw_path, out = tmp_path / "raw.wav", tmp_path / "delivery.wav"
    raw = speech_like(24000, bursts=(1.0, 1.0), gaps=(0.9,))
    write_raw(str(raw_path), raw, 24000)
    pipeline = DeliveryPipeline()
    pipeline.process(raw_path, out, DeliveryConfig())
    delivery, rate = decode_wav(out.read_bytes())
    expected = measure_signal(raw.astype("float32").astype("float64"), 24000, delivery, rate)
    assert pipeline.signal_stats(raw_path, out) == expected
