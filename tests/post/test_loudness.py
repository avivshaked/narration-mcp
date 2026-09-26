"""The loudness meter and the true peak (design section 13, steps 3 and 6)."""

from __future__ import annotations

import importlib.metadata

import numpy as np
import pytest
import scipy

from narration.post import delivery_tools, fade_edges
from narration.post.loudness import FILTER_CLASS, integrated_loudness, true_peak_dbtp, whole_blocks_end
from narration.post.resample import resample

from .signals import reference_loudness, sine, speech_like


def test_meter_reads_the_bs1770_4_reference_sine_s13() -> None:
    # A 997 Hz sine at -20 dBFS peak in one channel of weight 1.0 reads -23.01 LUFS (mean square -3.01 dB,
    # K-weighting +0.69 dB at 997 Hz, offset -0.691).
    level = integrated_loudness(sine(997.0, 5.0, 48000, peak_dbfs=-20.0), 48000)
    assert level == pytest.approx(-23.01, abs=0.02)


@pytest.mark.parametrize("tail_s", [0.4, 0.43, 0.47, 0.5])
def test_meter_agrees_with_the_standard_written_out_s13(tail_s: float) -> None:
    # Lengths that leave a partial final block: the meter counts whole blocks only, as BS.1770-4 does.
    y = resample(speech_like(24000, head_s=0.3, bursts=(2.0, 1.5), gaps=(0.7,), tail_s=tail_s), 24000, 48000)
    level = integrated_loudness(y, 48000)
    assert level is not None
    assert level == pytest.approx(reference_loudness(y), abs=0.001)


def test_meter_ignores_a_loud_partial_final_block_s13() -> None:
    # A burst in the last 50 ms lies in no whole block, so BS.1770-4 does not measure it.
    y = resample(speech_like(24000, head_s=0.0, bursts=(3.0,), tail_s=0.0, floor=0.0), 24000, 48000)
    whole = y[:139200]  # 19200 + 25 steps of 4800: ends exactly where a whole block ends
    ending = np.concatenate([whole, 0.5 * np.ones(2400)])
    assert whole_blocks_end(ending.shape[0], 48000) == 139200
    assert integrated_loudness(ending, 48000) == integrated_loudness(whole, 48000)


@pytest.mark.parametrize(
    ("samples", "rate", "end"),
    [(19200, 48000, 19200), (23999, 48000, 19200), (24000, 48000, 24000), (100000, 48000, 96000),
     (17640, 44100, 17640), (22049, 44100, 17640), (22050, 44100, 22050), (4411, 11025, 4410)],
)  # fmt: skip
def test_meter_whole_blocks_end_where_the_last_whole_block_ends_s13(samples: int, rate: int, end: int) -> None:
    assert whole_blocks_end(samples, rate) == end


def test_meter_gates_out_silence_s13() -> None:
    # Blocks that straddle the end of the speech count; blocks of silence alone fall under the absolute gate.
    speech = resample(speech_like(24000, head_s=0.0, bursts=(3.0,), tail_s=0.0, floor=0.0), 24000, 48000)
    short_pause = integrated_loudness(np.concatenate([speech, np.zeros(48000)]), 48000)
    long_pause = integrated_loudness(np.concatenate([speech, np.zeros(48000 * 10)]), 48000)
    assert short_pause is not None and long_pause is not None
    assert long_pause == pytest.approx(short_pause, abs=1e-9)


def test_meter_has_no_loudness_for_silence_s13() -> None:
    assert integrated_loudness(np.zeros(48000), 48000) is None
    assert integrated_loudness(np.zeros(0), 48000) is None
    assert integrated_loudness(np.full(48000, 1e-6), 48000) is None


def test_meter_measures_less_than_a_block_as_one_block_s13() -> None:
    short = sine(997.0, 0.2, 48000, peak_dbfs=-20.0)
    padded = np.concatenate([short, np.zeros(19200 - short.shape[0])])
    assert integrated_loudness(short, 48000) == integrated_loudness(padded, 48000)


def test_meter_name_and_version_enter_the_delivery_key_s10_2() -> None:
    meter = delivery_tools().loudness_meter
    assert meter.startswith(f"pyloudnorm {importlib.metadata.version('pyloudnorm')} ")  # it has no __version__
    assert f"scipy {scipy.__version__}, numpy {np.__version__}" in meter
    assert f"filter_class={FILTER_CLASS}" in meter
    assert "BS.1770-4" in meter
    assert "mono weight 1.0" in meter
    assert "whole blocks only" in meter
    assert "true peak 4x" in meter


def test_true_peak_finds_the_peak_between_samples_s13() -> None:
    # A 12 kHz sine at 48 kHz with phase pi/4 has every sample at 0.707 of its peak: the sample peak reads
    # -9.03 dBFS while the true peak is -6.02 dBTP. The tone fades in and out: an abrupt edge would ring.
    x = fade_edges(sine(12000.0, 1.0, 48000, peak_dbfs=-6.0206, phase=np.pi / 4), 4800)
    sample_peak = 20 * np.log10(np.max(np.abs(x)))
    peak = true_peak_dbtp(x, 48000)
    assert sample_peak == pytest.approx(-9.03, abs=0.01)
    assert peak == pytest.approx(-6.02, abs=0.05)


def test_true_peak_is_never_below_the_sample_peak_s13() -> None:
    x = np.zeros(4800)
    x[2400] = 0.5
    peak = true_peak_dbtp(x, 48000)
    assert peak is not None
    assert peak >= 20 * np.log10(0.5)


def test_true_peak_of_silence_is_none_s13() -> None:
    assert true_peak_dbtp(np.zeros(4800), 48000) is None
    assert true_peak_dbtp(np.zeros(0), 48000) is None
