"""The pinned resampler (design section 13, step 2; section 10.2's delivery key)."""

from __future__ import annotations

import importlib.metadata

import numpy as np
import pytest

from narration.post import delivery_tools
from narration.post.resample import KAISER_BETA, ZERO_CROSSINGS, lowpass, ratio, resample, resampled_length

from .signals import sine


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [(24000, 48000, (2, 1)), (22050, 48000, (320, 147)), (16000, 48000, (3, 1)), (96000, 48000, (1, 2))],
)
def test_resampler_ratio_is_reduced_s13(source: int, target: int, expected: tuple[int, int]) -> None:
    assert ratio(source, target) == expected


@pytest.mark.parametrize("source", [24000, 22050, 16000, 44100, 96000])
@pytest.mark.parametrize("samples", [1, 7, 480, 12345])
def test_resampler_length_is_ceil_of_the_ratio_s13(source: int, samples: int) -> None:
    up, down = ratio(source, 48000)
    y = resample(np.ones(samples), source, 48000)
    assert y.shape[0] == resampled_length(samples, source, 48000) == -(-samples * up // down)


def test_resampler_at_the_same_rate_is_a_copy_s13() -> None:
    x = np.linspace(-1.0, 1.0, 101)
    y = resample(x, 48000, 48000)
    assert y is not x
    assert np.array_equal(y, x)


def test_resampler_keeps_a_tone_s13() -> None:
    x = sine(1000.0, 1.0, 24000, peak_dbfs=-6.0)
    y = resample(x, 24000, 48000)
    expected = sine(1000.0, 1.0, 48000, peak_dbfs=-6.0)
    middle = slice(4800, 43200)
    assert np.max(np.abs(y[middle] - expected[middle])) < 1e-4


def test_resampler_rejects_the_image_of_a_high_tone_s13() -> None:
    # An 11 kHz tone at 24 kHz images at 13 kHz once upsampled; the pinned filter keeps it >= 70 dB down.
    y = resample(sine(11000.0, 1.0, 24000, peak_dbfs=0.0), 24000, 48000)
    spectrum = np.abs(np.fft.rfft(y[4800:43200] * np.hanning(38400)))
    freqs = np.fft.rfftfreq(38400, 1 / 48000)
    tone = spectrum[np.argmin(np.abs(freqs - 11000.0))]
    image = spectrum[np.argmin(np.abs(freqs - 13000.0))]
    assert 20 * np.log10(image / tone) < -70.0


def test_resampler_filter_is_pinned_s13() -> None:
    taps = lowpass(2, 1)
    assert taps.shape == (2 * ZERO_CROSSINGS * 2 + 1,)
    assert not taps.flags.writeable
    assert taps.sum() == pytest.approx(1.0, abs=1e-12)


def test_resampler_name_and_version_enter_the_delivery_key_s10_2() -> None:
    tools = delivery_tools()
    assert tools.resampler.startswith(f"scipy.signal.resample_poly {importlib.metadata.version('scipy')} ")
    assert f"kaiser beta={KAISER_BETA}" in tools.resampler
    assert f"{ZERO_CROSSINGS} zero crossings" in tools.resampler
    assert f"numpy {importlib.metadata.version('numpy')}" in tools.resampler
