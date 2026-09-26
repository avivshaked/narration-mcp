"""Deterministic synthetic audio and independent reference meters for the post-processing tests.

No audio file is kept in git: every signal is generated here from fixed parameters and a fixed seed.

The reference meters are written independently of ``narration.post`` so the tests can check it:

- ``reference_loudness``: ITU-R BS.1770-4 at 48 kHz from the standard's tabulated K-weighting
  coefficients, with the gating written out here (400 ms blocks, 75 % overlap, whole blocks only);
- ``reference_true_peak``: band-limited (Kaiser-windowed sinc) interpolation at 16x around the largest
  samples, without the pinned resampler.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import soundfile
from scipy.signal import lfilter

Audio = npt.NDArray[np.float64]

# ITU-R BS.1770-4, Tables 1 and 2: the K-weighting filter at 48 kHz.
K_STAGE1_B = (1.53512485958697, -2.69169618940638, 1.19839281085285)
K_STAGE1_A = (1.0, -1.69065929318241, 0.73248077421585)
K_STAGE2_B = (1.0, -2.0, 1.0)
K_STAGE2_A = (1.0, -1.99004745483398, 0.99007225036621)


def speech_like(
    sample_rate: int = 24000,
    *,
    head_s: float = 0.5,
    bursts: tuple[float, ...] = (4.0,),
    gaps: tuple[float, ...] = (),
    tail_s: float = 0.7,
    level: float = 0.05,
    floor: float = 1e-5,
    seed: int = 0,
) -> Audio:
    """A voice-like signal: harmonic bursts (f0 120 Hz, 20 harmonics, a syllable envelope that never falls
    below 30 %) separated by ``gaps`` and framed by ``head_s`` and ``tail_s`` of a noise floor."""
    if len(gaps) != len(bursts) - 1:
        raise ValueError("one gap between each pair of bursts")
    rng = np.random.default_rng(seed)
    parts: list[Audio] = [np.zeros(round(head_s * sample_rate))]
    for i, burst in enumerate(bursts):
        n = round(burst * sample_rate)
        t = np.arange(n, dtype=np.float64) / sample_rate
        voice = sum(np.sin(2.0 * np.pi * 120.0 * k * t + 0.3 * k) / k for k in range(1, 21))
        envelope = 0.3 + 0.7 * np.sin(np.pi * 4.0 * t) ** 2
        parts.append(level * np.asarray(voice, dtype=np.float64) * envelope)
        if i < len(gaps):
            parts.append(np.zeros(round(gaps[i] * sample_rate)))
    parts.append(np.zeros(round(tail_s * sample_rate)))
    x = np.concatenate(parts)
    return x + floor * rng.standard_normal(x.shape[0])


def frames_of(amplitudes: list[float], frame: int) -> Audio:
    """One frame per amplitude, each an alternating +a/-a square wave, so its RMS is exactly ``a``."""
    signs = np.where(np.arange(frame) % 2 == 0, 1.0, -1.0)
    return np.concatenate([a * signs for a in amplitudes]).astype(np.float64)


def sine(freq: float, seconds: float, sample_rate: int, *, peak_dbfs: float, phase: float = 0.0) -> Audio:
    """A sine of the given peak level."""
    t = np.arange(round(seconds * sample_rate), dtype=np.float64) / sample_rate
    return 10.0 ** (peak_dbfs / 20.0) * np.sin(2.0 * np.pi * freq * t + phase)


def write_raw(path: str, x: Audio, sample_rate: int) -> None:
    """Write a raw take as the render layer does: WAV float32 mono."""
    soundfile.write(path, x.astype(np.float32), sample_rate, format="WAV", subtype="FLOAT")


def reference_loudness(x: Audio, sample_rate: int = 48000) -> float:
    """BS.1770-4 integrated loudness of a mono 48 kHz signal, written from the standard."""
    if sample_rate != 48000:
        raise ValueError("the tabulated coefficients are for 48 kHz")
    y = np.asarray(lfilter(K_STAGE2_B, K_STAGE2_A, lfilter(K_STAGE1_B, K_STAGE1_A, x)), dtype=np.float64)
    block, step = 19200, 4800
    z = np.array([np.mean(np.square(y[s : s + block])) for s in range(0, y.shape[0] - block + 1, step)])
    with np.errstate(divide="ignore"):
        blocks = -0.691 + 10.0 * np.log10(z)
    above = z[blocks > -70.0]
    relative = -0.691 + 10.0 * math.log10(float(np.mean(above))) - 10.0
    gated = z[(blocks > -70.0) & (blocks > relative)]
    return -0.691 + 10.0 * math.log10(float(np.mean(gated)))


def reference_true_peak(x: Audio, *, oversample: int = 16, half: int = 256, candidates: int = 64) -> float:
    """The true peak in dBTP, by windowed-sinc interpolation between the samples around the largest ones."""
    magnitude = np.abs(x)
    best = float(magnitude.max())
    order = np.argsort(magnitude)[::-1][:candidates]
    offsets = np.arange(-oversample + 1, oversample, dtype=np.float64) / oversample
    for i in order:
        lo, hi = max(int(i) - half, 0), min(int(i) + half + 1, x.shape[0])
        n = np.arange(lo, hi, dtype=np.float64)
        t = int(i) + offsets
        d = t[:, None] - n[None, :]
        window = np.i0(12.0 * np.sqrt(np.clip(1.0 - (d / (half + 1)) ** 2, 0.0, None))) / np.i0(12.0)
        values = (np.sinc(d) * window) @ x[lo:hi]
        best = max(best, float(np.abs(values).max()))
    return 20.0 * math.log10(best)
