"""The voice profile's measurements and pictures (design section 3.6 as changed by DC-1; Appendix A ``f0`` and
``profile``; plan.md WP22).

**No GPL code** (plan.md 1.3 item 3, DC-1): pitch comes from ``librosa.pyin`` (ISC); the harmonics-to-noise ratio
is computed here by Boersma's (1993) autocorrelation method, and the smoothed cepstral peak prominence in the manner
of Hillenbrand and Houde (1996). Each measure is documented below as what it is, with every setting it depends on,
and not as another tool's number: the values are not Praat's, and are not meant to be compared with Praat's. Each
is validated on synthetic signals with known values (``tests/test_voice.py``).

**The analysis rate.** Every measure but loudness runs on the audio mixed to mono and resampled to 16 kHz with
``scipy.signal.resample_poly`` (``align.to_mono_16k``, as the bake-off's ``eval/evaluate.py`` resampled), so a
24 kHz render and its 48 kHz delivery are measured alike. Loudness is measured at the file's own rate, with the
server's delivery meter settings, so a take's profile agrees with its ``loudness`` record (section 13).

**Speech frames.** Where a measure needs to know where the speech is (the voiced span, pauses, the spectral
centroid, CPPS), it uses the delivery trim's speech rule (DC-10): 20 ms frames of the signal with its mean
removed, speech where the frame's RMS is at least max(p95 of the frames' RMS − 40 dB, −70 dBFS).

A worker is a model runner (plan.md P1): these are numbers, and whether a voice suits a script is the caller's
judgement, by ear (section 3.6).
"""

from __future__ import annotations

import importlib.metadata
import math
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from .align import SAMPLE_RATE, to_mono_16k

Signal = npt.NDArray[np.float64]

ANALYSIS_RATE: Final = SAMPLE_RATE
"""16 kHz: the rate every measure but loudness runs at (see the module docstring)."""
METHOD_VERSION: Final = "narration-worker-qa.voice/1"
"""The version of this module's measures and settings; a change to any of them that changes a number needs a new
one. The profile's reply names it in ``method.version``."""

# ---------------------------------------------------------------------- pitch (pyin)
PYIN_FRAME_LENGTH: Final = 1024
"""64 ms at 16 kHz: two periods of a 31.25 Hz pitch, the lowest ``fmin_hz`` the ``f0`` op accepts."""
PYIN_HOP_LENGTH: Final = 160
"""10 ms at 16 kHz: the pitch track's hop (``hop_s``)."""
PYIN_SETTINGS: Final[Mapping[str, Any]] = {
    "n_thresholds": 100,
    "beta_parameters": (2, 18),
    "boltzmann_parameter": 2,
    "resolution": 0.1,
    "max_transition_rate": 35.92,
    "switch_prob": 0.01,
    "no_trough_prob": 0.01,
    "transition_min_prob": 0.0001,
    "center": True,
    "pad_mode": "constant",
}
"""Every other ``librosa.pyin`` setting, passed explicitly (librosa 1.0's values)."""
F0_MIN_HZ: Final = ANALYSIS_RATE / (PYIN_FRAME_LENGTH / 2)
"""The lowest ``fmin_hz``: two of its periods must fit a pyin frame (librosa warns below it)."""
F0_MAX_HZ: Final = ANALYSIS_RATE / 2
"""The highest ``fmax_hz``: the analysis rate's Nyquist frequency."""
PROFILE_FMIN_HZ: Final = 50.0
PROFILE_FMAX_HZ: Final = 400.0
"""The pitch range the profile searches (ASSUME: the range the bake-off measured its clips' median pitch in)."""
MIN_VOICED_FRAMES: Final = 10
"""Fewer voiced pitch frames (0.1 s) than this give no pitch statistics."""

# ---------------------------------------------------------------------- speech, pauses, rate
SPEECH_FRAME: Final = 320
"""20 ms at 16 kHz, as the delivery trim's frames (section 13)."""
SPEECH_REL_DB: Final = -40.0
SPEECH_FLOOR_DBFS: Final = -70.0
SPEECH_PERCENTILE: Final = 95.0
MIN_PAUSE_FRAMES: Final = 8
"""A pause is at least 8 speech frames (0.16 s) without speech inside the voiced span (ASSUME: long enough to
leave out the closure of a stop consonant)."""

# ---------------------------------------------------------------------- loudness (BS.1770-4, as the server's meter)
LOUDNESS_FILTER_CLASS: Final = "DeMan"
LOUDNESS_BLOCK_S: Final = 0.400
LOUDNESS_OVERLAP: Final = 0.75

# ---------------------------------------------------------------------- spectral centroid
CENTROID_N_FFT: Final = 512
"""32 ms Hann frames at 16 kHz, every 10 ms: the magnitude-weighted mean frequency over 0–8 kHz."""
CENTROID_HOP: Final = 160

# ---------------------------------------------------------------------- HNR (Boersma 1993)
HNR_MIN_PITCH_HZ: Final = 75.0
HNR_MAX_PITCH_HZ: Final = 600.0
HNR_PERIODS_PER_WINDOW: Final = 4.5
HNR_TIME_STEP_S: Final = 0.01
HNR_SILENCE_THRESHOLD: Final = 0.1
HNR_VOICING_THRESHOLD: Final = 0.45
HNR_UPSAMPLE: Final = 4
"""The autocorrelation is interpolated band-limitedly to a quarter of a sample (zero-padding its spectrum) before
the peak is refined with a parabola."""

# ---------------------------------------------------------------------- CPPS (Hillenbrand and Houde 1996)
CPPS_WINDOW: Final = 800
"""50 ms Hann frames at 16 kHz (three periods of 60 Hz), zero-padded to ``CPPS_N_FFT``."""
CPPS_N_FFT: Final = 1024
CPPS_HOP: Final = 32
"""2 ms."""
CPPS_PRE_EMPHASIS_HZ: Final = 50.0
CPPS_TIME_SMOOTH: Final = 11
"""Cepstra are averaged over 11 frames (about 20 ms) ..."""
CPPS_QUEFRENCY_SMOOTH: Final = 9
"""... and over 9 quefrency bins (0.5 ms), in dB, before the peak is found."""
CPPS_PITCH_RANGE_HZ: Final = (60.0, 330.0)
"""The peak is searched between the quefrencies of these pitches."""
CPPS_TREND_FROM_S: Final = 0.001
"""The regression line is fitted over quefrencies from 1 ms up to half the FFT (32 ms)."""

PROFILE_KEYS: Final = (
    "duration_s",
    "pitch_median_hz",
    "pitch_p10_hz",
    "pitch_p90_hz",
    "pitch_range_st",
    "speaking_rate_wpm",
    "pause_ratio",
    "loudness_lufs",
    "spectral_centroid_hz",
    "hnr_db",
    "cpps_db",
)
"""The measurements, as the server's ``models.ProfileMeasurements`` names them."""
PICTURES: Final = ("spectrogram", "pitch")


# ====================================================================== pitch


@dataclass(frozen=True, slots=True)
class PitchTrack:
    """A pyin pitch track at the analysis rate: ``f0_hz`` is NaN where the frame is not voiced."""

    hop_s: float
    f0_hz: Signal
    voiced_probability: Signal

    @property
    def times_s(self) -> Signal:
        """Each frame's centre in seconds (pyin's frames are centred)."""
        return np.arange(self.f0_hz.shape[0], dtype=np.float64) * self.hop_s


def pitch_track(audio_16k: npt.NDArray[np.floating[Any]], fmin_hz: float, fmax_hz: float) -> PitchTrack:
    """``librosa.pyin`` over mono 16 kHz audio, every setting explicit (``PYIN_*``).

    ``fmin_hz`` and ``fmax_hz`` must satisfy ``F0_MIN_HZ <= fmin_hz < fmax_hz <= F0_MAX_HZ`` (``ValueError``).
    """
    if not (F0_MIN_HZ <= fmin_hz < fmax_hz <= F0_MAX_HZ):
        raise ValueError(f"need {F0_MIN_HZ} <= fmin_hz < fmax_hz <= {F0_MAX_HZ}, not {fmin_hz} and {fmax_hz}")
    import librosa

    y = np.ascontiguousarray(audio_16k, dtype=np.float32)
    if y.size == 0:
        empty = np.zeros(0, dtype=np.float64)
        return PitchTrack(PYIN_HOP_LENGTH / ANALYSIS_RATE, empty, empty)
    f0, _, probability = librosa.pyin(
        y,
        fmin=float(fmin_hz),
        fmax=float(fmax_hz),
        sr=ANALYSIS_RATE,
        frame_length=PYIN_FRAME_LENGTH,
        hop_length=PYIN_HOP_LENGTH,
        fill_na=np.nan,
        **PYIN_SETTINGS,
    )
    return PitchTrack(
        PYIN_HOP_LENGTH / ANALYSIS_RATE,
        np.asarray(f0, dtype=np.float64),
        np.nan_to_num(np.asarray(probability, dtype=np.float64), nan=0.0),
    )


def pyin_method(fmin_hz: float, fmax_hz: float) -> str:
    """The pitch tracker, its version and settings, for a reply's ``method``."""
    return (
        f"librosa.pyin {_version('librosa')} (mono 16 kHz, frame {PYIN_FRAME_LENGTH}, hop {PYIN_HOP_LENGTH}, "
        f"fmin {fmin_hz:g} Hz, fmax {fmax_hz:g} Hz, other settings librosa 1.0's, passed explicitly)"
    )


@dataclass(frozen=True, slots=True)
class PitchStats:
    median_hz: float
    p10_hz: float
    p90_hz: float
    range_st: float


def pitch_stats(f0_hz: Signal) -> PitchStats | None:
    """Median and 10th/90th percentiles (linear interpolation) of the voiced frames, and the 10–90 range in
    semitones, 12·log2(p90/p10). None with fewer than ``MIN_VOICED_FRAMES`` voiced frames."""
    voiced = f0_hz[np.isfinite(f0_hz)]
    if voiced.size < MIN_VOICED_FRAMES:
        return None
    p10, median, p90 = (float(np.percentile(voiced, q, method="linear")) for q in (10.0, 50.0, 90.0))
    return PitchStats(median, p10, p90, 12.0 * math.log2(p90 / p10))


# ====================================================================== speech frames, pauses, speaking rate


def speech_frames(x: Signal) -> npt.NDArray[np.bool_]:
    """Which 20 ms frames of mono 16 kHz audio are speech, by the delivery trim's rule (module docstring).

    The last frame may be partial; it is measured over the samples it has.
    """
    n = int(x.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.bool_)
    y = x - float(np.mean(x))
    frames = -(-n // SPEECH_FRAME)
    padded = np.zeros(frames * SPEECH_FRAME, dtype=np.float64)
    padded[:n] = y
    counts = np.full(frames, SPEECH_FRAME, dtype=np.float64)
    counts[-1] = n - (frames - 1) * SPEECH_FRAME
    rms = np.sqrt((padded.reshape(frames, SPEECH_FRAME) ** 2).sum(axis=1) / counts)
    level = float(np.percentile(rms, SPEECH_PERCENTILE, method="linear"))
    threshold = max(level * 10.0 ** (SPEECH_REL_DB / 20.0), 10.0 ** (SPEECH_FLOOR_DBFS / 20.0))
    return (rms > 0.0) & (rms >= threshold)


@dataclass(frozen=True, slots=True)
class Speech:
    """Where the speech is: the voiced span (first to last speech frame, in seconds) and the pauses in it."""

    start_s: float | None
    end_s: float | None
    pause_s: float

    @property
    def span_s(self) -> float:
        return 0.0 if self.start_s is None or self.end_s is None else self.end_s - self.start_s

    @property
    def pause_ratio(self) -> float:
        """The share of the voiced span spent in pauses; 1.0 when there is no speech at all."""
        return 1.0 if self.span_s <= 0.0 else self.pause_s / self.span_s


def find_speech(x: Signal) -> Speech:
    """The voiced span of mono 16 kHz audio and the time in pauses (runs of at least ``MIN_PAUSE_FRAMES``
    frames without speech) inside it."""
    speech = speech_frames(x)
    where = np.flatnonzero(speech)
    if where.size == 0:
        return Speech(None, None, 0.0)
    frame_s = SPEECH_FRAME / ANALYSIS_RATE
    first, last = int(where[0]), int(where[-1])
    gaps = np.diff(where) - 1
    pause_frames = int(gaps[gaps >= MIN_PAUSE_FRAMES].sum())
    end = min((last + 1) * SPEECH_FRAME, int(x.shape[0])) / ANALYSIS_RATE
    return Speech(first * frame_s, end, pause_frames * frame_s)


def spoken_words(transcript: str) -> int:
    """Words as the server's text pipeline counts them (``narration.text.words``): whitespace-separated tokens
    with at least one character that is not punctuation (Unicode category P)."""
    import unicodedata

    return sum(1 for token in transcript.split() if any(not unicodedata.category(c).startswith("P") for c in token))


def speaking_rate_wpm(transcript: str | None, speech: Speech) -> float | None:
    """Spoken words per minute over the voiced span (as section 11.1 step 9 measures pace); None without a
    transcript, a word or a voiced span."""
    if not transcript or speech.span_s <= 0.0:
        return None
    words = spoken_words(transcript)
    return words / speech.span_s * 60.0 if words else None


# ====================================================================== loudness


def loudness_lufs(mono: Signal, sample_rate: int) -> float | None:
    """BS.1770-4 integrated loudness of a mono signal at its own rate, with the server's delivery meter settings
    (``narration.post.loudness``): pyloudnorm's ``DeMan`` filters, 400 ms blocks with 75 % overlap, whole blocks
    only, and a signal shorter than one block measured as if silence followed. None when no block passes the
    absolute gate (silence)."""
    import pyloudnorm

    minimum = math.ceil(LOUDNESS_BLOCK_S * sample_rate)
    x = np.asarray(mono, dtype=np.float64)
    if x.shape[0] < minimum:
        x = np.concatenate([x, np.zeros(minimum - x.shape[0], dtype=np.float64)])
    else:
        x = x[: max(_whole_blocks_end(int(x.shape[0]), sample_rate), minimum)]
    meter = pyloudnorm.Meter(
        sample_rate, filter_class=LOUDNESS_FILTER_CLASS, block_size=LOUDNESS_BLOCK_S, overlap=LOUDNESS_OVERLAP
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        value = float(meter.integrated_loudness(x))
    return value if math.isfinite(value) else None


def _whole_blocks_end(samples: int, sample_rate: int) -> int:
    """One past the last sample of the last whole gating block (``narration.post.loudness.whole_blocks_end``)."""
    step = 1.0 - LOUDNESS_OVERLAP
    j = math.floor((samples / sample_rate - LOUDNESS_BLOCK_S) / (LOUDNESS_BLOCK_S * step))
    while j > 0 and int(LOUDNESS_BLOCK_S * (j * step + 1) * sample_rate) > samples:
        j -= 1
    while int(LOUDNESS_BLOCK_S * ((j + 1) * step + 1) * sample_rate) <= samples:
        j += 1
    return int(LOUDNESS_BLOCK_S * (j * step + 1) * sample_rate)


# ====================================================================== short-time spectra


BLOCK_FRAMES: Final = 1024
"""Frames are processed this many at a time, so memory stays bounded on a long take."""


def _frames(x: Signal, length: int, hop: int) -> Signal:
    """A read-only view of frames of ``length`` samples every ``hop``, centred on ``i * hop`` (the signal
    zero-padded at both ends). Slice it in blocks: a copy of every frame of a long take is large."""
    pad = length // 2
    padded = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    return np.lib.stride_tricks.sliding_window_view(padded, length)[::hop]


def _hann(length: int) -> Signal:
    """A Hann window of ``length`` samples without its two zero end points."""
    return np.hanning(length + 2)[1:-1]


def _power_frames(x: Signal) -> Signal:
    """The power spectra of ``_frames(x, CENTROID_N_FFT, CENTROID_HOP)`` under a Hann window, frames × bins."""
    view = _frames(x, CENTROID_N_FFT, CENTROID_HOP)
    window = _hann(CENTROID_N_FFT)
    out = np.empty((view.shape[0], CENTROID_N_FFT // 2 + 1), dtype=np.float64)
    for start in range(0, view.shape[0], BLOCK_FRAMES):
        out[start : start + BLOCK_FRAMES] = np.abs(np.fft.rfft(view[start : start + BLOCK_FRAMES] * window)) ** 2
    return out


def _speech_at(times_s: Signal, speech: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """Whether each time lies in a speech frame."""
    index = np.floor(times_s * ANALYSIS_RATE / SPEECH_FRAME).astype(np.int64)
    inside = (index >= 0) & (index < speech.shape[0])
    out = np.zeros(times_s.shape[0], dtype=np.bool_)
    out[inside] = speech[index[inside]]
    return out


def spectral_centroid_hz(x: Signal, speech: npt.NDArray[np.bool_]) -> float | None:
    """The mean, over frames in speech, of the magnitude-weighted mean frequency of a 32 ms Hann frame (0–8 kHz
    at 16 kHz): the voice's brightness. Audio without a speech frame is measured over every frame with any
    energy (the server's profile needs a number); None only for audio that is all zeros."""
    if x.shape[0] == 0:
        return None
    magnitude = np.sqrt(_power_frames(x))
    freqs = np.fft.rfftfreq(CENTROID_N_FFT, 1.0 / ANALYSIS_RATE)
    total = magnitude.sum(axis=1)
    keep = _speech_at(np.arange(magnitude.shape[0]) * CENTROID_HOP / ANALYSIS_RATE, speech) & (total > 0)
    if not keep.any():
        keep = total > 0
    if not keep.any():
        return None
    centroids = (magnitude[keep] @ freqs) / total[keep]
    return float(np.mean(centroids))


# ====================================================================== HNR (Boersma 1993)


def _autocorrelation(frames: Signal, upsample: int) -> Signal:
    """Each row's autocorrelation for lags 0 .. n−1, at ``1 / upsample`` of a sample, normalised to 1 at lag 0.

    Computed through the power spectrum, zero-padded against wrap-around; its spectrum is zero-padded again, so
    the interpolation between lags is band-limited (exact for a band-limited autocorrelation).
    """
    n = frames.shape[1]
    nfft = 1 << (2 * n - 1).bit_length()
    power = np.abs(np.fft.rfft(frames, nfft, axis=1)) ** 2
    lags = np.fft.irfft(power, nfft * upsample, axis=1)[:, : n * upsample] * upsample
    zero = lags[:, :1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(zero > 0, lags / np.where(zero > 0, zero, 1.0), 0.0)


def hnr_frames(x: Signal) -> tuple[Signal, npt.NDArray[np.bool_]]:
    """Per 10 ms frame: the harmonicity r (the normalised autocorrelation's highest peak between the lags of
    ``HNR_MAX_PITCH_HZ`` and ``HNR_MIN_PITCH_HZ``) and whether the frame counts as voiced.

    Boersma's (1993) method: a Hanning window of ``HNR_PERIODS_PER_WINDOW`` periods of the lowest pitch (60 ms),
    the frame's mean removed, and the windowed frame's autocorrelation divided by the window's own, which undoes
    the window's taper. The peak is found on a band-limited interpolation (``HNR_UPSAMPLE``) and refined with a
    parabola; r is kept below 1. A frame is voiced when r beats the unvoiced candidate's strength in Praat's
    rule, with its pitch defaults (silence threshold 0.1, voicing threshold 0.45): ``voicing + max(0, 2 −
    (local peak / global peak) / (silence / (1 + voicing)))``. Frames whose window would run off the signal
    are not measured.
    """
    window_len = round(HNR_PERIODS_PER_WINDOW / HNR_MIN_PITCH_HZ * ANALYSIS_RATE) | 1
    hop = round(HNR_TIME_STEP_S * ANALYSIS_RATE)
    if x.shape[0] < window_len:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.bool_)
    view = np.lib.stride_tricks.sliding_window_view(x, window_len)[::hop]
    global_peak = float(np.abs(x - x.mean()).max())
    window = _hann(window_len)
    r_window = _autocorrelation(window[None, :], HNR_UPSAMPLE)[0]
    lo = math.ceil(ANALYSIS_RATE / HNR_MAX_PITCH_HZ * HNR_UPSAMPLE)
    hi = math.floor(ANALYSIS_RATE / HNR_MIN_PITCH_HZ * HNR_UPSAMPLE)
    strength = np.zeros(view.shape[0], dtype=np.float64)
    local_peak = np.zeros(view.shape[0], dtype=np.float64)
    has_peak = np.zeros(view.shape[0], dtype=np.bool_)
    for start in range(0, view.shape[0], BLOCK_FRAMES // 4):
        block = np.asarray(view[start : start + BLOCK_FRAMES // 4], dtype=np.float64)
        local = block - block.mean(axis=1, keepdims=True)
        local_peak[start : start + block.shape[0]] = np.abs(local).max(axis=1)
        r = _autocorrelation(local * window, HNR_UPSAMPLE)[:, lo : hi + 2] / r_window[lo : hi + 2]
        inner = r[:, 1:-1]
        peaks = np.where((inner > r[:, :-2]) & (inner >= r[:, 2:]), inner, -np.inf)
        best = np.argmax(peaks, axis=1)
        rows = np.arange(r.shape[0])
        found = np.isfinite(peaks[rows, best])
        left, mid, right = r[rows, best], r[rows, best + 1], r[rows, best + 2]
        denom = left - 2.0 * mid + right
        with np.errstate(invalid="ignore", divide="ignore"):
            offset = np.where(denom < 0, 0.5 * (left - right) / denom, 0.0)
        height = mid - 0.25 * (left - right) * offset
        strength[start : start + block.shape[0]] = np.where(found, np.clip(height, 0.0, 1.0 - 1e-9), 0.0)
        has_peak[start : start + block.shape[0]] = found
    ratio = local_peak / global_peak if global_peak > 0 else np.zeros_like(local_peak)
    unvoiced = HNR_VOICING_THRESHOLD + np.maximum(
        0.0, 2.0 - ratio / (HNR_SILENCE_THRESHOLD / (1.0 + HNR_VOICING_THRESHOLD))
    )
    return strength, has_peak & (strength > unvoiced)


def hnr_db(x: Signal) -> float | None:
    """The harmonics-to-noise ratio in dB: the mean over voiced frames of 10·log10(r / (1 − r)) (``hnr_frames``).
    For a periodic signal in white noise, r is the periodic share of the power, so this is its SNR. None with no
    voiced frame."""
    strength, voiced = hnr_frames(x)
    if not voiced.any():
        return None
    r = strength[voiced]
    return float(np.mean(10.0 * np.log10(r / (1.0 - r))))


# ====================================================================== CPPS (Hillenbrand and Houde 1996)


def cepstral_peaks(x: Signal) -> tuple[Signal, Signal, Signal]:
    """Per 2 ms frame: the smoothed cepstral peak prominence (dB), the peak's quefrency (s), and the frame's time.

    A first-order pre-emphasis from 50 Hz; 50 ms Hann frames zero-padded to 1024 points; the power spectrum in dB;
    the power cepstrum, |FFT of that|², in dB; the cepstra averaged over 11 frames and 9 quefrency bins (a moving
    average that repeats the edge values); the peak between the quefrencies of 330 Hz and 60 Hz (refined with a
    parabola); a least-squares line over quefrencies from 1 ms to 32 ms; the prominence is the peak's height
    above the line at the peak's quefrency.
    """
    from scipy.ndimage import uniform_filter1d

    empty = np.zeros(0, dtype=np.float64)
    if x.shape[0] == 0:
        return empty, empty, empty
    alpha = math.exp(-2.0 * math.pi * CPPS_PRE_EMPHASIS_HZ / ANALYSIS_RATE)
    y = np.concatenate([x[:1], x[1:] - alpha * x[:-1]])
    view = _frames(y, CPPS_WINDOW, CPPS_HOP)
    window = _hann(CPPS_WINDOW)
    times = np.arange(view.shape[0]) * CPPS_HOP / ANALYSIS_RATE
    half = CPPS_N_FFT // 2
    quefrency = np.arange(half) / ANALYSIS_RATE
    lo = math.ceil(ANALYSIS_RATE / CPPS_PITCH_RANGE_HZ[1])
    hi = math.floor(ANALYSIS_RATE / CPPS_PITCH_RANGE_HZ[0])
    trend = slice(math.ceil(CPPS_TREND_FROM_S * ANALYSIS_RATE), half)
    cepstra = np.empty((view.shape[0], half), dtype=np.float32)  # float32: a long take has many 2 ms frames
    for start in range(0, view.shape[0], BLOCK_FRAMES):
        block = view[start : start + BLOCK_FRAMES] * window
        spectrum_db = 10.0 * np.log10(np.abs(np.fft.rfft(block, CPPS_N_FFT, axis=1)) ** 2 + 1e-20)
        cepstrum = np.abs(np.fft.irfft(spectrum_db, CPPS_N_FFT, axis=1)[:, :half]) ** 2
        cepstra[start : start + BLOCK_FRAMES] = 10.0 * np.log10(cepstrum + 1e-20)
    smoothed = uniform_filter1d(cepstra, CPPS_TIME_SMOOTH, axis=0, mode="nearest")
    smoothed = uniform_filter1d(smoothed, CPPS_QUEFRENCY_SMOOTH, axis=1, mode="nearest")
    q_fit = quefrency[trend]
    q_centred = q_fit - q_fit.mean()
    q_spread = float(q_centred @ q_centred)
    prominence = np.empty(view.shape[0], dtype=np.float64)
    peak_q = np.empty(view.shape[0], dtype=np.float64)
    for start in range(0, view.shape[0], BLOCK_FRAMES):
        block = np.asarray(smoothed[start : start + BLOCK_FRAMES], dtype=np.float64)
        fit = block[:, trend]
        slope = (fit @ q_centred) / q_spread  # least squares, in closed form
        intercept = fit.mean(axis=1) - slope * q_fit.mean()
        region = block[:, lo : hi + 1]
        best = np.clip(np.argmax(region, axis=1), 1, region.shape[1] - 2)
        rows = np.arange(region.shape[0])
        left, mid, right = region[rows, best - 1], region[rows, best], region[rows, best + 1]
        denom = left - 2.0 * mid + right
        with np.errstate(invalid="ignore", divide="ignore"):
            offset = np.where(denom < 0, 0.5 * (left - right) / denom, 0.0)
        height = mid - 0.25 * (left - right) * offset
        q = (lo + best + offset) / ANALYSIS_RATE
        prominence[start : start + block.shape[0]] = height - (intercept + slope * q)
        peak_q[start : start + block.shape[0]] = q
    return prominence, peak_q, times


def cpps_db(x: Signal, speech: npt.NDArray[np.bool_]) -> float | None:
    """The smoothed cepstral peak prominence: the mean of ``cepstral_peaks`` over frames in speech (pauses left
    out, so the pause ratio does not leak into voice quality). None with no speech."""
    prominence, _, times = cepstral_peaks(x)
    keep = _speech_at(times, speech)
    return float(np.mean(prominence[keep])) if keep.any() else None


# ====================================================================== the profile


def measure(
    audio: npt.NDArray[np.floating[Any]], sample_rate: int, transcript: str | None
) -> tuple[dict[str, float | None], PitchTrack, Signal]:
    """Every measurement of ``PROFILE_KEYS`` for audio at any rate and channel count; also the pitch track and the
    16 kHz mono signal the pictures are drawn from."""
    data = np.asarray(audio, dtype=np.float64)
    mono = data.mean(axis=1) if data.ndim > 1 else data
    x = np.asarray(to_mono_16k(data, sample_rate), dtype=np.float64)
    speech_mask = speech_frames(x)
    speech = find_speech(x)
    track = pitch_track(x, PROFILE_FMIN_HZ, PROFILE_FMAX_HZ)
    stats = pitch_stats(track.f0_hz)
    loudness = loudness_lufs(mono, sample_rate) if mono.shape[0] else None
    rate = speaking_rate_wpm(transcript, speech)
    centroid = spectral_centroid_hz(x, speech_mask)
    hnr = hnr_db(x)
    cpps = cpps_db(x, speech_mask)
    measurements: dict[str, float | None] = {
        "duration_s": round(mono.shape[0] / sample_rate, 4),
        "pitch_median_hz": _round(stats.median_hz if stats else None, 2),
        "pitch_p10_hz": _round(stats.p10_hz if stats else None, 2),
        "pitch_p90_hz": _round(stats.p90_hz if stats else None, 2),
        "pitch_range_st": _round(stats.range_st if stats else None, 3),
        "speaking_rate_wpm": _round(rate, 2),
        "pause_ratio": round(speech.pause_ratio, 4),
        "loudness_lufs": _round(loudness, 2),
        "spectral_centroid_hz": _round(centroid, 1),
        "hnr_db": _round(hnr, 2),
        "cpps_db": _round(cpps, 2),
    }
    return measurements, track, x


def method() -> dict[str, str]:
    """How each measurement is made, with the libraries' versions: the profile reply's ``method``."""
    return {
        "version": METHOD_VERSION,
        "analysis": "mono mix resampled to 16 kHz (scipy.signal.resample_poly); loudness at the file's own rate",
        "f0": pyin_method(PROFILE_FMIN_HZ, PROFILE_FMAX_HZ),
        "pitch": "voiced pyin frames: median, 10th and 90th percentiles (linear); range 12*log2(p90/p10) semitones",
        "speech": "20 ms frames, mean removed; speech at max(p95 frame RMS - 40 dB, -70 dBFS) (the trim's rule)",
        "speaking_rate": "spoken words of the transcript per minute of the voiced span (first to last speech frame)",
        "pause_ratio": f"time in runs of >= {MIN_PAUSE_FRAMES} non-speech frames / the voiced span; 1.0 without speech",
        "loudness": (
            f"pyloudnorm {_version('pyloudnorm')} BS.1770-4 integrated, filter class {LOUDNESS_FILTER_CLASS}, mono "
            "weight 1.0, whole 400 ms blocks only"
        ),
        "spectral_centroid": "mean over speech frames of the magnitude-weighted mean frequency, 32 ms Hann, 0-8 kHz",
        "hnr": (
            "Boersma (1993) autocorrelation: Hanning window of 4.5 periods of 75 Hz, window-corrected, peak between "
            "600 Hz and 75 Hz; mean dB over frames voiced by Praat's rule (silence 0.1, voicing 0.45)"
        ),
        "cpps": (
            "smoothed cepstral peak prominence (Hillenbrand and Houde 1996): pre-emphasis from 50 Hz, 50 ms Hann "
            "every 2 ms, power cepstrum in dB smoothed over 11 frames and 9 bins, peak 60-330 Hz, least-squares "
            "line from 1 ms; mean over speech frames"
        ),
        "numpy": np.__version__,
    }


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


# ====================================================================== pictures


def draw_pictures(x: Signal, track: PitchTrack, stats: Mapping[str, float | None], out_dir: Path) -> dict[str, str]:
    """The spectrogram and the pitch contour as PNG files in ``out_dir`` (``spectrogram.png``, ``pitch.png``),
    each written to a temporary name and renamed into place; returns their paths by name.

    matplotlib draws with its Agg back end and writes no cache outside ``MPLCONFIGDIR`` (the worker points that
    into the store). The pictures carry no text from the audio: axes, units and the pitch statistics only.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure

    out_dir.mkdir(parents=True, exist_ok=True)
    pictures: dict[str, str] = {}
    duration = x.shape[0] / ANALYSIS_RATE

    figure = Figure(figsize=(10, 4), dpi=100)
    axes = figure.add_subplot()
    if x.shape[0]:
        db = 10.0 * np.log10(_power_frames(x) + 1e-12)
        db = np.maximum(db, db.max() - 80.0)
        image = axes.imshow(
            db.T, origin="lower", aspect="auto", cmap="magma", extent=(0.0, duration, 0.0, ANALYSIS_RATE / 2)
        )
        figure.colorbar(image, ax=axes, label="dB")
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("Frequency (Hz)")
    pictures["spectrogram"] = str(_save(figure, out_dir / "spectrogram.png"))

    figure = Figure(figsize=(10, 3), dpi=100)
    axes = figure.add_subplot()
    axes.plot(track.times_s, track.f0_hz, linewidth=1.2, color="tab:blue")
    for key, style in (("pitch_p10_hz", ":"), ("pitch_median_hz", "-"), ("pitch_p90_hz", ":")):
        value = stats.get(key)
        if value is not None:
            axes.axhline(value, linestyle=style, linewidth=0.8, color="grey")
    axes.set_xlim(0.0, max(duration, 1e-3))
    axes.set_ylim(PROFILE_FMIN_HZ, PROFILE_FMAX_HZ)
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("Pitch (Hz)")
    pictures["pitch"] = str(_save(figure, out_dir / "pitch.png"))
    return pictures


def _save(figure: Any, path: Path) -> Path:
    figure.tight_layout()
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        figure.savefig(str(tmp), format="png")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def track_reply(track: PitchTrack) -> tuple[list[float | None], list[float]]:
    """The ``f0`` reply's lists: Hz (null where unvoiced) and the voiced probability, rounded for the wire."""
    f0: list[float | None] = [None if not math.isfinite(v) else round(float(v), 3) for v in track.f0_hz.tolist()]
    probability = [round(float(v), 6) for v in track.voiced_probability.tolist()]
    return f0, probability


def as_signal(values: Sequence[float]) -> Signal:
    """A float64 array (for tests and callers holding lists)."""
    return np.asarray(values, dtype=np.float64)
