"""The delivery pipeline (design section 13): raw take → ``delivery.wav``, on the CPU.

In this order:

1. **Trim** (``narration.post.trim``): the silence at each end, found with frame RMS measured on the take
   with its mean removed against max(speech level (p95 of 20 ms frame RMS) - 40 dB, -70 dBFS); at most
   ``pad_s`` (0.08 s) of it is kept at each end, and none is added (DC-9, DC-10).
2. **Resample** to ``sample_rate`` (48 kHz) with the pinned resampler (``narration.post.resample``).
3. **Gain**: one static gain to ``target_lufs`` (-23 LUFS integrated by default, BS.1770-4,
   ``narration.post.loudness``) measured on the resampled signal. No limiter. The gain is rounded to
   0.0001 dB before it is applied, so the gain the record states is exactly the gain used.
4. **Fades**: ``fade_s`` (0.01 s) raised-cosine ramps at both edges; the first and last samples are zero.
   A raised cosine rather than a linear ramp, because many takes start or end with little silence (DC-9).
5. **Quantise** to ``subtype`` (PCM_24) mono, and write the WAV bytes (``narration.post.pcm``).
6. **True peak** of this final file, 4x oversampled. Above ``true_peak_dbtp`` (-1.0 dBTP), the gain is
   lowered to what the ceiling allows and the signal re-quantised, then measured again (and lowered by
   0.0001 dB steps while quantisation keeps it above); the take is flagged ``LOUDNESS_UNDER_TARGET`` (info)
   with the shortfall. **The ceiling wins over the loudness target.**

The loudness record's ``measured_lufs`` and ``true_peak_dbtp`` are measured on the final file. A gain above
+12 dB is flagged ``GAIN_HIGH`` (info).

**CPU thread cap (section 4.1).** No step calls BLAS or starts a thread: the resampler, the meter's filters
and numpy's element-wise operations do their work on the calling thread (a test measures the CPU time of
every thread while ``deliver`` runs). The native pools a library starts at import (OpenBLAS's) are capped by
the environment of the process that runs this (``OMP_NUM_THREADS`` and the like), set when it starts.

**Determinism.** On one machine, with the same pinned versions, the same raw audio and profile give the same
bytes, in every run and every process (section 10.1 promises no more). Across machines, libm and SIMD
differences of a few ULPs in the measurements can in principle change a gain's last rounded digit and so
the bytes; rounding the gain to 0.0001 dB makes that rare, not impossible. Any change to a rule here that
changes bytes needs a new ``names.POST_RULES``, which is in the delivery key.

**Degenerate takes are delivered, not refused**, so that QA can fail them and trigger a retake: non-finite
raw samples are treated as zero (``SignalStats.raw_nonfinite`` reports them), and a take with no
measurable loudness (every 400 ms block under the -70 LUFS absolute gate, e.g. silence) gets no gain; its
record states ``measured_lufs`` null, and ``true_peak_dbtp`` null when the file is all zero.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from narration.config import DeliveryConfig
from narration.contracts import codes
from narration.contracts.interfaces import DeliveryOutput, SignalStats
from narration.contracts.models import DeliveryTools, Flag, Loudness, Trim
from narration.contracts.names import POST_RULES

from . import loudness, resample
from .pcm import bits_of, encode_wav, quantise, read_mono, to_float, write_atomic
from .trim import find_trim, frame_length, rule_text, speech_frames

GAIN_HIGH_DB: Final = 12.0
"""A gain above this is flagged ``GAIN_HIGH`` (section 13)."""
DECIMALS: Final = 4
"""Gain, loudness and true peak are stated (and the gain applied) to 0.0001 dB."""
GAIN_STEP_DB: Final = 10.0**-DECIMALS
MAX_CEILING_STEPS: Final = 1000
FULL_SCALE: Final = 32767 / 32768
"""A raw sample at or beyond this magnitude counts as clipped (the largest positive 16-bit value)."""
VOICED_REL_DB: Final = -40.0
VOICED_FLOOR_DBFS: Final = -70.0
"""The speech rule ``SignalStats`` uses on the delivery file: the trim's rule at its defaults (DC-10)."""
DEFAULT_FADE_S: Final = DeliveryConfig().fade_s
"""The delivery's fades at their default: ``SignalStats`` leaves this much of each end unmeasured."""

Audio = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True, kw_only=True)
class Delivered:
    """A delivery made in memory: the WAV file's bytes and what the processor reports about it."""

    wav: bytes
    output: DeliveryOutput


@functools.cache
def delivery_tools() -> DeliveryTools:
    """The resampler's and meter's names, versions and pinned parameters, and the rules' version (10.2)."""
    return DeliveryTools(resampler=resample.describe(), loudness_meter=loudness.describe(), post=POST_RULES)


def fade_edges(y: Audio, samples: int) -> Audio:
    """Raised-cosine fades of ``samples`` at both edges; the first and last samples become zero."""
    out = np.array(y, dtype=np.float64, copy=True)
    k = min(samples, int(out.shape[0]))
    if k <= 0:
        return out
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(k, dtype=np.float64) / samples)
    out[:k] *= ramp
    out[out.shape[0] - k :] *= ramp[::-1]
    return out


def _linear(gain_db: float) -> float:
    return 10.0 ** (gain_db / 20.0)


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, DECIMALS)


def _flag(code: str, message: str, details: dict[str, Any]) -> Flag:
    severity = codes.flag_code(code).severities[0]
    return Flag(
        code=code,
        severity=severity,
        message=message,
        retake_trigger=codes.is_retake_trigger(code, severity),
        details=details,
    )


def _under_target(record: Loudness, target_gain: float | None, profile: DeliveryConfig) -> Flag:
    reduction = round((target_gain if target_gain is not None else 0.0) - record.gain_db, DECIMALS)
    details: dict[str, Any] = {
        "target_lufs": profile.target_lufs,
        "measured_lufs": record.measured_lufs,
        "gain_reduction_db": reduction,
        "true_peak_dbtp": record.true_peak_dbtp,
        "ceiling_dbtp": profile.true_peak_dbtp,
    }
    message = f"The true-peak ceiling of {profile.true_peak_dbtp:g} dBTP lowered the gain by {reduction:.2f} dB"
    if record.measured_lufs is not None:
        shortfall = round(profile.target_lufs - record.measured_lufs, DECIMALS)
        details["shortfall_db"] = shortfall
        message += (
            f", so this take sits at {record.measured_lufs:.2f} LUFS, {shortfall:.2f} dB under the "
            f"{profile.target_lufs:g} LUFS target"
        )
    return _flag(
        codes.LOUDNESS_UNDER_TARGET,
        message + ". To level takes of one script, use each take's loudness record.",
        details,
    )


def deliver(raw: Audio, sample_rate: int, profile: DeliveryConfig) -> Delivered:
    """Post-process one raw take held in memory (section 13); the file's bytes are returned, not written.

    ``raw`` is mono, full scale 1.0, at ``sample_rate``.
    """
    bits = bits_of(profile.subtype)
    rate = profile.sample_rate
    ceiling = profile.true_peak_dbtp
    x = np.nan_to_num(np.asarray(raw, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    # 1. trim
    points = find_trim(
        x, sample_rate, rel_db=profile.trim_rel_db, floor_dbfs=profile.trim_floor_dbfs, pad_s=profile.trim_pad_s
    )
    # 2. resample
    y = resample.resample(x[points.start : points.stop], sample_rate, rate)
    # 3. gain, measured before the fades as the design orders the steps
    measured_in = loudness.integrated_loudness(y, rate)
    target_gain = None if measured_in is None else round(profile.target_lufs - measured_in, DECIMALS)
    gain_db = 0.0 if target_gain is None else target_gain
    # 4. fades
    faded = fade_edges(y, round(profile.fade_s * rate))
    # 5. quantise, 6. true peak on the final file
    q = quantise(faded * _linear(gain_db), bits)
    peak = loudness.true_peak_dbtp(to_float(q, bits), rate)
    ceiling_applied = False
    if peak is not None and peak > ceiling:
        ceiling_applied = True
        unity = loudness.true_peak_dbtp(faded, rate)
        allowed = gain_db if unity is None else math.floor((ceiling - unity) / GAIN_STEP_DB) * GAIN_STEP_DB
        gain_db = round(min(gain_db - GAIN_STEP_DB, allowed), DECIMALS)
        for _ in range(MAX_CEILING_STEPS):
            q = quantise(faded * _linear(gain_db), bits)
            peak = loudness.true_peak_dbtp(to_float(q, bits), rate)
            if peak is None or peak <= ceiling:
                break
            gain_db = round(gain_db - GAIN_STEP_DB, DECIMALS)
        else:  # pragma: no cover - quantisation moves a peak by far less than the steps allow
            raise RuntimeError(f"the true-peak ceiling of {ceiling} dBTP was not met after {MAX_CEILING_STEPS} steps")

    record = Loudness(
        measured_lufs=_rounded(loudness.integrated_loudness(to_float(q, bits), rate)),
        gain_db=gain_db,
        true_peak_dbtp=_rounded(peak),
        ceiling_applied=ceiling_applied,
        target_lufs=profile.target_lufs,
    )
    flags: list[Flag] = []
    if ceiling_applied:
        flags.append(_under_target(record, target_gain, profile))
    if gain_db > GAIN_HIGH_DB:
        flags.append(
            _flag(
                codes.GAIN_HIGH,
                f"The take needed {gain_db:+.2f} dB of gain to reach {profile.target_lufs:g} LUFS, more than "
                f"+{GAIN_HIGH_DB:g} dB: the raw render was unusually quiet, so listen for raised noise.",
                {"gain_db": gain_db, "limit_db": GAIN_HIGH_DB},
            )
        )

    samples = int(q.shape[0])
    output = DeliveryOutput(
        samples=samples,
        sample_rate=rate,
        duration_s=samples / rate,
        trim=Trim(
            head_s=points.head_s,
            tail_s=points.tail_s,
            pad_s=profile.trim_pad_s,
            rule=rule_text(profile.trim_rel_db, profile.trim_floor_dbfs),
        ),
        loudness=record,
        flags=tuple(flags),
    )
    return Delivered(wav=encode_wav(q, rate, profile.subtype), output=output)


def measure_signal(
    raw: Audio, raw_rate: int, delivery: Audio, delivery_rate: int, *, fade_s: float = DEFAULT_FADE_S
) -> SignalStats:
    """The signal facts QA needs (section 11.1 step 1), from a raw take and its delivery, both in memory.

    Clipping counts raw samples at or beyond ``FULL_SCALE`` (32767/32768) in magnitude, infinities included.
    The DC offset is the mean of the raw samples, non-finite ones as zero. Voiced bounds and the longest
    internal silence use the trim's speech rule (DC-10: frame RMS with the mean removed, against
    max(p95 - 40 dB, -70 dBFS)) on the delivery, leaving out the ``fade_s`` at each end: the fades turn a
    DC offset into ramps that are not speech. ``voiced_start_s`` / ``voiced_end_s`` are the edges of its
    first and last speech frames (None when it has none), and the longest silence is the longest run of
    non-speech frames between them. ``internal_silences_s`` is every such run's length, in order (empty when
    there is no speech, or none between the edges).
    """
    n = int(raw.shape[0])
    finite = np.isfinite(raw)
    clean = np.where(finite, raw, 0.0)
    magnitude = np.abs(np.nan_to_num(raw, nan=0.0))
    frame = frame_length(delivery_rate)
    total = int(delivery.shape[0])
    edge = round(fade_s * delivery_rate)
    speech = np.flatnonzero(speech_frames(delivery, delivery_rate, VOICED_REL_DB, VOICED_FLOOR_DBFS, edge=edge))
    silences: tuple[float, ...] = ()
    if speech.size == 0:
        start: float | None = None
        end: float | None = None
        longest = 0.0
    else:
        start = int(speech[0]) * frame / delivery_rate
        end = min((int(speech[-1]) + 1) * frame, total) / delivery_rate
        gaps = np.diff(speech) - 1
        longest = int(gaps.max()) * frame / delivery_rate if gaps.size else 0.0
        silences = tuple(int(g) * frame / delivery_rate for g in gaps if g > 0)
    return SignalStats(
        raw_samples=n,
        raw_sample_rate=raw_rate,
        raw_clipping_fraction=float(np.count_nonzero(magnitude >= FULL_SCALE)) / n if n else 0.0,
        raw_nonfinite=not bool(np.all(finite)),
        raw_dc_offset=float(np.mean(clean)) if n else 0.0,
        delivery_duration_s=total / delivery_rate,
        voiced_start_s=start,
        voiced_end_s=end,
        longest_internal_silence_s=longest,
        internal_silences_s=silences,
    )


class DeliveryPipeline:
    """The ``DeliveryProcessor`` of section 13 (``narration.contracts.interfaces``), built by WP13.

    Stateless: every call depends only on its arguments and ``fade_s``. See the module docstring for the
    steps. ``fade_s`` is the configured ``[delivery] fade_s``: ``signal_stats`` reads delivery files, which
    do not say how long their fades are, and leaves that much of each end unmeasured.
    """

    def __init__(self, *, fade_s: float = DEFAULT_FADE_S) -> None:
        if fade_s < 0:
            raise ValueError(f"fade_s must be at least 0, got {fade_s}")
        self._fade_s = fade_s

    @property
    def tools(self) -> DeliveryTools:
        """The resampler's and meter's names and versions, and ``post`` (all enter the delivery key)."""
        return delivery_tools()

    def process(self, raw_path: Path, out_path: Path, profile: DeliveryConfig) -> DeliveryOutput:
        """Post-process the raw take at ``raw_path`` into ``out_path`` (temp name, then rename).

        On one machine with the same pinned versions, the same input gives the same bytes. Raises
        ``ValueError`` for audio with more than one channel or an unsupported ``profile.subtype``, and
        soundfile's errors for a file it cannot read.
        """
        raw, sample_rate = read_mono(raw_path)
        made = deliver(raw, sample_rate, profile)
        write_atomic(out_path, made.wav)
        return made.output

    def signal_stats(self, raw_path: Path, delivery_path: Path) -> SignalStats:
        """The signal facts QA needs (section 11.1 step 1); see ``measure_signal``."""
        raw, raw_rate = read_mono(raw_path)
        delivery, delivery_rate = read_mono(delivery_path)
        return measure_signal(raw, raw_rate, delivery, delivery_rate, fade_s=self._fade_s)
