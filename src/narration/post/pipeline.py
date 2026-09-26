"""The delivery pipeline (design section 13): raw take → ``delivery.wav``, on the CPU, deterministically.

In this order:

1. **Trim** (``narration.post.trim``): the silence at each end below the take's speech level (p95 of 20 ms
   frame RMS) - 40 dB, keeping up to ``pad_s`` (0.08 s) of it at each end.
2. **Resample** to ``sample_rate`` (48 kHz) with the pinned resampler (``narration.post.resample``).
3. **Gain**: one static gain to ``target_lufs`` (-16 LUFS integrated, BS.1770-4, ``narration.post.loudness``)
   measured on the resampled signal. No limiter. The gain is rounded to 0.0001 dB, so the gain the record
   states is exactly the gain applied, and float noise in the measurement cannot move a sample.
4. **Fades**: ``fade_s`` (0.01 s) raised-cosine ramps at both edges; the first and last samples are zero.
5. **Quantise** to ``subtype`` (PCM_24) mono (``narration.post.pcm``).
6. **True peak** of this final file, 4x oversampled. Above ``true_peak_dbtp`` (-1.0 dBTP), the gain is
   lowered to what the ceiling allows and the signal re-quantised, then measured again (and lowered by
   0.0001 dB steps while quantisation keeps it above); the take is flagged ``LOUDNESS_UNDER_TARGET`` (info)
   with the shortfall. **The ceiling wins over the loudness target.**

The loudness record's ``measured_lufs`` and ``true_peak_dbtp`` are measured on the final file. A gain above
+12 dB is flagged ``GAIN_HIGH`` (info). The same raw audio and profile give the same bytes, every time and in
every process. Everything runs on one thread, so the pipeline is within any CPU thread cap (section 4.1).

**Degenerate takes are delivered, not refused**, so that QA can fail them and trigger a retake:
non-finite raw samples are treated as zero (``SignalStats.raw_nonfinite`` reports them), and a take with no
measurable loudness (every 400 ms block under the -70 LUFS absolute gate, e.g. digital silence) gets no
gain; its record then states ``measured_lufs`` = -70.0 ("at or below the absolute gate") and, for an all-zero
file, ``true_peak_dbtp`` at one least significant bit.
"""

from __future__ import annotations

import functools
import importlib.metadata
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

from . import loudness, resample
from .pcm import bits_of, encode_wav, lsb_dbfs, quantise, read_mono, to_float, write_atomic
from .trim import find_trim, frame_length, rule_text, speech_frames

GAIN_HIGH_DB: Final = 12.0
"""A gain above this is flagged ``GAIN_HIGH`` (section 13)."""
DECIMALS: Final = 4
"""Gain, loudness and true peak are stated (and the gain applied) to 0.0001 dB."""
GAIN_STEP_DB: Final = 10.0**-DECIMALS
MAX_CEILING_STEPS: Final = 1000
SILENT_LUFS: Final = loudness.ABSOLUTE_GATE_LUFS
"""``measured_lufs`` of a take with no defined loudness: at or below the absolute gate."""
FULL_SCALE: Final = 32767 / 32768
"""A raw sample at or beyond this magnitude counts as clipped (the largest positive 16-bit value)."""
VOICED_REL_DB: Final = -40.0
"""The speech rule ``SignalStats`` uses on the delivery file: the trim's rule at its default."""

Audio = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True, kw_only=True)
class Delivered:
    """A delivery made in memory: the WAV file's bytes and what the processor reports about it."""

    wav: bytes
    output: DeliveryOutput


@functools.cache
def delivery_tools() -> DeliveryTools:
    """The resampler's and loudness meter's names and versions, as installed (section 10.2)."""
    version = importlib.metadata.version
    return DeliveryTools(
        resampler=resample.describe(version("scipy"), version("numpy")),
        loudness_meter=loudness.describe(version("pyloudnorm"), version("scipy"), version("numpy")),
    )


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


def _flag(code: str, message: str, details: dict[str, Any]) -> Flag:
    severity = codes.flag_code(code).severities[0]
    return Flag(
        code=code,
        severity=severity,
        message=message,
        retake_trigger=codes.is_retake_trigger(code, severity),
        details=details,
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
    points = find_trim(x, sample_rate, rel_db=profile.trim_rel_db, pad_s=profile.trim_pad_s)
    # 2. resample
    y = resample.resample(x[points.start : points.stop], sample_rate, rate)
    # 3. gain (measured before the fades, as the design orders the steps)
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

    final = to_float(q, bits)
    measured = loudness.integrated_loudness(final, rate)
    record = Loudness(
        measured_lufs=SILENT_LUFS if measured is None else round(measured, DECIMALS),
        gain_db=gain_db,
        true_peak_dbtp=lsb_dbfs(bits) if peak is None else round(peak, DECIMALS),
        ceiling_applied=ceiling_applied,
        target_lufs=profile.target_lufs,
    )
    flags: list[Flag] = []
    if ceiling_applied:
        shortfall = round(profile.target_lufs - record.measured_lufs, DECIMALS)
        reduction = round((target_gain if target_gain is not None else 0.0) - gain_db, DECIMALS)
        flags.append(
            _flag(
                codes.LOUDNESS_UNDER_TARGET,
                f"The true-peak ceiling of {ceiling:g} dBTP lowered the gain by {reduction:.2f} dB, so this take "
                f"sits at {record.measured_lufs:.2f} LUFS, {shortfall:.2f} dB under the {profile.target_lufs:g} LUFS "
                "target. To level takes of one script, use each take's loudness record.",
                {
                    "target_lufs": profile.target_lufs,
                    "measured_lufs": record.measured_lufs,
                    "shortfall_db": shortfall,
                    "gain_reduction_db": reduction,
                    "true_peak_dbtp": record.true_peak_dbtp,
                    "ceiling_dbtp": ceiling,
                },
            )
        )
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
            rule=rule_text(profile.trim_rel_db),
        ),
        loudness=record,
        flags=tuple(flags),
    )
    return Delivered(wav=encode_wav(q, rate, profile.subtype), output=output)


def measure_signal(raw: Audio, raw_rate: int, delivery: Audio, delivery_rate: int) -> SignalStats:
    """The signal facts QA needs (section 11.1 step 1), from a raw take and its delivery, both in memory.

    Clipping counts raw samples at or beyond ``FULL_SCALE`` (32767/32768) in magnitude, infinities included.
    The DC offset is the mean of the raw samples, non-finite ones as zero. Voiced bounds and the longest
    internal silence use the speech-frame rule of the trim (p95 of 20 ms frame RMS - 40 dB) on the delivery:
    ``voiced_start_s`` / ``voiced_end_s`` are the edges of its first and last speech frames (None when it
    has none), and the longest silence is the longest run of non-speech frames between them.
    """
    n = int(raw.shape[0])
    finite = np.isfinite(raw)
    clean = np.where(finite, raw, 0.0)
    magnitude = np.abs(np.nan_to_num(raw, nan=0.0))
    frame = frame_length(delivery_rate)
    total = int(delivery.shape[0])
    speech = np.flatnonzero(speech_frames(delivery, delivery_rate, VOICED_REL_DB))
    if speech.size == 0:
        start: float | None = None
        end: float | None = None
        longest = 0.0
    else:
        start = int(speech[0]) * frame / delivery_rate
        end = min((int(speech[-1]) + 1) * frame, total) / delivery_rate
        gaps = np.diff(speech) - 1
        longest = int(gaps.max()) * frame / delivery_rate if gaps.size else 0.0
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
    )


class DeliveryPipeline:
    """The ``DeliveryProcessor`` of section 13 (``narration.contracts.interfaces``), built by WP13.

    Stateless: every call depends only on its arguments. See the module docstring for the steps.
    """

    @property
    def tools(self) -> DeliveryTools:
        """The resampler's and loudness meter's names and versions (they enter the delivery key)."""
        return delivery_tools()

    def process(self, raw_path: Path, out_path: Path, profile: DeliveryConfig) -> DeliveryOutput:
        """Post-process the raw take at ``raw_path`` into ``out_path`` (temp name, then rename).

        The same input gives the same bytes, every time. Raises ``ValueError`` for audio with more than one
        channel or an unsupported ``profile.subtype``, and soundfile's errors for a file it cannot read.
        """
        raw, sample_rate = read_mono(raw_path)
        made = deliver(raw, sample_rate, profile)
        write_atomic(out_path, made.wav)
        return made.output

    def signal_stats(self, raw_path: Path, delivery_path: Path) -> SignalStats:
        """The signal facts QA needs (section 11.1 step 1); see ``measure_signal``."""
        raw, raw_rate = read_mono(raw_path)
        delivery, delivery_rate = read_mono(delivery_path)
        return measure_signal(raw, raw_rate, delivery, delivery_rate)
