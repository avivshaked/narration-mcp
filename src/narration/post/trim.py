"""The relative trim (design section 13, step 1, as revised by DC-9 and DC-10) and its speech-frame rule.

Frames are 20 ms, consecutive and non-overlapping from sample 0; the last, partial frame is measured over
the samples it has. **Frame RMS is measured on the take with its mean removed**, for measurement only: the
audio is never changed (DC-10: a DC offset of 0.001 made every frame speech).

A take's **speech level** is the 95th percentile of its frame RMS. A frame is **speech** when its RMS is at
least ``max(speech level + rel_db, floor_dbfs)``: -40 dB under the speech level, and never below
-70 dBFS (DC-10: a take under 5 % speech was never trimmed). Every other frame is silence. The rule is gain
independent for every take whose speech level is above ``floor_dbfs - rel_db`` (-30 dBFS).

``head`` and ``tail`` are the silence found before the first and after the last speech frame. At most
``pad_s`` of each is kept: a take with less silence than that at an end keeps what it has, and no silence
is added (DC-9). A take with no speech frame at all (silence) is not trimmed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

FRAME_S: Final = 0.020
"""The analysis frame: 20 ms (section 13)."""
SPEECH_PERCENTILE: Final = 95.0
"""The speech level is this percentile of the frame RMS (section 13), linear interpolation."""

Audio = npt.NDArray[np.float64]
Mask = npt.NDArray[np.bool_]


def frame_length(sample_rate: int) -> int:
    """Samples in one 20 ms frame at ``sample_rate`` (at least one)."""
    return max(1, round(FRAME_S * sample_rate))


def frame_rms(x: Audio, frame: int) -> Audio:
    """The RMS of each non-overlapping frame of ``frame`` samples, from sample 0.

    The last frame may be partial; it is measured over the samples it has. The mean is not removed here.
    """
    n = int(x.shape[0])
    full, rest = divmod(n, frame)
    rms = np.empty(full + (1 if rest else 0), dtype=np.float64)
    if full:
        blocks = x[: full * frame].reshape(full, frame)
        rms[:full] = np.sqrt(np.mean(np.square(blocks), axis=1))
    if rest:
        rms[full] = np.sqrt(np.mean(np.square(x[full * frame :])))
    return rms


def speech_threshold(rms: Audio, rel_db: float, floor_dbfs: float) -> float:
    """``max(p95(rms) + rel_db, floor_dbfs)`` as a linear RMS."""
    level = float(np.percentile(rms, SPEECH_PERCENTILE, method="linear")) if rms.size else 0.0
    return max(level * 10.0 ** (rel_db / 20.0), 10.0 ** (floor_dbfs / 20.0))


def speech_frames(x: Audio, sample_rate: int, rel_db: float, floor_dbfs: float, *, edge: int = 0) -> Mask:
    """Which 20 ms frames of ``x`` are speech under the rule of the module docstring.

    ``edge`` samples at each end are not measured (a delivery's fades: a fade turns a DC offset into a ramp,
    which the mean removal would otherwise read as sound). The mean is taken over the measured samples, a
    frame is measured over its measured samples only, and a frame with none is silence and is left out of
    the percentile. With ``edge`` 0 (the trim) every sample is measured.
    """
    n = int(x.shape[0])
    frame = frame_length(sample_rate)
    frames = -(-n // frame)
    lo, hi = min(max(edge, 0), n), max(n - max(edge, 0), 0)
    if hi <= lo:
        return np.zeros(frames, dtype=np.bool_)
    y = x - np.mean(x[lo:hi])
    rms = frame_rms(y, frame)
    measured = np.ones(frames, dtype=np.bool_)
    if lo > 0 or hi < n:
        edges = set(range(0, -(-lo // frame))) | set(range(hi // frame, frames))
        for i in sorted(edges):
            start, stop = max(i * frame, lo), min((i + 1) * frame, hi)
            if stop <= start:
                rms[i], measured[i] = 0.0, False
            else:
                rms[i] = np.sqrt(np.mean(np.square(y[start:stop])))
    threshold = speech_threshold(rms[measured], rel_db, floor_dbfs)
    return measured & (rms > 0.0) & (rms >= threshold)


@dataclass(frozen=True, slots=True, kw_only=True)
class TrimPoints:
    """Where the trim cuts a raw take, in samples at the raw rate.

    ``head`` and ``tail`` are the silence found at each end; ``pad`` is ``pad_s`` in samples. The delivery
    keeps ``[start, stop)``, so its length before resampling is
    ``samples - max(head - pad, 0) - max(tail - pad, 0)`` (section 13, DC-9).
    """

    sample_rate: int
    samples: int
    head: int
    tail: int
    pad: int

    @property
    def start(self) -> int:
        """The first raw sample kept."""
        return max(self.head - self.pad, 0)

    @property
    def stop(self) -> int:
        """One past the last raw sample kept."""
        return self.samples - max(self.tail - self.pad, 0)

    @property
    def kept(self) -> int:
        """Raw samples kept: ``stop - start``."""
        return self.stop - self.start

    @property
    def head_s(self) -> float:
        """The silence found before the speech, in seconds."""
        return self.head / self.sample_rate

    @property
    def tail_s(self) -> float:
        """The silence found after the speech, in seconds."""
        return self.tail / self.sample_rate


def find_trim(x: Audio, sample_rate: int, *, rel_db: float, floor_dbfs: float, pad_s: float) -> TrimPoints:
    """Find the silence at each end of a raw take (section 13 step 1)."""
    n = int(x.shape[0])
    speech = np.flatnonzero(speech_frames(x, sample_rate, rel_db, floor_dbfs))
    if speech.size == 0:
        head = tail = 0
    else:
        frame = frame_length(sample_rate)
        head = int(speech[0]) * frame
        tail = n - min((int(speech[-1]) + 1) * frame, n)
    return TrimPoints(sample_rate=sample_rate, samples=n, head=head, tail=tail, pad=round(pad_s * sample_rate))


def rule_text(rel_db: float, floor_dbfs: float) -> str:
    """The trim rule in words, as the take record states it (App. B):
    ``max(p95_frame_rms - 40 dB, -70 dBFS), mean removed; head_s/tail_s found, up to pad_s kept``."""
    sign = "-" if rel_db < 0 else "+"
    return (
        f"max(p95_frame_rms {sign} {abs(rel_db):g} dB, {floor_dbfs:g} dBFS), mean removed; "
        "head_s/tail_s found, up to pad_s kept"
    )
