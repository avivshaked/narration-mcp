"""The relative trim (design section 13, step 1) and the speech-frame rule it rests on.

A take's **speech level** is the 95th percentile of its 20 ms frame RMS. A frame is **speech** when its RMS
is above zero and at least the speech level plus ``rel_db`` (-40 dB by default); every other frame is
silence. Because the threshold moves with the take's own level, the trim is gain independent.

Frames are consecutive and non-overlapping, starting at sample 0; the last, partial frame is measured over
the samples it has. ``head`` and ``tail`` are the silence found before the first and after the last speech
frame. Up to ``pad_s`` of each is kept: when less silence than that was found, all of it is kept and
nothing is added. A take with no speech frame at all (digital silence) is not trimmed.
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

    The last frame may be partial; it is measured over the samples it has.
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


def speech_frames(x: Audio, sample_rate: int, rel_db: float) -> Mask:
    """Which 20 ms frames of ``x`` are speech under the relative rule (the module docstring)."""
    rms = frame_rms(x, frame_length(sample_rate))
    if rms.size == 0:
        return np.zeros(0, dtype=np.bool_)
    level = float(np.percentile(rms, SPEECH_PERCENTILE, method="linear"))
    threshold = level * 10.0 ** (rel_db / 20.0)
    return (rms > 0.0) & (rms >= threshold)


@dataclass(frozen=True, slots=True, kw_only=True)
class TrimPoints:
    """Where the trim cuts a raw take, in samples at the raw rate.

    ``head`` and ``tail`` are the silence found at each end; ``pad`` is ``pad_s`` in samples. The delivery
    keeps ``[start, stop)``, so its length before resampling is
    ``samples - max(head - pad, 0) - max(tail - pad, 0)``.
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


def find_trim(x: Audio, sample_rate: int, *, rel_db: float, pad_s: float) -> TrimPoints:
    """Find the silence at each end of a raw take (section 13 step 1)."""
    n = int(x.shape[0])
    speech = np.flatnonzero(speech_frames(x, sample_rate, rel_db))
    if speech.size == 0:
        head = tail = 0
    else:
        frame = frame_length(sample_rate)
        head = int(speech[0]) * frame
        tail = n - min((int(speech[-1]) + 1) * frame, n)
    return TrimPoints(sample_rate=sample_rate, samples=n, head=head, tail=tail, pad=round(pad_s * sample_rate))


def rule_text(rel_db: float) -> str:
    """The trim rule in words, as the take record states it (App. B): ``p95_frame_rms - 40 dB; …``."""
    sign = "-" if rel_db < 0 else "+"
    return f"p95_frame_rms {sign} {abs(rel_db):g} dB; head_s/tail_s found, pad_s kept"
