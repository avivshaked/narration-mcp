"""Pauses in the delivery audio, for snapping cue boundaries (design section 11.2 step 4).

The audio is cut into 20 ms energy frames. A frame is silent when its RMS level is more than
``silence_below_db`` under a reference, the 95th percentile of the take's frame levels, so the rule does not
depend on the take's loudness. A **pause** is a run of at least ``min_pause_frames`` silent frames; a run
that touches the start or the end of the file counts at any length, because the file's edge closes it.

Starting values, ASSUME until the alignment benchmark (WP38) sets them: 35 dB and 2 frames (40 ms). On the
bake-off's six clone takes every gap between two sentences held such a pause, and 16 % of the gaps between
words inside a sentence did (KNOW: ``spikes/b-forced-align-cpu/calibration.json``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True, kw_only=True)
class PauseParams:
    """How pauses are found. Every field is part of the aligner's method id (section 11.2)."""

    frame_s: float = 0.02
    reference_percentile: float = 95.0
    silence_below_db: float = 35.0
    min_pause_frames: int = 2


@dataclass(frozen=True, slots=True)
class Pause:
    """A run of silent frames, ``[start_s, end_s)`` in seconds of the audio."""

    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def frame_levels_db(audio: npt.NDArray[Any], sample_rate: int, frame_s: float) -> npt.NDArray[np.float64]:
    """RMS level of each ``frame_s`` frame in dB (full scale 0 dB); the last, partial frame is included."""
    data = np.asarray(audio, dtype=np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)
    n = max(1, round(frame_s * sample_rate))
    count = -(-data.shape[0] // n)
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    padded = np.zeros(count * n, dtype=np.float64)
    padded[: data.shape[0]] = np.nan_to_num(data)
    squares = (padded.reshape(count, n) ** 2).sum(axis=1)
    lengths = np.full(count, n, dtype=np.float64)
    lengths[-1] = data.shape[0] - (count - 1) * n
    rms = np.sqrt(squares / lengths)
    return 20.0 * np.log10(np.maximum(rms, 1e-10))


def find_pauses(audio: npt.NDArray[Any], sample_rate: int, params: PauseParams) -> tuple[Pause, ...]:
    """Every pause in the audio, in order, in seconds."""
    levels = frame_levels_db(audio, sample_rate, params.frame_s)
    if levels.size == 0:
        return ()
    duration = np.asarray(audio).shape[0] / sample_rate
    threshold = float(np.percentile(levels, params.reference_percentile)) - params.silence_below_db
    silent = levels < threshold
    pauses: list[Pause] = []
    start: int | None = None
    for i, is_silent in enumerate([*silent.tolist(), False]):
        if is_silent and start is None:
            start = i
        elif not is_silent and start is not None:
            at_edge = start == 0 or i == levels.size
            if at_edge or i - start >= params.min_pause_frames:
                pauses.append(Pause(start * params.frame_s, min(i * params.frame_s, duration)))
            start = None
    return tuple(pauses)
