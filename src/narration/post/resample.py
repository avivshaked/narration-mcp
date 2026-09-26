"""The pinned resampler (design section 13, step 2; its name and version enter the delivery key, 10.2).

``scipy.signal.resample_poly`` (polyphase, zero padding at the edges) with a filter pinned here rather than
scipy's default: a Kaiser-windowed sinc (``firwin``) with beta 8.6 and 32 zero crossings on each side at the
higher of the two rates, cut off at the lower rate's Nyquist frequency. From 24 kHz to 48 kHz it is flat
to within 0.001 dB up to 11 kHz and at least 75 dB down from 13 kHz (KNOW: measured with ``freqz``).

The output has ``ceil(samples * up / down)`` samples, where ``up / down`` is the reduced ratio of the
rates. It runs on one thread, so it is within any CPU thread cap (section 4.1).
"""

from __future__ import annotations

import functools
import math
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from scipy.signal import firwin, resample_poly

ZERO_CROSSINGS: Final = 32
"""Zero crossings of the sinc on each side, counted at the higher rate of the pair."""
KAISER_BETA: Final = 8.6

Audio = npt.NDArray[np.float64]


def ratio(source_rate: int, target_rate: int) -> tuple[int, int]:
    """``(up, down)``: the reduced ratio ``target_rate / source_rate``."""
    if source_rate < 1 or target_rate < 1:
        raise ValueError(f"sample rates must be positive, got {source_rate} and {target_rate}")
    g = math.gcd(source_rate, target_rate)
    return target_rate // g, source_rate // g


@functools.cache
def lowpass(up: int, down: int) -> Audio:
    """The pinned anti-imaging / anti-aliasing filter for a ratio ``up / down`` (read-only)."""
    m = max(up, down)
    # firwin documents ``window`` as a name or a (name, parameter) tuple; its signature is inferred as str.
    window: Any = ("kaiser", KAISER_BETA)
    taps = np.asarray(firwin(2 * ZERO_CROSSINGS * m + 1, 1.0 / m, window=window), dtype=np.float64)
    taps.setflags(write=False)
    return taps


def resampled_length(samples: int, source_rate: int, target_rate: int) -> int:
    """The number of samples ``resample`` returns: ``ceil(samples * up / down)``."""
    up, down = ratio(source_rate, target_rate)
    return -(-samples * up // down)


def resample(x: Audio, source_rate: int, target_rate: int) -> Audio:
    """Resample a mono float64 signal with the pinned filter."""
    up, down = ratio(source_rate, target_rate)
    if up == down == 1:
        return np.array(x, dtype=np.float64, copy=True)
    if x.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    y = resample_poly(x, up, down, window=lowpass(up, down), padtype="constant")
    return np.asarray(y, dtype=np.float64)


def describe(scipy_version: str, numpy_version: str) -> str:
    """The resampler's name, version and pinned parameters, for the delivery key (section 10.2)."""
    return (
        f"scipy.signal.resample_poly {scipy_version} "
        f"(firwin kaiser beta={KAISER_BETA}, {ZERO_CROSSINGS} zero crossings, padtype=constant; numpy {numpy_version})"
    )
