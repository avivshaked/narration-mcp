"""The loudness meter (design section 13, steps 3 and 6; its name and version enter the delivery key, 10.2).

**Integrated loudness**: ITU-R BS.1770-4, from ``pyloudnorm``, on the mono signal as a single channel with
weight 1.0. The K-weighting filters are pyloudnorm's ``"DeMan"`` class, whose coefficients at 48 kHz are
the ones BS.1770-4 tabulates (KNOW: they agree to 1e-8; pyloudnorm's default ``"K-weighting"`` class,
derived from RBJ's cookbook, reads 0.04 LU low on a 997 Hz sine). Gating as BS.1770-4: 400 ms blocks with
75 % overlap, an absolute gate at -70 LUFS and a relative gate 10 LU below.

- pyloudnorm rounds the number of blocks, so it may count a final block that runs past the end of the
  signal (as if silence followed). Meters that count whole blocks only can read up to about 0.06 LU higher
  on a short take (KNOW: at most 0.059 LU on the 48 bakeoff clone paragraphs), inside EBU Tech 3341's
  tolerance of 0.1 LU.
- A signal shorter than one 400 ms block is measured as if silence followed it up to one block (BS.1770-4
  defines no loudness for less than a block).
- A signal with no block above the absolute gate has no defined loudness: the functions return None.

**True peak**: the signal oversampled 4x with the pinned resampler (``narration.post.resample``), and the
largest absolute value of the oversampled and the original samples, in dB relative to full scale (dBTP).
This is BS.1770-4 Annex 2's method with the pinned interpolation filter in place of the Annex's 48-tap
example filter. Both run on one thread.
"""

from __future__ import annotations

import math
import warnings
from typing import Final

import numpy as np
import numpy.typing as npt
import pyloudnorm

from .resample import resample

FILTER_CLASS: Final = "DeMan"
"""pyloudnorm's filter class with BS.1770-4's own K-weighting coefficients."""
BLOCK_S: Final = 0.400
ABSOLUTE_GATE_LUFS: Final = -70.0
OVERSAMPLING: Final = 4

Audio = npt.NDArray[np.float64]


def integrated_loudness(x: Audio, sample_rate: int) -> float | None:
    """BS.1770-4 integrated loudness of a mono signal in LUFS, or None when it is undefined (silence)."""
    minimum = math.ceil(BLOCK_S * sample_rate)
    if x.shape[0] < minimum:
        x = np.concatenate([x, np.zeros(minimum - x.shape[0], dtype=np.float64)])
    meter = pyloudnorm.Meter(sample_rate, filter_class=FILTER_CLASS, block_size=BLOCK_S)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        value = float(meter.integrated_loudness(x))
    return value if math.isfinite(value) else None


def true_peak_dbtp(x: Audio, sample_rate: int) -> float | None:
    """The true peak of a mono signal in dBTP (4x oversampled), or None for an empty or all-zero signal."""
    if x.shape[0] == 0:
        return None
    over = resample(x, sample_rate, sample_rate * OVERSAMPLING)
    peak = max(float(np.max(np.abs(x))), float(np.max(np.abs(over))))
    return 20.0 * math.log10(peak) if peak > 0.0 else None


def describe(pyloudnorm_version: str, scipy_version: str, numpy_version: str) -> str:
    """The meter's name, version and pinned parameters, for the delivery key (section 10.2)."""
    return (
        f"pyloudnorm {pyloudnorm_version} (BS.1770-4, filter_class={FILTER_CLASS}, mono weight 1.0; "
        f"true peak {OVERSAMPLING}x oversampled with the pinned resampler; "
        f"scipy {scipy_version}, numpy {numpy_version})"
    )
