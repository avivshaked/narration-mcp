"""The loudness meter (design section 13, steps 3 and 6; its name and version enter the delivery key, 10.2).

**Integrated loudness**: ITU-R BS.1770-4, from ``pyloudnorm``, on the mono signal as a single channel with
weight 1.0. The K-weighting filters are pyloudnorm's ``"DeMan"`` class, whose coefficients at 48 kHz are
the ones BS.1770-4 tabulates (KNOW: they agree to 1e-8; pyloudnorm's default ``"K-weighting"`` class,
derived from RBJ's cookbook, reads 0.04 LU low on a 997 Hz sine). Gating as BS.1770-4: 400 ms blocks with
75 % overlap, an absolute gate at -70 LUFS and a relative gate 10 LU below.

- **Whole blocks only**, as BS.1770-4 (and libebur128) count them. pyloudnorm rounds its block count, so
  it would also count a final block that runs past the end of the signal, as if silence followed: on the
  48 bakeoff clone paragraphs that read up to 0.13 LU low, beyond EBU Tech 3341's 0.1 LU tolerance (KNOW).
  So pyloudnorm is given the signal up to the end of its last whole block; the samples after it lie in no
  whole block, so nothing the standard measures is dropped.
- A signal shorter than one 400 ms block is measured as if silence followed it up to one block (BS.1770-4
  defines no loudness for less than a block).
- A signal with no block above the absolute gate has no defined loudness: the functions return None.

**True peak**: the signal oversampled 4x with the pinned resampler (``narration.post.resample``), and the
largest absolute value of the oversampled and the original samples, in dB relative to full scale (dBTP).
This is BS.1770-4 Annex 2's method with the pinned interpolation filter in place of the Annex's 48-tap
example filter. Neither hands work to other threads (``narration.post.pipeline``, on the thread cap).
"""

from __future__ import annotations

import importlib.metadata
import math
import warnings
from typing import Final

import numpy as np
import numpy.typing as npt
import pyloudnorm
import scipy

from .resample import resample

FILTER_CLASS: Final = "DeMan"
"""pyloudnorm's filter class with BS.1770-4's own K-weighting coefficients."""
BLOCK_S: Final = 0.400
OVERLAP: Final = 0.75
ABSOLUTE_GATE_LUFS: Final = -70.0
OVERSAMPLING: Final = 4

Audio = npt.NDArray[np.float64]


def whole_blocks_end(samples: int, sample_rate: int) -> int:
    """One past the last sample of the last whole gating block of a signal of ``samples`` samples.

    Blocks start every ``BLOCK_S * (1 - OVERLAP)`` seconds and end where pyloudnorm computes their upper
    bound, ``int(BLOCK_S * (j * (1 - OVERLAP) + 1) * sample_rate)``; the last whole block is the last one
    that ends at or before ``samples``. Requires at least one whole block.
    """
    step = 1.0 - OVERLAP
    j = math.floor((samples / sample_rate - BLOCK_S) / (BLOCK_S * step))
    while j > 0 and int(BLOCK_S * (j * step + 1) * sample_rate) > samples:
        j -= 1
    while int(BLOCK_S * ((j + 1) * step + 1) * sample_rate) <= samples:
        j += 1
    return int(BLOCK_S * (j * step + 1) * sample_rate)


def integrated_loudness(x: Audio, sample_rate: int) -> float | None:
    """BS.1770-4 integrated loudness of a mono signal in LUFS, or None when it is undefined (silence)."""
    minimum = math.ceil(BLOCK_S * sample_rate)
    if x.shape[0] < minimum:
        x = np.concatenate([x, np.zeros(minimum - x.shape[0], dtype=np.float64)])
    else:
        x = x[: max(whole_blocks_end(int(x.shape[0]), sample_rate), minimum)]
    meter = pyloudnorm.Meter(sample_rate, filter_class=FILTER_CLASS, block_size=BLOCK_S, overlap=OVERLAP)
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


def describe() -> str:
    """The meter's name, version and pinned parameters, for the delivery key (section 10.2).

    scipy's and numpy's versions are those of the loaded modules. pyloudnorm has no ``__version__``, so its
    version is its installed distribution's.
    """
    return (
        f"pyloudnorm {importlib.metadata.version('pyloudnorm')} (BS.1770-4, filter_class={FILTER_CLASS}, "
        f"mono weight 1.0, whole blocks only; true peak {OVERSAMPLING}x oversampled with the pinned resampler; "
        f"scipy {scipy.__version__}, numpy {np.__version__})"
    )
