"""The calibration set's numbers: the voice's anchor and similarity baseline (design sections 3.2 and 11.1).

Pure functions over speaker embeddings (the QA worker's WavLM-SV x-vectors; plan.md P1: the worker returns
raw outputs, and every number here is the server's).

- **The anchor** is the centroid over the clip and the calibration takes: each embedding is scaled to unit
  length, they are averaged with equal weight, and the mean is scaled to unit length again. That is the
  centroid ``narration.qa.consistency`` uses for a request's takes, so the two agree.
- **The similarity baseline** is the distribution of the calibration takes' similarities (cosine):
  ``anchor_p5`` and ``anchor_p50`` of each take's similarity to the anchor, and ``consistency_p5`` of each
  take's similarity to the centroid of the takes alone (the clip left out), which is what the per-job
  consistency report compares a request's suggested takes with (section 11.1). QA warns below
  ``anchor_p5 - sim_warn_margin`` and reports an outlier below ``consistency_p5 - sim_warn_margin``.
- **Percentiles** are numpy's default (linear interpolation between closest ranks), so with the default 12
  calibration takes ``p5`` lies between the lowest and the second lowest similarity.

Values are rounded to 6 decimals, as QA rounds the thresholds it derives from them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from narration.contracts.models import Anchor, SimilarityBaseline

DECIMALS: Final = 6


@dataclass(frozen=True, slots=True)
class Calibration:
    """The anchor, the baseline, and each calibration take's similarity to the anchor (in the order given)."""

    anchor: Anchor
    similarity: SimilarityBaseline
    take_sims: tuple[float, ...]


def _unit_rows(embeddings: Sequence[Sequence[float]]) -> npt.NDArray[np.float64]:
    matrix = np.asarray([list(e) for e in embeddings], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("the embeddings must all have the same, non-zero length")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0.0) or not np.all(np.isfinite(matrix)):
        raise ValueError("a zero or non-finite embedding has no direction")
    return matrix / norms[:, None]


def centroid(embeddings: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """The unit-length mean of the unit-length embeddings. Raises ValueError for no embeddings, embeddings of
    different lengths, a zero or non-finite one, or embeddings that cancel out."""
    if not embeddings:
        raise ValueError("no embeddings to average")
    mean = _unit_rows(embeddings).mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0.0:
        raise ValueError("the embeddings cancel out: their centroid has no direction")
    return tuple(float(v) for v in mean / norm)


def similarities(embeddings: Sequence[Sequence[float]], to: Sequence[float]) -> tuple[float, ...]:
    """Each embedding's cosine similarity to ``to``."""
    rows = _unit_rows(embeddings)
    target = _unit_rows([to])[0]
    if rows.shape[1] != target.shape[0]:
        raise ValueError("the embeddings and the target have different lengths")
    return tuple(float(v) for v in rows @ target)


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile, by linear interpolation between closest ranks (numpy's default)."""
    if not values:
        raise ValueError("no values")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def calibrate(clip: Sequence[float], takes: Sequence[Sequence[float]], *, model: str) -> Calibration:
    """The anchor over the clip and the calibration takes, and the takes' similarity baseline (the module
    docstring). ``model`` names the speaker model (``repo@revision``). Raises ValueError for no takes or
    embeddings that cannot be compared."""
    if not takes:
        raise ValueError("no calibration takes to build an anchor from")
    anchor = centroid([clip, *takes])
    to_anchor = similarities(takes, anchor)
    to_set = similarities(takes, centroid(takes))
    return Calibration(
        anchor=Anchor(model=model, dim=len(anchor), embedding=anchor),
        similarity=SimilarityBaseline(
            anchor_p5=round(percentile(to_anchor, 5), DECIMALS),
            anchor_p50=round(percentile(to_anchor, 50), DECIMALS),
            consistency_p5=round(percentile(to_set, 5), DECIMALS),
        ),
        take_sims=tuple(round(s, DECIMALS) for s in to_anchor),
    )


__all__ = ["DECIMALS", "Calibration", "calibrate", "centroid", "percentile", "similarities"]
