"""Speaker consistency across a request's suggested takes (design section 11.1): a report, never a verdict.

After every take is scored and each segment's suggestion is made, each suggested take's embedding is compared
with the centroid of all of them. Takes below the voice's consistency baseline (``consistency_p5`` minus the
margin) are ``outliers``, each flagged ``SPK_OUTLIER`` (info) for ``listen_first``. The result depends on
which other takes are in the request, so it is computed per job and cached nowhere.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

import numpy as np

from narration.contracts import codes
from narration.contracts.models import Consistency, Flag, MeasurementRecord

from ._flags import make_flag

__all__ = ["consistency"]


def consistency(
    suggested: Sequence[tuple[str, tuple[float, ...]]],
    measurement: MeasurementRecord | None,
    margin: float,
) -> tuple[Consistency, tuple[Flag, ...]]:
    """Each suggested take's similarity to the set's centroid, and the outliers below the baseline.

    ``suggested`` is (take id, embedding) per suggested take. With fewer than two takes there is no set to
    compare with, so ``min`` and ``median`` are None. Without a measurement there is no baseline, so there are
    no outliers. Raises ValueError for embeddings of different lengths or a zero embedding.
    """
    if len(suggested) < 2:
        return Consistency(min=None, median=None), ()
    matrix = np.asarray([e for _, e in suggested], dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("embeddings of different lengths")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("a zero embedding has no direction")
    unit = matrix / norms[:, None]
    centroid = unit.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    sims = [float(s) for s in unit @ centroid]
    threshold = round(measurement.similarity.consistency_p5 - margin, 6) if measurement is not None else None
    outliers: list[str] = []
    flags: list[Flag] = []
    for (take_id, _), sim in zip(suggested, sims, strict=True):
        if threshold is not None and sim < threshold:
            outliers.append(take_id)
            flags.append(
                make_flag(
                    codes.SPK_OUTLIER,
                    "info",
                    f"take {take_id} stands apart from the request's other suggested takes: similarity "
                    f"{sim:.4f} to their centroid, below this voice's consistency baseline {threshold}",
                    segment_id=None,
                    details={"take_id": take_id, "similarity": sim, "threshold": threshold},
                )
            )
    return Consistency(min=min(sims), median=statistics.median(sims), outliers=tuple(outliers)), tuple(flags)
