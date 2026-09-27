"""The calibration set's anchor and similarity baseline (design sections 3.2 and 11.1; ``measure.baseline``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from narration.measure.baseline import calibrate, centroid, percentile, similarities

MODEL = "microsoft/wavlm-base-plus-sv@0000000000000000000000000000000000000000"


def unit(*values: float) -> tuple[float, ...]:
    norm = math.sqrt(sum(v * v for v in values))
    return tuple(v / norm for v in values)


def test_centroid_weighs_every_embedding_alike_whatever_its_length_s3_2() -> None:
    assert centroid([(3.0, 0.0), (0.0, 0.5)]) == pytest.approx(unit(1.0, 1.0))
    assert math.isclose(math.hypot(*centroid([(1.0, 2.0), (2.0, 1.0), (5.0, 0.1)])), 1.0)


@pytest.mark.parametrize(
    "embeddings",
    [[], [(0.0, 0.0)], [(1.0, 0.0), (-1.0, 0.0)], [(1.0, float("nan"))], [()]],
    ids=["none", "zero", "cancel", "nan", "empty"],
)
def test_centroid_refuses_embeddings_with_no_direction_s3_2(embeddings: list[tuple[float, ...]]) -> None:
    with pytest.raises(ValueError):
        centroid(embeddings)


def test_similarities_are_cosines_s3_2() -> None:
    assert similarities([(2.0, 0.0), (0.0, 1.0), (1.0, 1.0)], (1.0, 0.0)) == pytest.approx((1.0, 0.0, math.sqrt(0.5)))
    with pytest.raises(ValueError, match="different lengths"):
        similarities([(1.0, 0.0)], (1.0, 0.0, 0.0))


def test_percentiles_interpolate_between_closest_ranks_s3_2() -> None:
    values = [0.90, 0.95, 0.96, 0.97, 0.98]
    assert percentile(values, 50) == pytest.approx(0.96)
    assert percentile(values, 5) == pytest.approx(0.90 + 0.2 * 0.05)
    with pytest.raises(ValueError):
        percentile([], 5)


def test_calibrate_builds_the_anchor_over_the_clip_and_the_takes_s3_2() -> None:
    rng = np.random.default_rng(7)
    base = rng.normal(size=16)
    clip = tuple(float(v) for v in base)
    takes = [tuple(float(v) for v in base + rng.normal(scale=0.1, size=16)) for _ in range(12)]
    result = calibrate(clip, takes, model=MODEL)
    assert result.anchor.model == MODEL
    assert result.anchor.dim == 16 == len(result.anchor.embedding)
    assert result.anchor.embedding == pytest.approx(centroid([clip, *takes]))
    sims = similarities(takes, result.anchor.embedding)
    assert result.take_sims == tuple(round(s, 6) for s in sims)
    assert result.similarity.anchor_p5 == round(float(np.percentile(sims, 5)), 6)
    assert result.similarity.anchor_p50 == round(float(np.percentile(sims, 50)), 6)
    own = similarities(takes, centroid(takes))
    assert result.similarity.consistency_p5 == round(float(np.percentile(own, 5)), 6)
    assert min(sims) <= result.similarity.anchor_p5 <= sorted(sims)[1]


def test_the_clip_pulls_the_anchor_but_not_the_consistency_baseline_s11_1() -> None:
    takes = [unit(1.0, 0.1), unit(1.0, -0.1)]
    near = calibrate(unit(1.0, 0.0), takes, model=MODEL)
    far = calibrate(unit(0.0, 1.0), takes, model=MODEL)
    assert far.similarity.anchor_p50 < near.similarity.anchor_p50
    assert far.similarity.consistency_p5 == near.similarity.consistency_p5


def test_calibrate_needs_takes_s3_2() -> None:
    with pytest.raises(ValueError, match="no calibration takes"):
        calibrate((1.0, 0.0), [], model=MODEL)
