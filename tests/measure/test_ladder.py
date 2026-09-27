"""The length ladder's rules on numbers (design section 3.2, R8; ``narration.measure.ladder``).

Pace is spoken characters per second of speaking time (WP47); the numbers are invented."""

from __future__ import annotations

import pytest

from narration.contracts.models import PacePoint, PaceTrend
from narration.measure import ladder as lad

TREND = PaceTrend(intercept_cps=14.0, per_100_chars=0.5, band_max_chars=300)


def seed(
    cps: float | None = 15.0,
    *,
    attempt: int = 0,
    wer: float | None = 0.0,
    errors: int | None = 0,
    sim: float | None = 0.98,
    duration: float | None = 10.0,
    exact_mismatch: bool = False,
    head_insertion: bool = False,
    token_cap: bool = False,
    spoken_cps: float | None = None,
) -> lad.SeedTake:
    return lad.SeedTake(
        seed=1000 + attempt,
        attempt=attempt,
        take_id=f"tk_{attempt:016x}",
        cps=cps,
        wer_adj=wer,
        word_errors=errors,
        sim=sim,
        verdict="pass",
        duration_s=duration,
        exact_mismatch=exact_mismatch,
        head_insertion=head_insertion,
        token_cap=token_cap,
        spoken_cps=spoken_cps,
    )


def rung(chars: int, *seeds: lad.SeedTake, target: int | None = None) -> lad.Rung:
    takes = seeds or (seed(attempt=0), seed(attempt=1), seed(attempt=2))
    return lad.Rung(paragraph_id=f"ladder-{chars:03d}", target=target or chars, chars=chars, seeds=takes)


def judge(r: lad.Rung, *, tol: float = 0.10, sim_warn: float = 0.95) -> lad.RungJudgement:
    return lad.judge(r, trend=TREND, tol=tol, sim_warn=sim_warn)


# ======================================================================== the trend and tol


def test_median_ignores_missing_values_s3_2() -> None:
    assert lad.median([3.0, None, 1.0, 2.0]) == 2.0
    assert lad.median([None, None]) is None


def test_trend_is_least_squares_over_band_rung_medians_s3_2() -> None:
    rungs = [
        rung(100, seed(15.0, attempt=0), seed(15.1, attempt=1), seed(14.9, attempt=2)),
        rung(200, seed(15.5, attempt=0), seed(20.0, attempt=1), seed(15.4, attempt=2)),  # median 15.5
        rung(300, seed(16.0)),
        rung(400, seed(99.9)),  # above the band: not fitted
    ]
    trend = lad.fit_trend(rungs, 300)
    assert trend.per_100_chars == pytest.approx(0.5)
    assert trend.intercept_cps == pytest.approx(14.5)
    assert trend.band_max_chars == 300
    assert lad.trend_at(trend, 200) == pytest.approx(15.5)


def test_band_membership_is_by_the_rungs_target_not_its_length_s3_2() -> None:
    """ladder-300 may read 301 characters; its target, 300, keeps it in the band."""
    rungs = [rung(151, seed(15.0), target=150), rung(301, seed(16.0), target=300)]
    trend = lad.fit_trend(rungs, 300)
    assert trend.per_100_chars == pytest.approx(1.0 / 150.0 * 100.0)


def test_a_band_of_one_length_gives_a_flat_trend_s3_2() -> None:
    trend = lad.fit_trend([rung(80, seed(13.0))], 300)
    assert (trend.intercept_cps, trend.per_100_chars) == (13.0, 0.0)


def test_a_band_with_no_pace_cannot_be_fitted_s3_2() -> None:
    with pytest.raises(ValueError, match="no rung"):
        lad.fit_trend([rung(80, seed(None)), rung(400, seed(15.0))], 300)


def test_seed_spread_is_range_over_median_s3_2() -> None:
    assert lad.seed_spread(rung(80, seed(10.0), seed(11.0, attempt=1), seed(11.7, attempt=2))) == pytest.approx(
        0.17 / 1.1
    )
    assert lad.seed_spread(rung(80, seed(10.0), seed(None, attempt=1))) == 0.0


def test_tol_is_never_below_its_floor_or_the_bands_largest_spread_s3_2() -> None:
    steady = [rung(80, seed(10.0), seed(10.1, attempt=1))]
    assert lad.pace_tol(steady, 300, 0.10) == 0.10
    spread = [*steady, rung(150, seed(10.0), seed(11.7, attempt=1), seed(11.0, attempt=2))]
    assert lad.pace_tol(spread, 300, 0.10) == pytest.approx(round(1.7 / 11.0, 6))
    above = [*steady, rung(400, seed(10.0), seed(15.0, attempt=1), target=400)]
    assert lad.pace_tol(above, 300, 0.10) == 0.10  # a spread above the band does not widen tol


# ======================================================================== a rung's judgement


def test_a_rung_passes_when_every_rule_holds_s3_2() -> None:
    result = judge(rung(200))
    assert result.passes and result.reasons == ()
    assert result.median_cps == 15.0
    assert result.limit_cps == pytest.approx(15.0 * 1.10)  # the trend at 200 characters is 15 cps
    assert result.median_duration_s == 10.0


def test_pace_is_judged_against_the_trend_at_the_rungs_length_s3_2() -> None:
    limit = lad.trend_at(TREND, 200) * 1.10
    assert judge(rung(200, seed(limit))).passes  # at the limit: not above it
    fails = judge(rung(200, seed(limit + 0.001)))
    assert not fails.passes and "median pace" in fails.reasons[0]
    # a longer rung's trend is higher, so the same pace passes there
    assert judge(rung(500, seed(limit + 0.001))).passes


def test_pace_is_the_median_over_seeds_s3_2() -> None:
    limit = lad.trend_at(TREND, 200) * 1.10
    assert judge(rung(200, seed(15.0), seed(limit * 2, attempt=1), seed(15.0, attempt=2))).passes


def test_wer_fails_only_when_the_medians_meet_the_fail_rule_s3_2() -> None:
    one_slip = rung(80, *(seed(wer=0.07, errors=1, attempt=a) for a in range(3)))
    assert judge(one_slip).passes  # one error over 6 %: QA's warn, not its fail
    two = rung(80, *(seed(wer=0.07, errors=2, attempt=a) for a in range(3)))
    result = judge(two)
    assert not result.passes and "fail rule" in result.reasons[0]
    low_rate = rung(500, *(seed(wer=0.05, errors=4, attempt=a) for a in range(3)))
    assert judge(low_rate).passes  # four errors in a long rung stay under 6 %
    median_ok = rung(
        80, seed(wer=0.2, errors=3), seed(wer=0.0, errors=0, attempt=1), seed(wer=0.0, errors=0, attempt=2)
    )
    assert judge(median_ok).passes


def test_similarity_must_reach_the_warn_threshold_s3_2() -> None:
    assert judge(rung(80, seed(sim=0.95)), sim_warn=0.95).passes
    result = judge(rung(80, seed(sim=0.9499)), sim_warn=0.95)
    assert not result.passes and "similarity" in result.reasons[0]


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("exact_mismatch", "attempt 1 had an exact-span mismatch"),
        ("head_insertion", "attempt 1 had a head insertion"),
        ("token_cap", "attempt 1 had a token-cap hit"),
    ],
)
def test_one_seed_with_a_hard_fault_fails_the_rung_s3_2(fault: str, reason: str) -> None:
    r = rung(80, seed(attempt=0), seed(attempt=1, **{fault: True}), seed(attempt=2))
    result = judge(r)
    assert not result.passes
    assert result.reasons == (reason,)


def test_a_rung_with_nothing_measured_does_not_pass_s3_2() -> None:
    result = judge(rung(80, seed(None, wer=None, errors=None, sim=None)))
    assert not result.passes
    assert len(result.reasons) == 3


# ======================================================================== the outcome


def test_max_segment_chars_ends_at_the_first_failing_rung_s3_2() -> None:
    rungs = [
        rung(80, seed(duration=5.0), seed(duration=6.0, attempt=1), seed(duration=7.0, attempt=2)),
        rung(151, seed(duration=9.0)),
        rung(248, seed(duration=15.0)),
        rung(301, seed(duration=19.0)),
    ]
    judgements = [judge(rungs[0]), judge(rungs[1]), lad.RungJudgement(passes=False), judge(rungs[3])]
    result = lad.outcome(rungs, judgements)
    assert result.run == 2
    assert result.max_segment_chars == 151
    assert result.max_segment_seconds == 9.0
    assert result.curve == (PacePoint(chars=80, cps=15.0), PacePoint(chars=151, cps=15.0))


def test_max_segment_seconds_is_the_median_duration_at_that_rung_s3_2() -> None:
    r = rung(80, seed(duration=5.0), seed(duration=7.0, attempt=1), seed(duration=6.5, attempt=2))
    assert lad.outcome([r], [judge(r)]).max_segment_seconds == 6.5


def test_no_passing_rung_means_no_reliable_length_s3_2() -> None:
    result = lad.outcome([rung(80)], [lad.RungJudgement(passes=False)])
    assert (result.run, result.max_segment_chars, result.max_segment_seconds, result.curve) == (0, None, None, ())


def test_the_ladder_table_keeps_every_seed_s7_6() -> None:
    r = rung(80, seed(15.0123456789, sim=0.9812345678, spoken_cps=13.2), seed(attempt=1, token_cap=True))
    row = lad.ladder_rung(r, judge(r))
    assert row.chars == 80 and row.paragraph_id == "ladder-080" and not row.passes
    assert [s.attempt for s in row.seeds] == [0, 1]
    assert row.seeds[0].cps == 15.012346 and row.seeds[0].sim == 0.981235
    assert row.seeds[0].spoken_cps == 13.2 and row.seeds[1].spoken_cps is None


# ======================================================================== the speaking share (WP47)


def test_speaking_share_is_the_bands_median_speaking_time_over_voiced_span_s3_2() -> None:
    """spoken_cps / cps is speaking time / voiced span, since both rates have the rung's characters."""
    rungs = [
        rung(80, seed(16.0, spoken_cps=16.0), seed(16.0, spoken_cps=14.4, attempt=1)),  # 1.0 and 0.9
        rung(300, seed(16.0, spoken_cps=12.8), target=300),  # 0.8
        rung(400, seed(16.0, spoken_cps=8.0), target=400),  # above the band: not counted
    ]
    assert lad.speaking_share(rungs, 300) == pytest.approx(0.9)
    assert lad.speaking_share([rung(80, seed(16.0))], 300) == 1.0  # nothing to go on: no pauses assumed
