"""Suggestion tiers (design section 8), the consistency report (section 11.1) and fit (section 12)."""

from __future__ import annotations

import itertools
import math

import pytest

from narration.contracts import codes
from narration.contracts.models import Fit, Flag
from narration.qa import Scorer, fit_report
from narration.qa._flags import verdict_of
from narration.qa.suggest import suggest, tier_of

from .builders import measurement, scored, unit

UNALIGNED = Flag(code=codes.CUE_UNALIGNED, severity="warn", message="unplaced", cue=0, retake_trigger=True)


def fails(n: int) -> list[Flag]:
    return [Flag(code=codes.WER_HIGH, severity="fail", message=f"f{i}") for i in range(n)]


# ======================================================================== suggestion (section 8)


def test_tier_1_is_the_lowest_passing_attempt_s8() -> None:
    takes = [scored("tk_w0", 0, "warn"), scored("tk_p2", 2, "pass"), scored("tk_p1", 1, "pass")]
    take_id, suggestion = suggest(takes)
    assert take_id == "tk_p1"
    assert suggestion is not None and (suggestion.tier, suggestion.reason) == (1, "lowest passing attempt")


def test_tier_2_is_a_warned_take_with_every_cue_placed_s8() -> None:
    takes = [
        scored("tk_f0", 0, "fail", flags=fails(1)),
        scored("tk_u1", 1, "warn", flags=[UNALIGNED], placed=False),
        scored("tk_w2", 2, "warn"),
    ]
    take_id, suggestion = suggest(takes)
    assert (take_id, suggestion and suggestion.tier) == ("tk_w2", 2)


def test_tier_3_is_a_warned_take_with_a_cue_unplaced_s8() -> None:
    takes = [
        scored("tk_f0", 0, "fail", flags=fails(1)),
        scored("tk_u3", 3, "warn", flags=[UNALIGNED], placed=False),
        scored("tk_u1", 1, "warn", flags=[UNALIGNED], placed=False),
    ]
    take_id, suggestion = suggest(takes)
    assert (take_id, suggestion and suggestion.tier) == ("tk_u1", 3)


def test_tier_4_prefers_placed_cues_then_fewest_fail_flags_s8() -> None:
    takes = [
        scored("tk_a0", 0, "fail", flags=[*fails(1), UNALIGNED], placed=False),
        scored("tk_a1", 1, "fail", flags=fails(3)),
        scored("tk_a2", 2, "fail", flags=fails(2)),
        scored("tk_a3", 3, "fail", flags=fails(2)),
    ]
    take_id, suggestion = suggest(takes)
    assert (take_id, suggestion and suggestion.tier) == ("tk_a2", 4)


def test_a_pass_with_an_unplaced_cue_is_not_tier_1_s8() -> None:
    assert tier_of(scored("tk_x", 0, "pass", placed=False)) == 3


def test_no_takes_no_suggestion_s8() -> None:
    assert suggest([]) == (None, None)


def test_the_suggestion_does_not_depend_on_take_order_s8() -> None:
    takes = [scored("tk_w0", 0, "warn"), scored("tk_p2", 2, "pass"), scored("tk_f1", 1, "fail", flags=fails(1))]
    assert {suggest(list(p))[0] for p in itertools.permutations(takes)} == {"tk_p2"}


# ======================================================================== consistency (section 11.1)


def rotated(angle: float) -> tuple[float, ...]:
    return unit(math.cos(angle), math.sin(angle))


def test_consistency_needs_at_least_two_takes_s11_1() -> None:
    report, flags = Scorer().consistency([("tk_a", unit(1.0))], measurement())
    assert (report.min, report.median, report.outliers, flags) == (None, None, (), ())


def test_consistency_lists_outliers_below_the_baseline_s11_1() -> None:
    suggested = [("tk_a", rotated(0.0)), ("tk_b", rotated(0.01)), ("tk_c", rotated(-0.01)), ("tk_d", rotated(0.6))]
    report, flags = Scorer().consistency(suggested, measurement(consistency_p5=0.98))
    assert report.outliers == ("tk_d",)
    assert report.min is not None and report.median is not None and report.min < report.median
    (flag,) = flags
    assert (flag.code, flag.severity, flag.retake_trigger) == (codes.SPK_OUTLIER, "info", False)
    assert flag.details is not None and flag.details["take_id"] == "tk_d" and flag.details["threshold"] == 0.97
    # A report, never a verdict: the flag is info and changes nothing.
    assert verdict_of(flags) == "pass"


def test_consistency_baseline_edge_s11_1() -> None:
    # Two takes at angle 2a: each is cos(a) from their centroid.
    angle = 0.2
    sim = math.cos(angle)
    suggested = [("tk_a", rotated(angle)), ("tk_b", rotated(-angle))]
    above = measurement(consistency_p5=round(sim + 0.01 - 1e-6, 6))  # threshold just below the similarity
    below = measurement(consistency_p5=round(sim + 0.01 + 1e-6, 6))  # threshold just above it
    assert Scorer().consistency(suggested, above)[0].outliers == ()
    assert Scorer().consistency(suggested, below)[0].outliers == ("tk_a", "tk_b")


def test_consistency_without_a_measurement_reports_no_outliers_s11_1() -> None:
    report, flags = Scorer().consistency([("tk_a", rotated(0.0)), ("tk_b", rotated(1.0))], None)
    assert report.min == pytest.approx(math.cos(0.5))
    assert (report.outliers, flags) == ((), ())


def test_consistency_rejects_mismatched_embeddings() -> None:
    with pytest.raises(ValueError):
        Scorer().consistency([("tk_a", (1.0, 0.0)), ("tk_b", (1.0, 0.0, 0.0))], None)


# ======================================================================== fit (section 12)


def test_no_scene_seconds_no_fit_of_any_kind_s12() -> None:
    assert fit_report(12.0, None, Fit(lead_in_s=1.0), segment_id="p01") is None


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        (7.0 - 1e-6, []),  # slack just over 1.0 s
        (7.0, []),  # slack exactly 1.0 s: not tight
        (7.0 + 1e-6, [codes.FIT_TIGHT]),
        (8.0, [codes.FIT_TIGHT]),  # slack 0: fits, tightly
        (8.0 + 1e-6, [codes.OVER_SCENE]),
        (9.5, [codes.OVER_SCENE]),
    ],
)
def test_fit_edges_s12(duration: float, expected: list[str]) -> None:
    report = fit_report(duration, 10.0, Fit(lead_in_s=1.5, tail_s=0.5), segment_id="p01")
    assert report is not None
    assert report.budget_s == 8.0
    assert report.slack_s == pytest.approx(8.0 - duration)
    assert report.overrun_s == pytest.approx(max(0.0, duration - 8.0))
    assert [f.code for f in report.flags] == expected
    assert all(f.severity == "warn" and f.retake_trigger is False for f in report.flags)


def test_fit_margins_default_to_zero_s12() -> None:
    report = fit_report(8.0, 10.0, None, segment_id="p01")
    assert report is not None and (report.budget_s, report.slack_s, report.flags) == (10.0, 2.0, ())
