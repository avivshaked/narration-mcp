"""Fit against a scene, reported only (design section 12).

Without ``scene_seconds`` there is no fit computation, no fit flag and no fit block (R4), so ``fit_report``
returns None. With it: budget = ``scene_seconds - lead_in_s - tail_s`` (both default to 0, so the service
adds no padding of its own); ``slack_s`` always; ``OVER_SCENE`` (warn, with ``overrun_s``) when the take is
longer than the budget; ``FIT_TIGHT`` (warn) when it fits with less than 1.0 s to spare. Nothing is truncated
or stretched. Fit is computed per request from the take's duration and is outside every cache key.
"""

from __future__ import annotations

from typing import Final

from narration.contracts import codes
from narration.contracts.models import Fit, FitReport

from ._flags import make_flag

__all__ = ["FIT_TIGHT_BELOW_S", "fit_report"]

FIT_TIGHT_BELOW_S: Final = 1.0
"""``FIT_TIGHT`` when the slack is below this many seconds (section 11.1's table, section 12)."""


def fit_report(
    duration_s: float, scene_seconds: float | None, fit: Fit | None, *, segment_id: str | None
) -> FitReport | None:
    """The fit of one take's delivery duration to the segment's scene, or None without ``scene_seconds``.

    A take over budget gets ``OVER_SCENE`` only; ``FIT_TIGHT`` is for a take that fits with little slack.
    """
    if scene_seconds is None:
        return None
    lead_in = fit.lead_in_s if fit is not None else 0.0
    tail = fit.tail_s if fit is not None else 0.0
    budget = scene_seconds - lead_in - tail
    slack = budget - duration_s
    overrun = max(0.0, -slack)
    details = {"budget_s": budget, "duration_s": duration_s, "slack_s": slack}
    if duration_s > budget:
        flag = make_flag(
            codes.OVER_SCENE,
            "warn",
            f"the take is {overrun:.2f} s over its {budget:.2f} s budget (reported, not remedied)",
            segment_id=segment_id,
            details={**details, "overrun_s": overrun},
        )
        flags = (flag,)
    elif slack < FIT_TIGHT_BELOW_S:
        flag = make_flag(
            codes.FIT_TIGHT,
            "warn",
            f"the take fits its {budget:.2f} s budget with only {slack:.2f} s to spare",
            segment_id=segment_id,
            details=details,
        )
        flags = (flag,)
    else:
        flags = ()
    return FitReport(
        scene_seconds=scene_seconds,
        budget_s=budget,
        duration_s=duration_s,
        slack_s=slack,
        overrun_s=overrun,
        flags=flags,
    )
