"""Building flags the same way everywhere in ``narration.qa``."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from narration.contracts.codes import is_retake_trigger
from narration.contracts.models import Flag
from narration.contracts.names import Severity, Verdict

__all__ = ["make_flag", "verdict_of"]


def make_flag(
    code: str,
    severity: Severity,
    message: str,
    *,
    segment_id: str | None = None,
    cue: int | None = None,
    details: dict[str, Any] | None = None,
) -> Flag:
    """A flag with ``retake_trigger`` set by the contract's rule (section 11.1), which reads its details (DC-12)."""
    return Flag(
        code=code,
        severity=severity,
        message=message,
        segment_id=segment_id,
        cue=cue,
        retake_trigger=is_retake_trigger(code, severity, details),
        details=details,
    )


def verdict_of(flags: Iterable[Flag]) -> Verdict:
    """``fail`` if any flag is ``fail`` (or ``error``), else ``warn`` if any is ``warn``, else ``pass``.

    ``info`` flags never change a verdict.
    """
    severities = {flag.severity for flag in flags}
    if severities & {"fail", "error"}:
        return "fail"
    if "warn" in severities:
        return "warn"
    return "pass"
