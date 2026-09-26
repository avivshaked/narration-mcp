"""Every code of design section 14 is present with the design's retryability, severities and retake rule."""

from __future__ import annotations

import pytest

from narration.contracts import codes

# Section 14's error table, verbatim: code → retryable (None = "maybe"). Plus DC-2's two codes.
ERROR_TABLE = {
    "INVALID_ARGUMENT": False,
    "LIMIT_EXCEEDED": False,
    "NOT_FOUND": False,
    "PATH_NOT_ALLOWED": False,
    "UNSUPPORTED_AUDIO": False,
    "VOICE_FILE_MISMATCH": False,
    "VOICE_NOT_SYNTHETIC": False,
    "VOICE_NOT_MEASURED": False,
    "REF_TEXT_MISMATCH": False,
    "ENGINE_CHANGED": False,
    "CONTROL_UNSUPPORTED": False,
    "TEXT_REFUSED": False,
    "ENGINE_DRIFT": False,
    "BACKEND_NOT_INSTALLED": False,
    "DAEMON_UNAVAILABLE": True,
    "GPU_UNAVAILABLE": True,
    "STORE_FULL": True,
    "JOB_NOT_CANCELLABLE": False,
    "INTERNAL": None,
    "QUEUE_FULL": True,  # DC-2
    "RATE_LIMITED": True,  # DC-2
}

# Section 14's flag table: code → (severities, retake rule).
FLAG_TABLE = {
    "WRITTEN_FORM_TOKEN": ({"warn", "info"}, "never"),
    "TERM_SPLIT_ACROSS_CUES": ({"warn"}, "never"),
    "SEGMENT_TOO_LONG": ({"warn"}, "never"),
    "WER_HIGH": ({"warn", "fail"}, "at_fail"),
    "EXACT_SPAN_MISMATCH": ({"fail"}, "always"),
    "TERM_UNVERIFIED": ({"warn"}, "never"),
    "SPK_SIM_LOW": ({"warn", "fail"}, "at_fail"),
    "SPK_OUTLIER": ({"info"}, "never"),
    "PACE_FAST": ({"warn", "fail"}, "at_fail"),
    "PACE_SLOW": ({"warn", "fail"}, "at_fail"),
    "HEAD_INSERTION": ({"warn", "fail"}, "always"),
    "END_INSERTION": ({"warn", "fail"}, "at_fail"),
    "SILENCE_LONG": ({"warn", "fail"}, "at_fail"),
    "CLIPPING": ({"warn"}, "never"),
    "SIGNAL_INVALID": ({"warn", "fail"}, "at_fail"),  # plan.md DC-5: a gap-fill for section 11.1 step 1
    "TOKEN_CAP_HIT": ({"fail"}, "always"),
    "CUE_UNALIGNED": ({"warn"}, "always"),
    "CUE_LOW_CONFIDENCE": ({"warn"}, "never"),
    "CUE_ALIGNMENT_DISAGREE": ({"warn"}, "never"),
    "CUE_BOUNDARY_NO_PAUSE": ({"info"}, "never"),
    "ALIGNMENT_ERROR": ({"fail"}, "always"),
    "FIT_TIGHT": ({"warn"}, "never"),
    "OVER_SCENE": ({"warn"}, "never"),
    "CANARY_MISMATCH": ({"info"}, "never"),
    "LOUDNESS_UNDER_TARGET": ({"info"}, "never"),
    "GAIN_HIGH": ({"info"}, "never"),
    "RETAKEN": ({"info"}, "never"),
    "RENDER_FAILED": ({"error"}, "never"),
    "WORKER_CRASHED": ({"error"}, "never"),
    "GPU_OOM": ({"error"}, "never"),
    "QA_UNAVAILABLE": ({"error"}, "never"),
    "CANCELLED": ({"error"}, "never"),
}


def test_every_error_code_of_section_14_and_dc2_is_present_s14() -> None:
    assert set(codes.ERRORS) == set(ERROR_TABLE)
    for code, retryable in ERROR_TABLE.items():
        assert codes.ERRORS[code].retryable is retryable, code
        assert codes.ERRORS[code].hint, f"{code} needs a hint that says what to do next"


def test_every_flag_code_of_section_14_is_present_s14() -> None:
    assert set(codes.FLAGS) == set(FLAG_TABLE)
    for code, (severities, rule) in FLAG_TABLE.items():
        assert set(codes.FLAGS[code].severities) == severities, code
        assert codes.FLAGS[code].retake == rule, code


def test_module_constants_name_their_own_code() -> None:
    for code in [*ERROR_TABLE, *FLAG_TABLE]:
        assert getattr(codes, code) == code


@pytest.mark.parametrize(
    ("code", "severity", "expected"),
    [
        ("WER_HIGH", "warn", False),
        ("WER_HIGH", "fail", True),
        ("EXACT_SPAN_MISMATCH", "fail", True),
        ("CUE_UNALIGNED", "warn", True),
        ("HEAD_INSERTION", "warn", True),
        ("TERM_UNVERIFIED", "warn", False),
        ("SPK_OUTLIER", "info", False),
        ("TOKEN_CAP_HIT", "fail", True),
    ],
)
def test_retake_triggers_are_fails_cue_unaligned_and_head_insertion_s11_1(
    code: str, severity: codes.Severity, expected: bool
) -> None:
    assert codes.is_retake_trigger(code, severity) is expected


def test_every_retryable_error_says_to_wait_and_resend_dc2() -> None:
    for code in ("GPU_UNAVAILABLE", "QUEUE_FULL", "RATE_LIMITED"):
        assert "retry_after_s" in codes.ERRORS[code].hint


def test_unsupported_platform_is_daemon_unavailable_not_retryable_q2() -> None:
    from narration.contracts.errors import UnsupportedPlatform

    exc = UnsupportedPlatform("singleton", "linux")
    assert exc.code == "DAEMON_UNAVAILABLE"
    assert exc.retryable is False
    assert "Windows" in (exc.error.hint or "")
    assert exc.error.details == {"operation": "singleton", "platform": "linux"}


def test_qa_unavailable_names_its_code_s14() -> None:
    from narration.contracts.errors import QaUnavailable

    assert QaUnavailable.code == codes.QA_UNAVAILABLE


def test_a_cue_no_retake_can_place_is_not_a_retake_trigger_dc12_s11_1() -> None:
    no_words = {"reason": codes.CUE_NO_ALIGNABLE_WORDS}
    assert codes.CUE_NO_ALIGNABLE_WORDS == "no_alignable_words"
    assert codes.is_retake_trigger("CUE_UNALIGNED", "warn", no_words) is False
    assert codes.is_retake_trigger("CUE_UNALIGNED", "warn", {"reason": "low_confidence"}) is True
    assert codes.is_retake_trigger("CUE_UNALIGNED", "warn", {}) is True
    assert codes.is_retake_trigger("CUE_UNALIGNED", "warn", None) is True
    # The exception is CUE_UNALIGNED's alone: a fail, or another trigger, is still a trigger.
    assert codes.is_retake_trigger("ALIGNMENT_ERROR", "fail", no_words) is True
    assert codes.is_retake_trigger("HEAD_INSERTION", "warn", no_words) is True
