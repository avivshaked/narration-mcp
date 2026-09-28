"""The job report's "Failures" section (plan.md WP48): every take that failed QA or that a retake replaced, with
its fail and warn flags and the take that finally filled its slot, and no file path.

The results are written by hand in ``get_results``' layout: one segment whose slot took three attempts (two
failed, the third passed), and one whose only take failed with no retake left.
"""

from __future__ import annotations

from typing import Any

from narration.contracts import codes
from narration.qa import Scorer
from narration.qa.report import replacements

SCORER = Scorer()


def flag(code: str, severity: str, message: str = "m", **extra: Any) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": f"{code} {message}", **extra}


def retaken(*earlier: int) -> dict[str, Any]:
    """The ``RETAKEN`` flag the job writes on a retake, naming every earlier attempt of the slot."""
    replaced = [{"attempt": n, "take_id": f"tk_{n}", "verdict": "fail"} for n in earlier]
    return flag(codes.RETAKEN, "info", details={"slot": 0, "replaced": replaced})


def take(
    attempt: int, verdict: str, *qa_flags: dict[str, Any], flags: tuple[dict[str, Any], ...] = ()
) -> dict[str, Any]:
    return {
        "take_id": f"tk_{attempt}",
        "attempt": attempt,
        "delivery": {"path": "/somewhere/in/the/store/delivery.wav", "sha256": "0" * 64},
        "qa": {"verdict": verdict, "flags": list(qa_flags)},
        "flags": list(flags),
    }


def results() -> dict[str, Any]:
    return {
        "job": {"job_id": "job_01TEST", "kind": "generate", "status": "completed", "outcome": "needs_attention"},
        "segments": [
            {
                "segment_id": "p01",
                "takes": [
                    take(0, "fail", flag(codes.TOKEN_CAP_HIT, "fail"), flag(codes.CUE_BOUNDARY_NO_PAUSE, "info")),
                    take(1, "warn", flag(codes.HEAD_INSERTION, "warn", cue=0), flags=(retaken(0),)),
                    take(2, "pass", flags=(retaken(0, 1),)),
                ],
            },
            {"segment_id": "p02", "takes": [take(0, "fail", flag(codes.WER_HIGH, "fail"))]},
            {"segment_id": "p03", "takes": [take(0, "pass")]},
        ],
    }


def test_replacements_names_the_slots_last_attempt_wp48() -> None:
    attempts = [(0, []), (1, [retaken(0)]), (2, [retaken(0, 1)]), (3, []), (4, [retaken(3)])]
    assert replacements(attempts) == {0: 2, 1: 2, 3: 4}  # the retake naming the most is the slot's last
    assert replacements([(0, [flag(codes.WER_HIGH, "fail")])]) == {}
    odd = flag(codes.RETAKEN, "info", details={"replaced": [{"attempt": True}, {"attempt": "1"}, {}]})
    assert replacements([(5, [odd])]) == {}  # only whole attempt numbers count


def test_report_md_lists_every_failed_or_replaced_take_with_its_flags_wp48() -> None:
    md = SCORER.report_md(results())
    section = md.split("## Failures\n", 1)[1].split("\n## ", 1)[0]
    assert md.index("## Failures") < md.index("## Segments")
    assert "(3)" in section
    lines = [line for line in section.splitlines() if line.startswith("- ")]
    assert lines == [
        "- p01 · attempt 0 · `tk_0` · **fail** · replaced: the slot was filled by attempt 2 `tk_2` (pass)",
        "- p01 · attempt 1 · `tk_1` · **warn** · replaced: the slot was filled by attempt 2 `tk_2` (pass)",
        "- p02 · attempt 0 · `tk_0` · **fail** · not replaced: the last take of its slot",
    ]
    assert f"**fail** `{codes.TOKEN_CAP_HIT}`" in section and f"**warn** `{codes.HEAD_INSERTION}` (cue 0)" in section
    assert f"`{codes.WER_HIGH}`" in section
    assert codes.CUE_BOUNDARY_NO_PAUSE not in section  # an info flag is no reason
    assert "delivery.wav" not in md and "/somewhere" not in md  # the report holds no file path


def test_report_json_has_the_same_failures_wp48() -> None:
    failures = SCORER.report_json(results())["failures"]
    assert [(f["segment_id"], f["attempt"], f["replaced"], f["final_take"]) for f in failures] == [
        ("p01", 0, True, {"attempt": 2, "take_id": "tk_2", "verdict": "pass"}),
        ("p01", 1, True, {"attempt": 2, "take_id": "tk_2", "verdict": "pass"}),
        ("p02", 0, False, {"attempt": 0, "take_id": "tk_0", "verdict": "fail"}),
    ]
    assert [[f["code"] for f in failure["flags"]] for failure in failures] == [
        [codes.TOKEN_CAP_HIT],
        [codes.HEAD_INSERTION],
        [codes.WER_HIGH],
    ]


def test_a_job_with_no_failure_says_so_wp48() -> None:
    clean = {"job": {"job_id": "job_01TEST"}, "segments": [{"segment_id": "p01", "takes": [take(0, "pass")]}]}
    assert "## Failures\n\nNone: no take failed QA, and no retake replaced one." in SCORER.report_md(clean)
    assert SCORER.report_json(clean)["failures"] == []
    assert "## Failures" in SCORER.report_md({})  # a partial object still reads
