"""listen_first (design section 11.1) and the job report (sections 7.5, 11.1), on a small job of three segments."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from narration.contracts import codes
from narration.contracts.models import Consistency, CueTiming, Flag, QaResult, SegmentResult
from narration.contracts.names import REPORT_SCHEMA, Severity
from narration.contracts.serial import to_json
from narration.qa import Scorer
from narration.qa._flags import verdict_of
from narration.qa.listen_first import group_of

from .builders import Cue, inputs, segment, segment_result, take_result


def flag(code: str, severity: Severity, cue: int | None = None, message: str = "m") -> Flag:
    return Flag(code=code, severity=severity, message=f"{code} {message}", cue=cue)


def with_flags(result: QaResult, *flags: Flag) -> QaResult:
    """The result with more flags, and the verdict they give."""
    merged = (*result.flags, *flags)
    return dataclasses.replace(result, flags=merged, verdict=verdict_of(merged))


SCORER = Scorer()
SEG_A = segment("A calm night fell.", Cue("Then nine boats came in.", exact=((1, 2),)), segment_id="p01")
SEG_B = segment("The Velmoranth lamp was lit.", "Its keeper waited [sic].", segment_id="p02")
SEG_C = segment("The harbour slept.", segment_id="p03")


def job() -> list[SegmentResult]:
    a_qa = SCORER.score(inputs(SEG_A, "A calm night fell. Then ten boats came in."))  # EXACT_SPAN_MISMATCH, fail
    a_qa = with_flags(a_qa, flag(codes.PACE_FAST, "warn"))
    a_take = take_result(
        "tk_a0",
        a_qa,
        SEG_A,
        flags=[flag(codes.SPK_OUTLIER, "info"), flag(codes.LOUDNESS_UNDER_TARGET, "info")],
    )
    unplaced = (
        CueTiming(index=0, start_s=None, end_s=None, confidence=None),
        CueTiming(index=1, start_s=3.0, end_s=5.0, confidence=0.9),
    )
    b_qa = with_flags(
        SCORER.score(inputs(SEG_B)),
        flag(codes.CUE_UNALIGNED, "warn", 0),
        flag(codes.TERM_UNVERIFIED, "warn", 1),
        flag(codes.CUE_BOUNDARY_NO_PAUSE, "info", 1),
    )
    b_take = take_result("tk_b1", b_qa, SEG_B, attempt=1, cues=unplaced)
    assert b_take.alignment is not None
    b_take = dataclasses.replace(
        b_take, alignment=dataclasses.replace(b_take.alignment, flags=(flag(codes.CUE_UNALIGNED, "warn", 0),))
    )
    b_failed = take_result("tk_b0", with_flags(SCORER.score(inputs(SEG_B)), flag(codes.WER_HIGH, "fail")), SEG_B)
    return [
        segment_result(SEG_A, [a_take], suggested="tk_a0"),
        segment_result(
            SEG_B,
            [b_failed, b_take],
            suggested="tk_b1",
            flags=[flag(codes.SEGMENT_TOO_LONG, "warn")],
            cue_warnings={1: [flag(codes.WRITTEN_FORM_TOKEN, "warn", message="'[sic]'")]},
        ),
        segment_result(SEG_C, [], suggested=None, flags=[flag(codes.SEGMENT_TOO_LONG, "warn")]),
    ]


def test_listen_first_follows_the_designs_order_s11_1() -> None:
    items = SCORER.listen_first(job())
    got = [(i.segment_id, i.take_id, i.cue, i.reason.split(" ")[0]) for i in items]
    assert got == [
        ("p01", "tk_a0", 1, codes.EXACT_SPAN_MISMATCH),  # 1. fails
        ("p02", "tk_b1", 1, codes.TERM_UNVERIFIED),  # 2. exact-span and term flags
        ("p02", "tk_b1", 0, codes.CUE_UNALIGNED),  # 3. cue alignment (listed once, from qa and alignment)
        ("p01", "tk_a0", None, codes.PACE_FAST),  # 4. insertion, similarity and pace, and outliers
        ("p01", "tk_a0", None, codes.SPK_OUTLIER),
        ("p01", "tk_a0", None, codes.WER_HIGH),  # the slip's warning ("ten" for "nine": 1 error in 9 words)
        ("p02", "tk_b1", None, codes.SEGMENT_TOO_LONG),  # 5. over-long segments and text warnings
        ("p02", "tk_b1", 1, codes.WRITTEN_FORM_TOKEN),
        ("p03", None, None, codes.SEGMENT_TOO_LONG),
    ]


def test_listen_first_gives_where_to_listen_s11_1() -> None:
    items = {(i.segment_id, i.reason.split(" ")[0]): i for i in SCORER.listen_first(job())}
    exact = items[("p01", codes.EXACT_SPAN_MISMATCH)]
    assert (exact.from_s, exact.to_s) == pytest.approx((3.08, 5.78))  # cue 1 of the take
    whole = items[("p01", codes.PACE_FAST)]
    assert (whole.from_s, whole.to_s) == (0.0, 10.0)  # the whole take
    unplaced = items[("p02", codes.CUE_UNALIGNED)]
    assert (unplaced.from_s, unplaced.to_s) == (None, None)  # never interpolated
    no_take = items[("p03", codes.SEGMENT_TOO_LONG)]
    assert (no_take.take_id, no_take.from_s, no_take.to_s) == (None, None, None)


def test_listen_first_covers_the_suggested_take_only_s11_1() -> None:
    assert all(i.take_id != "tk_b0" for i in SCORER.listen_first(job()))


def test_insertion_windows_s11_1() -> None:
    seg = segment("One cue here.", "Two cue here.")
    qa = with_flags(
        SCORER.score(inputs(seg)), flag(codes.HEAD_INSERTION, "warn", 0), flag(codes.END_INSERTION, "warn", 1)
    )
    items = SCORER.listen_first([segment_result(seg, [take_result("tk_x", qa, seg)], suggested="tk_x")])
    windows = {i.reason.split(" ")[0]: (i.from_s, i.to_s) for i in items}
    assert windows[codes.HEAD_INSERTION] == pytest.approx((0.0, 2.78))  # from the start to the end of cue 0
    assert windows[codes.END_INSERTION] == pytest.approx((3.08, 10.0))  # from the last cue to the end


def test_listen_first_groups_s11_1() -> None:
    assert group_of(flag(codes.RENDER_FAILED, "error")) is None  # nothing to hear
    assert group_of(flag(codes.GAIN_HIGH, "info")) is None
    assert group_of(flag(codes.CUE_BOUNDARY_NO_PAUSE, "info")) is None
    assert group_of(flag(codes.WRITTEN_FORM_TOKEN, "info")) == 5  # a lone letter, reported for a listener
    assert group_of(flag(codes.SILENCE_LONG, "warn")) == 4
    assert group_of(flag(codes.HEAD_INSERTION, "fail")) == 1


# ======================================================================== the report


def results() -> dict[str, Any]:
    segments = job()
    engine_cue = dataclasses.replace(segments[1].text.cues[0], engine="The Vel-mor-anth lamp was lit.")
    segments[1] = dataclasses.replace(
        segments[1], text=dataclasses.replace(segments[1].text, cues=(engine_cue, *segments[1].text.cues[1:]))
    )
    return {
        "job": {
            "job_id": "job_01TEST",
            "kind": "generate",
            "status": "completed",
            "outcome": "needs_attention",
            "label": "harbour | draft *1*",
        },
        "voice": {"voice_hash": "sha256:" + "a" * 64, "clip_sha256": "b" * 64},
        "engine_profile": {"id": "qwen3-base-1.7b.p1", "hash": "sha256:" + "c" * 64},
        "measurement": {"max_segment_chars": 450, "max_segment_seconds": 31.5, "measured_at": "2026-09-26"},
        "segments": [to_json(s) for s in segments],
        "consistency": to_json(Consistency(min=0.97, median=0.985, outliers=("tk_a0",))),
        "listen_first": [to_json(i) for i in SCORER.listen_first(segments)],
    }


def test_report_md_lists_every_flag_and_each_cues_text_s11_1() -> None:
    md = SCORER.report_md(results())
    assert md.startswith("# Narration report: harbour \\| draft \\*1\\*\n")
    for code in (
        codes.EXACT_SPAN_MISMATCH,
        codes.PACE_FAST,
        codes.SPK_OUTLIER,
        codes.LOUDNESS_UNDER_TARGET,
        codes.CUE_UNALIGNED,
        codes.TERM_UNVERIFIED,
        codes.CUE_BOUNDARY_NO_PAUSE,
        codes.SEGMENT_TOO_LONG,
        codes.WRITTEN_FORM_TOKEN,
    ):
        assert f"`{code}`" in md, code
    assert "`WER_HIGH`" in md  # the replaced (non-suggested) attempt's flags are listed too
    assert "“The Velmoranth lamp was lit.”\n  → “The Vel-mor-anth lamp was lit.”" in md
    assert "“Its keeper waited \\[sic\\].” (the engine got the same text)" in md
    assert "## Listen first" in md and "1. p01 · `tk_a0` · cue 1 · 3.08–5.78 s: EXACT\\_SPAN\\_MISMATCH" in md
    assert "outliers: `tk_a0`" in md


def test_report_md_is_deterministic_and_tolerates_partial_results_s7_5() -> None:
    assert SCORER.report_md(results()) == SCORER.report_md(results())
    assert SCORER.report_md({}).startswith("# Narration report: narration job\n")
    partial = {"job": {"job_id": "job_x", "status": "cancelled"}, "segments": [{"segment_id": "p09", "takes": [{}]}]}
    assert "not scored" in SCORER.report_md(partial)


def test_report_md_gives_a_takes_pace_in_characters_per_second_of_speaking_s7_5() -> None:
    new = results()
    take = new["segments"][0]["takes"][0]
    take["qa"]["pace"] = {"spoken_wpm": 171.4, "articulation_cps": 16.94}
    take["qa"]["pace_expected"] = {"spoken_wpm": 150.2, "articulation_cps": 16.41}
    assert "pace 16.9 characters/s of speaking (expected 16.4), 171 spoken wpm" in SCORER.report_md(new)


def test_report_md_reads_a_take_scored_before_contracts_1_6_9_in_words_per_minute_s7_5() -> None:
    """A job finished before WP47 has no characters per second in its results: its pace line stays in spoken
    words per minute, as its pace was judged then, and never reads "n/a" (the owner's first jobs)."""
    for articulation in ({}, {"articulation_cps": None}):  # absent, or null in get_results
        old = results()
        for seg in old["segments"]:
            for take in seg["takes"]:
                take["qa"]["pace"] = {"spoken_wpm": 171.4, **articulation}
                take["qa"]["pace_expected"] = {"spoken_wpm": 150.2, **articulation}
        md = SCORER.report_md(old)
        assert md.count("171 spoken wpm (expected 150)") == 3
        assert "characters/s" not in md
        assert "n/a spoken wpm" not in md and "(expected n/a)" not in md


def test_report_json_counts_every_flag_s11_1() -> None:
    report = SCORER.report_json(results())
    assert report["schema"] == REPORT_SCHEMA
    assert report["summary"]["segments"] == 3
    assert report["summary"]["takes"] == 3
    assert report["summary"]["verdicts"] == {"fail": 2, "warn": 1}
    assert report["summary"]["flags"][codes.CUE_UNALIGNED] == 1  # once, though in qa and alignment
    assert [t["suggested"] for t in report["segments"][1]["takes"]] == [False, True]
    assert report["consistency"]["outliers"] == ["tk_a0"]
