"""``CtcAligner.resolve`` (design section 11.2 steps 3 to 8) on synthetic audio and worker-shaped replies.

The audio is a tone where speech is and exact zeros in pauses (``_support.audio``); a reply places each
word's letters where the test says (``_support.reply``). So every expected time below is known exactly.
"""

from __future__ import annotations

import dataclasses
from typing import cast

import pytest
from jsonschema import Draft202012Validator

from narration.align import AlignerParams, CtcAligner
from narration.config import AlignmentConfig
from narration.contracts import codes
from narration.contracts.errors import WorkerFailure
from narration.contracts.interfaces import AlignerCore
from narration.contracts.models import Alignment, ErrorStats, Hint, MeasuredError
from narration.contracts.names import ALIGNMENT_METHOD, MODEL_ALIGNER, MODEL_ASR
from narration.contracts.schemas import record_schema
from narration.contracts.serial import from_json, to_json
from narration.contracts.worker import AlignReply

from ._support import RATE, REVISION, audio, codes_of, cue_times, reply, segment

ALIGNER = CtcAligner(revision=REVISION)

# Two cues with a pause: speech 0.20–1.40, pause 1.40–1.90, speech 1.90–2.80, silence to 3.00. The aligner's
# letters end 60 ms before the first speech ends and start 60 ms after the second starts.
TWO_CUES = ("Then it turns.", "For home.")
TWO_AUDIO = audio(3.0, [(0.20, 1.40), (1.90, 2.80)])
TWO_WORDS = {
    (0, 0): (0.24, 0.50),
    (0, 1): (0.56, 0.80),
    (0, 2): (0.86, 1.34),
    (1, 0): (1.96, 2.20),
    (1, 1): (2.26, 2.74),
}


def _resolve(cues, speech_audio, word_times, *, hints=(), asr_words=(), scores=None, aligner=ALIGNER):
    seg = segment(*cues, hints=hints)
    t = aligner.build_transcript(seg, list(hints))
    r = reply(t, word_times, samples=speech_audio.shape[0], scores=scores)
    return aligner.resolve(t, r, speech_audio, RATE, list(asr_words), None)


def test_boundary_snapped_to_the_pause_edges_s11_2() -> None:
    a = _resolve(TWO_CUES, TWO_AUDIO, TWO_WORDS)
    assert cue_times(a) == [(0.2, 1.4), (1.9, 2.8)]
    assert codes_of(a) == []


def test_first_onset_and_last_offset_snapped_s11_2() -> None:
    a = _resolve(TWO_CUES, TWO_AUDIO, TWO_WORDS)
    assert a.cues[0].start_s == 0.2 and a.cues[-1].end_s == 2.8


def test_edge_words_take_the_snapped_times_middle_words_keep_theirs_s11_2() -> None:
    a = _resolve(TWO_CUES, TWO_AUDIO, TWO_WORDS)
    first, second = a.cues
    assert [(w.text, w.start_s, w.end_s) for w in first.words] == [
        ("Then", 0.2, 0.5),
        ("it", 0.56, 0.8),
        ("turns", 0.86, 1.4),
    ]
    assert [(w.text, w.start_s, w.end_s) for w in second.words] == [("For", 1.9, 2.2), ("home", 2.26, 2.8)]


def test_no_pause_keeps_the_aligner_times_and_says_so_s11_2() -> None:
    speech = audio(3.0, [(0.20, 2.80)])
    words = TWO_WORDS | {(1, 0): (1.40, 1.64), (1, 1): (1.70, 2.74)}
    a = _resolve(("Then it turns", "for home."), speech, words)
    assert cue_times(a) == [(0.2, 1.34), (1.4, 2.8)]
    [flag] = a.flags
    assert (flag.code, flag.severity, flag.cue, flag.details) == (
        codes.CUE_BOUNDARY_NO_PAUSE,
        "info",
        0,
        {"next_cue": 1},
    )
    assert flag.retake_trigger is False


def test_first_onset_is_the_pause_before_the_first_word_not_speech_before_it_s11_2() -> None:
    # 0.1–0.5 s is speech before the text (reference bleed, say): the cue starts after the pause that follows.
    speech = audio(1.8, [(0.10, 0.50), (0.80, 1.60)])
    a = _resolve(("Then sun.",), speech, {(0, 0): (0.84, 1.10), (0, 1): (1.16, 1.56)})
    assert cue_times(a) == [(0.8, 1.6)]


def test_confidence_is_the_mean_posterior_of_the_cues_letters_s11_2() -> None:
    a = _resolve(TWO_CUES, TWO_AUDIO, TWO_WORDS, scores={0: 0.9, 1: 0.8})
    assert [c.confidence for c in a.cues] == [0.9, 0.8]


def test_low_confidence_is_flagged_and_the_cue_kept_s11_2() -> None:
    a = _resolve(TWO_CUES, TWO_AUDIO, TWO_WORDS, scores={1: 0.6})
    assert cue_times(a) == [(0.2, 1.4), (1.9, 2.8)]
    [flag] = a.flags
    assert (flag.code, flag.severity, flag.cue, flag.retake_trigger) == (codes.CUE_LOW_CONFIDENCE, "warn", 1, False)
    assert flag.details == {"confidence": 0.6, "threshold": 0.75}


# Three cues; the middle one is not in the audio as written (its letters score 0.2).
THREE_CUES = ("Rain came.", "Nobody knew.", "Then sun.")
THREE_AUDIO = audio(3.6, [(0.20, 1.00), (1.40, 2.20), (2.60, 3.40)])
THREE_WORDS = {
    (0, 0): (0.24, 0.50),
    (0, 1): (0.56, 0.96),
    (1, 0): (1.44, 1.80),
    (1, 1): (1.86, 2.16),
    (2, 0): (2.64, 2.90),
    (2, 1): (2.96, 3.36),
}


def test_unplaced_cue_is_null_never_interpolated_s11_2() -> None:
    a = _resolve(THREE_CUES, THREE_AUDIO, THREE_WORDS, scores={1: 0.2})
    assert cue_times(a) == [(0.2, 1.0), (None, None), (2.6, 3.4)]
    middle = a.cues[1]
    assert middle.confidence == 0.2
    assert [(w.text, w.start_s, w.end_s) for w in middle.words] == [("Nobody", None, None), ("knew", None, None)]
    [flag] = [f for f in a.flags if f.cue == 1]
    assert (flag.code, flag.severity, flag.retake_trigger) == (codes.CUE_UNALIGNED, "warn", True)
    assert flag.details == {"reason": "low_confidence", "confidence": 0.2, "threshold": 0.5}
    # The neighbours end and start at their own pauses; nothing covers the unplaced cue's audio.
    assert a.cues[0].end_s == 1.0 and a.cues[2].start_s == 2.6
    assert not any(f.code == codes.CUE_LOW_CONFIDENCE for f in a.flags)


# Speech 0.20-1.00, a pause, then 1.40-2.60 with no pause inside: the middle cue is spoken 1.42-1.78 and runs
# straight on into "Then", whose letters start at 1.84.
RUN_ON_AUDIO = audio(3.0, [(0.20, 1.00), (1.40, 2.60)])
RUN_ON_WORDS = {
    (0, 0): (0.24, 0.50),
    (0, 1): (0.56, 0.96),
    (2, 0): (1.84, 2.10),
    (2, 1): (2.16, 2.56),
}


@pytest.mark.parametrize(
    ("cues", "middle", "scores"),
    [
        (("Rain came.", "12.", "Then sun."), {(1, 0): (1.42, 1.78)}, None),  # a wildcard alone
        (("Rain came.", "Nobody knew.", "Then sun."), {(1, 0): (1.42, 1.60), (1, 1): (1.64, 1.78)}, {1: 0.2}),
    ],
    ids=["no_alignable_words", "low_confidence"],
)
def test_cue_after_an_unplaced_cue_does_not_take_its_speech_s11_2(cues, middle, scores) -> None:
    # Regression (review F1): the pause before the unplaced cue's speech is not "Then"'s pause. Snapping to it
    # gave cue 2 and "Then" 1.40, 0.44 s of another cue's speech before its letters.
    a = _resolve(cues, RUN_ON_AUDIO, RUN_ON_WORDS | middle, scores=scores)
    assert cue_times(a) == [(0.2, 1.0), (None, None), (1.84, 2.6)]
    assert a.cues[2].words[0].start_s == 1.84
    [flag] = [f for f in a.flags if f.code == codes.CUE_BOUNDARY_NO_PAUSE]
    assert (flag.severity, flag.cue, flag.retake_trigger) == ("info", 1, False)
    assert flag.details == {"next_cue": 2, "edge": "start", "speech_s": 0.44}


def test_first_placed_cue_after_an_unplaced_head_keeps_its_own_start_s11_2() -> None:
    # Regression (review F1): "12" is spoken 0.20-0.60 and runs on into "Then" (letters from 0.64); the
    # file's leading silence is not "Then"'s pause, so cue 1 starts at its letters, not at 0.20.
    speech = audio(2.0, [(0.20, 1.60)])
    words = {(0, 0): (0.22, 0.58), (1, 0): (0.64, 0.90), (1, 1): (0.96, 1.20), (1, 2): (1.26, 1.56)}
    a = _resolve(("12.", "Then sun came."), speech, words)
    assert cue_times(a) == [(None, None), (0.64, 1.6)]
    assert a.cues[1].words[0].start_s == 0.64
    [flag] = [f for f in a.flags if f.code == codes.CUE_BOUNDARY_NO_PAUSE]
    assert (flag.cue, flag.details) == (0, {"next_cue": 1, "edge": "start", "speech_s": 0.44})


def test_last_placed_cue_before_an_unplaced_tail_keeps_its_own_end_s11_2() -> None:
    # Regression (review F1): "came" ends at 0.96 and "12" runs on from it to 1.40; the pause after "12" is
    # not "came"'s, so cue 0 ends at its letters, not at 1.40.
    speech = audio(2.0, [(0.20, 1.40)])
    words = {(0, 0): (0.24, 0.44), (0, 1): (0.50, 0.70), (0, 2): (0.76, 0.96), (1, 0): (1.02, 1.36)}
    a = _resolve(("Then sun came.", "12."), speech, words)
    assert cue_times(a) == [(0.2, 0.96), (None, None)]
    assert a.cues[0].words[-1].end_s == 0.96
    [flag] = [f for f in a.flags if f.code == codes.CUE_BOUNDARY_NO_PAUSE]
    assert (flag.cue, flag.details) == (0, {"next_cue": 1, "edge": "end", "speech_s": 0.44})


@pytest.mark.parametrize(("then_start", "expected"), [(1.52, 1.4), (1.54, 1.54)])
def test_pause_next_to_an_unplaced_cue_counts_only_within_reach_s11_2(then_start: float, expected: float) -> None:
    # snap_reach_start_s (0.12): at most that much speech between the pause and the cue's first letter.
    speech = audio(3.0, [(0.20, 1.00), (1.40, 2.60)])
    words = {(0, 0): (0.24, 0.50), (0, 1): (0.56, 0.96), (2, 0): (then_start, 2.10), (2, 1): (2.16, 2.56)}
    a = _resolve(
        ("Rain came.", "Nobody knew.", "Then sun."),
        speech,
        words | {(1, 0): (1.02, 1.20), (1, 1): (1.24, 1.36)},
        scores={1: 0.2},
    )
    assert a.cues[2].start_s == expected


def test_cue_with_no_word_in_the_alphabet_is_unplaced_and_no_retake_trigger_s11_2_dc12() -> None:
    # "12, 13." is one wildcard: its span shows speech at 1.40-2.20, but no letter says which words.
    words = {k: v for k, v in THREE_WORDS.items() if k[0] != 1} | {(1, 0): (1.40, 2.20)}
    a = _resolve(("Rain came.", "12, 13.", "Then sun."), THREE_AUDIO, words)
    assert cue_times(a) == [(0.2, 1.0), (None, None), (2.6, 3.4)]
    assert a.cues[1].confidence is None
    assert [(w.text, w.start_s) for w in a.cues[1].words] == [("12", None), ("13", None)]
    [flag] = [f for f in a.flags if f.cue == 1]
    assert (flag.code, flag.details) == (codes.CUE_UNALIGNED, {"reason": codes.CUE_NO_ALIGNABLE_WORDS})
    assert flag.retake_trigger is False


def test_every_unplaced_cue_says_why_s11_2_dc12() -> None:
    low = _resolve(THREE_CUES, THREE_AUDIO, THREE_WORDS, scores={1: 0.2})
    others = {k: v for k, v in THREE_WORDS.items() if k[0] != 1}
    none = _resolve(("Rain came.", "-", "Then sun."), THREE_AUDIO, others)
    t = ALIGNER.build_transcript(segment(*TWO_CUES), [])
    failed = ALIGNER.resolve(t, None, TWO_AUDIO, RATE, [], None)
    seen: dict[object, bool | None] = {}
    for alignment in (low, none, failed):
        for flag in alignment.flags:
            if flag.code == codes.CUE_UNALIGNED:
                assert flag.details is not None
                seen[flag.details["reason"]] = flag.retake_trigger
    assert seen == {"low_confidence": True, codes.CUE_NO_ALIGNABLE_WORDS: False, "alignment_error": True}


def test_cue_of_punctuation_only_is_unplaced_s11_2() -> None:
    words = {(0, 0): (0.24, 0.50), (0, 1): (0.56, 0.96), (2, 0): (2.64, 2.90), (2, 1): (2.96, 3.36)}
    a = _resolve(("Rain came.", "—", "Then sun."), THREE_AUDIO, words)
    assert cue_times(a) == [(0.2, 1.0), (None, None), (2.6, 3.4)]
    assert a.cues[1].words == ()
    assert (codes.CUE_UNALIGNED, 1) in codes_of(a)


def test_left_out_word_at_a_cue_head_starts_the_cue_at_its_speech_s11_2_dc11() -> None:
    # "12" is spoken 1.30-1.70, then a short gap, then "geese": its wildcard spans that speech, so cue 1
    # starts where "twelve" does, not after the gap nearest to its first aligned word.
    speech = audio(3.0, [(0.20, 0.90), (1.30, 1.70), (1.80, 2.80)])
    words = {
        (0, 0): (0.24, 0.40),
        (0, 1): (0.46, 0.86),
        (1, 0): (1.32, 1.68),
        (1, 1): (1.86, 2.30),
        (1, 2): (2.36, 2.74),
    }
    a = _resolve(("It rained.", "12 geese came."), speech, words)
    assert cue_times(a) == [(0.2, 0.9), (1.3, 2.8)]
    assert [(w.text, w.start_s) for w in a.cues[1].words] == [("12", None), ("geese", 1.86), ("came", 2.36)]


def test_left_out_word_at_a_cue_tail_ends_the_cue_after_its_speech_s11_2_dc11() -> None:
    # "12" is spoken 1.30-1.60 after a gap, and its wildcard spans it: cue 0 ends after that speech.
    speech = audio(3.0, [(0.20, 1.04), (1.30, 1.60), (1.80, 2.60)])
    words = {
        (0, 0): (0.24, 0.36),
        (0, 1): (0.42, 0.70),
        (0, 2): (0.76, 1.00),
        (0, 3): (1.32, 1.58),
        (1, 0): (1.86, 2.10),
        (1, 1): (2.16, 2.56),
    }
    a = _resolve(("The count was 12.", "Then sun."), speech, words)
    assert cue_times(a) == [(0.2, 1.6), (1.8, 2.6)]
    assert a.cues[0].words[-1].start_s is None


def test_alignment_error_unplaces_every_cue_s11_2() -> None:
    seg = segment(*TWO_CUES)
    t = ALIGNER.build_transcript(seg, [])
    details = {"reason": "too_short", "frames": 4, "tokens": len(t.tokens), "repeats": 0}
    a = ALIGNER.resolve(t, None, TWO_AUDIO, RATE, [], None, error=details)
    assert cue_times(a) == [(None, None), (None, None)]
    assert all(w.start_s is None for c in a.cues for w in c.words)
    error = a.flags[0]
    assert (error.code, error.severity, error.retake_trigger, error.details) == (
        codes.ALIGNMENT_ERROR,
        "fail",
        True,
        details,
    )
    assert codes_of(a)[1:] == [(codes.CUE_UNALIGNED, 0), (codes.CUE_UNALIGNED, 1)]
    assert all(f.retake_trigger for f in a.flags)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda r: r | {"revision": "0" * 40}, "model_mismatch"),
        (lambda r: r | {"model": "facebook/wav2vec2-base-960h"}, "model_mismatch"),
        (lambda r: r | {"spans": r["spans"][:-1]}, "span_mismatch"),
        (lambda r: r | {"num_frames": 5}, "too_short"),
        (lambda r: r | {"spans": [s | {"end_frame": 10_000} for s in r["spans"]]}, "bad_reply"),
        (lambda r: r | {"spans": [s | {"score": 1.5} for s in r["spans"]]}, "bad_reply"),
        (lambda r: r | {"frame_s": 0.0}, "bad_reply"),
        (lambda r: r | {"ok": False, "error": {"code": "ALIGNMENT_ERROR", "message": "x"}}, "worker_error"),
    ],
)
def test_reply_that_does_not_fit_is_an_alignment_error_s11_2(change, reason) -> None:
    t = ALIGNER.build_transcript(segment(*TWO_CUES), [])
    r = change(reply(t, TWO_WORDS, samples=TWO_AUDIO.shape[0]))
    a = ALIGNER.resolve(t, r, TWO_AUDIO, RATE, [], None)
    assert a.flags[0].code == codes.ALIGNMENT_ERROR
    assert a.flags[0].details is not None and a.flags[0].details["reason"] == reason
    assert cue_times(a) == [(None, None), (None, None)]


@pytest.mark.parametrize("code", ["INTERNAL", "GPU_OOM", "NOT_LOADED", "UNSUPPORTED_AUDIO", None])
def test_a_worker_failure_other_than_alignment_error_is_no_verdict_s11_2(code: str | None) -> None:
    # Regression (review F2): any ok: false reply became ALIGNMENT_ERROR (fail, a retake trigger). Only the
    # aligner's own ALIGNMENT_ERROR says anything about the take; anything else raises, as the worker client
    # does, and the caller records QA_UNAVAILABLE (section 14).
    t = ALIGNER.build_transcript(segment(*TWO_CUES), [])
    error = {"message": "it broke", "details": {"type": "RuntimeError"}} | ({"code": code} if code else {})
    r = cast(AlignReply, reply(t, TWO_WORDS, samples=TWO_AUDIO.shape[0]) | {"ok": False, "error": error})
    with pytest.raises(WorkerFailure) as caught:
        ALIGNER.resolve(t, r, TWO_AUDIO, RATE, [], None)
    assert caught.value.code == (code or "INTERNAL")
    if code:
        assert (caught.value.message, caught.value.details) == ("it broke", {"type": "RuntimeError"})


def test_the_workers_alignment_error_reply_is_the_verdict_with_its_details_s11_2() -> None:
    t = ALIGNER.build_transcript(segment(*TWO_CUES), [])
    details = {"reason": "too_short", "frames": 4, "tokens": len(t.tokens), "repeats": 0}
    error = {"code": "ALIGNMENT_ERROR", "message": "too short", "details": details}
    r = cast(AlignReply, reply(t, TWO_WORDS, samples=TWO_AUDIO.shape[0]) | {"ok": False, "error": error})
    a = ALIGNER.resolve(t, r, TWO_AUDIO, RATE, [], None)
    assert (a.flags[0].code, a.flags[0].severity, a.flags[0].retake_trigger) == (codes.ALIGNMENT_ERROR, "fail", True)
    assert a.flags[0].details == details
    assert cue_times(a) == [(None, None), (None, None)]


def test_no_alignable_word_needs_no_reply_and_is_not_an_alignment_error_s11_2_dc12() -> None:
    t = ALIGNER.build_transcript(segment("12.", "7, 8."), [])
    assert t.tokens == ("*", "|", "*")
    a = ALIGNER.resolve(t, None, TWO_AUDIO, RATE, [], None)
    assert codes_of(a) == [(codes.CUE_UNALIGNED, 0), (codes.CUE_UNALIGNED, 1)]
    assert all(f.details == {"reason": codes.CUE_NO_ALIGNABLE_WORDS} for f in a.flags)
    assert not any(f.retake_trigger for f in a.flags)


def test_a_wildcards_score_is_not_confidence_s11_2_dc11() -> None:
    speech = audio(3.0, [(0.20, 1.04), (1.30, 1.60), (1.80, 2.60)])
    words = {(0, 0): (0.24, 0.36), (0, 1): (0.42, 0.70), (0, 2): (0.76, 1.00), (0, 3): (1.32, 1.58)}
    words |= {(1, 0): (1.86, 2.10), (1, 1): (2.16, 2.56)}
    t = ALIGNER.build_transcript(segment("The count was 12.", "Then sun."), [])
    r = reply(t, words, samples=speech.shape[0], default_score=0.9)
    star = t.tokens.index("*")
    r["spans"][star] = {**r["spans"][star], "score": 0.05}
    a = ALIGNER.resolve(t, r, speech, RATE, [], None)
    assert [c.confidence for c in a.cues] == [0.9, 0.9]
    assert a.flags == ()


def test_times_are_delivery_file_seconds_rounded_to_the_millisecond_s11_2() -> None:
    a = _resolve(TWO_CUES, TWO_AUDIO, TWO_WORDS)
    times = [t for c in a.cues for w in c.words for t in (w.start_s, w.end_s)]
    assert all(t is not None and t == round(t, 3) for t in times)
    assert a.cues[1].words[1].start_s == 2.26  # frame 113 × 0.02 s, not 2.2600000000000002


def test_measured_error_and_identity_are_carried_s11_2() -> None:
    measured = MeasuredError(p50_s=0.03, p95_s=0.09, n=60, benchmark="alignment-en.v1", by_kind=None)
    t = ALIGNER.build_transcript(segment(*TWO_CUES), [])
    a = ALIGNER.resolve(t, reply(t, TWO_WORDS, samples=TWO_AUDIO.shape[0]), TWO_AUDIO, RATE, [], measured)
    assert (a.method, a.model, a.revision, a.device) == (ALIGNMENT_METHOD, MODEL_ALIGNER, REVISION, "cpu")
    assert a.measured_error == measured
    assert a.cross_check.model == MODEL_ASR


def test_alignment_validates_against_its_published_schema_s11_2() -> None:
    validator = Draft202012Validator(record_schema(Alignment))
    for a in (
        _resolve(TWO_CUES, TWO_AUDIO, TWO_WORDS),
        _resolve(THREE_CUES, THREE_AUDIO, THREE_WORDS, scores={1: 0.2}),
        ALIGNER.resolve(ALIGNER.build_transcript(segment(*TWO_CUES), []), None, TWO_AUDIO, RATE, [], None),
    ):
        data = to_json(a)
        validator.validate(data)
        assert from_json(Alignment, data) == a


def test_resolve_is_deterministic_s11_2() -> None:
    assert _resolve(THREE_CUES, THREE_AUDIO, THREE_WORDS, scores={1: 0.2}) == _resolve(
        THREE_CUES, THREE_AUDIO, THREE_WORDS, scores={1: 0.2}
    )


def test_aligner_is_an_aligner_core_on_the_cpu_s11_2() -> None:
    assert isinstance(ALIGNER, AlignerCore)
    assert (ALIGNER.model, ALIGNER.revision, ALIGNER.device) == (MODEL_ALIGNER, REVISION, "cpu")
    with pytest.raises(ValueError, match="CPU"):
        CtcAligner(revision=REVISION, device="cuda:0")
    with pytest.raises(ValueError, match="pinned"):
        CtcAligner(revision="")


def test_method_id_names_the_model_revision_and_every_setting_s11_2() -> None:
    method = ALIGNER.method_id
    assert method.startswith(f"ctc-snap/wav2vec2-large-960h-lv60-self@{REVISION}+p")
    assert CtcAligner(revision=REVISION).method_id == method
    changed = [
        AlignerParams(unplaced_below=0.4),
        AlignerParams(low_confidence_below=0.7),
        AlignerParams(disagree_above_s=0.2),
        AlignerParams(snap_tolerance_s=0.04),
        AlignerParams(snap_reach_start_s=0.2),
        AlignerParams(snap_reach_end_s=0.4),
        AlignerParams(pauses=dataclasses.replace(AlignerParams().pauses, silence_below_db=40.0)),
    ]
    ids = {CtcAligner(revision=REVISION, params=p).method_id for p in changed}
    assert len(ids) == len(changed) and method not in ids
    assert CtcAligner(revision="0" * 40).method_id != method


def test_from_config_takes_every_threshold_s11_2() -> None:
    config = AlignmentConfig(disagree_threshold_s=0.1, low_confidence_below=0.8, unplaced_below=0.4)
    aligner = CtcAligner.from_config(config, revision=REVISION)
    params = aligner.params
    assert (params.disagree_above_s, params.low_confidence_below, params.unplaced_below) == (0.1, 0.8, 0.4)
    assert (aligner.model, aligner.device) == (MODEL_ALIGNER, "cpu")
    assert CtcAligner.from_config(AlignmentConfig(), revision=REVISION).method_id == ALIGNER.method_id


@pytest.mark.parametrize(
    "bad",
    [
        {"unplaced_below": 0.8, "low_confidence_below": 0.7},
        {"unplaced_below": -0.1},
        {"low_confidence_below": 1.5},
        {"disagree_above_s": -1.0},
        {"snap_reach_start_s": -0.02},
        {"snap_reach_end_s": -0.02},
    ],
)
def test_thresholds_out_of_order_are_refused_s11_2(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(AlignerParams(), **bad)


def test_hints_are_the_segments_hints_used_s11_2() -> None:
    hint = Hint(term="turns", align_as="TERNS")
    a = _resolve(TWO_CUES, TWO_AUDIO, TWO_WORDS, hints=[hint])
    assert a.cues[0].words[2].text == "turns"  # the spoken text, whatever the aligner spelled


def test_error_stats_type_is_the_contracts_s11_2() -> None:
    stats = ErrorStats(p50_s=0.02, p95_s=0.05, n=30)
    measured = MeasuredError(p50_s=0.02, p95_s=0.05, n=30, benchmark="alignment-en.v1", by_kind={"pause": stats})
    t = ALIGNER.build_transcript(segment(*TWO_CUES), [])
    a = ALIGNER.resolve(t, reply(t, TWO_WORDS, samples=TWO_AUDIO.shape[0]), TWO_AUDIO, RATE, [], measured)
    Draft202012Validator(record_schema(Alignment)).validate(to_json(a))
