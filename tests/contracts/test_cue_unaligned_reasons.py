"""Every ``CUE_UNALIGNED`` carries a ``details.reason`` from ``codes.CUE_UNALIGNED_REASONS`` (design section 11.2
step 7, DC-12), and only ``no_alignable_words`` is not a retake trigger.

The second test drives every way the service makes a ``CUE_UNALIGNED``: the aligner's three (a cue under
``unplaced_below``, a take it could not align, a text with no letter) and QA's fallback (a cue the alignment
left without times and without a flag), each also through QA's ``alignment_flags``, as the scorer runs it.
"""

from __future__ import annotations

import dataclasses

from narration.align import CtcAligner
from narration.contracts import codes
from narration.contracts.models import Flag
from narration.qa.checks import alignment_flags
from tests.align._support import RATE, REVISION, audio, reply, segment

CUES = ("Rain came.", "Then sun.")
AUDIO = audio(3.0, [(0.20, 1.20), (1.80, 2.80)])
WORDS = {(0, 0): (0.24, 0.60), (0, 1): (0.66, 1.16), (1, 0): (1.84, 2.20), (1, 1): (2.26, 2.76)}


def test_cue_unaligned_has_four_reasons_and_only_no_alignable_words_is_exempt_dc12() -> None:
    reasons = frozenset({"no_alignable_words", "low_confidence", "alignment_error", "not_placed"})
    assert reasons == codes.CUE_UNALIGNED_REASONS
    assert (codes.CUE_UNPLACED_LOW_CONFIDENCE, codes.CUE_UNPLACED_ALIGNMENT_ERROR, codes.CUE_NOT_PLACED) == (
        "low_confidence",
        "alignment_error",
        "not_placed",
    )
    for reason in codes.CUE_UNALIGNED_REASONS:
        expected = reason != codes.CUE_NO_ALIGNABLE_WORDS
        assert codes.is_retake_trigger(codes.CUE_UNALIGNED, "warn", {"reason": reason}) is expected, reason


def test_every_cue_unaligned_the_aligner_or_qa_emits_has_a_known_reason_dc12() -> None:
    aligner = CtcAligner(revision=REVISION)
    spoken = segment(*CUES)
    transcript = aligner.build_transcript(spoken, [])
    low = aligner.resolve(
        transcript, reply(transcript, WORDS, samples=AUDIO.shape[0], scores={1: 0.2}), AUDIO, RATE, [], None
    )
    failed = aligner.resolve(transcript, None, AUDIO, RATE, [], None)
    digits = segment("12.", "7, 8.")
    no_letters = aligner.resolve(aligner.build_transcript(digits, []), None, AUDIO, RATE, [], None)
    unflagged = dataclasses.replace(low, flags=tuple(f for f in low.flags if f.code != codes.CUE_UNALIGNED))

    emitted: list[Flag] = []
    for alignment, seg in ((low, spoken), (failed, spoken), (no_letters, digits)):
        emitted += alignment.flags
        emitted += alignment_flags(alignment, seg)
    emitted += alignment_flags(unflagged, spoken)
    unaligned = [f for f in emitted if f.code == codes.CUE_UNALIGNED]
    reasons = {(f.details or {}).get("reason") for f in unaligned}
    assert reasons == codes.CUE_UNALIGNED_REASONS  # every reason is made somewhere, and nothing else is
    for flag in unaligned:
        assert flag.details is not None and flag.details["reason"] in codes.CUE_UNALIGNED_REASONS, flag
        assert flag.retake_trigger is (flag.details["reason"] != codes.CUE_NO_ALIGNABLE_WORDS), flag
