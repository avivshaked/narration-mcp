"""The transcript check (design section 3.2): the clip's transcript must be what the clip says.

In ICL mode Qwen reads the clip together with its transcript, and a transcript that does not match the clip
makes the reference bleed into the takes. So before anything is rendered, Whisper transcribes the clip and
the transcript the caller sent is compared with what it heard.

**The comparison** is QA's text match without hints (section 11.1 step 4): both sides are read by the number
reader (``names.NUMBER_READER``: Whisper's English normaliser, phrase by phrase, apostrophes removed), aligned
word by word, and the word errors counted against the transcript's words.

**The rule** (ASSUME, until measured on designed clips in WP40): the transcript does not match when the check
meets QA's fail rule for a take, a word error rate above 0.06 with at least 2 word errors (``QaProfile``).
One word Whisper spells differently passes; a transcript of another clip, a missing sentence or a changed
phrase does not. A mismatch refuses the measurement (``REF_TEXT_MISMATCH``), with what was heard.
"""

from __future__ import annotations

from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.models import TranscriptCheck
from narration.qa.normaliser import NumberReader, remove_apostrophes
from narration.qa.profile import DEFAULT_PROFILE, QaProfile
from narration.qa.textmatch import align

DECIMALS = 6


def _read(reader: NumberReader, text: str) -> list[str]:
    return reader.read(remove_apostrophes(text)).split()


def word_errors(reference: str, heard: str, reader: NumberReader) -> tuple[int, int]:
    """(word errors, reference words): substitutions, deletions and insertions of ``heard`` against
    ``reference``, both read by the number reader."""
    ref, hyp = _read(reader, reference), _read(reader, heard)
    errors = 0
    for chunk in align(ref, hyp):
        if chunk.kind in ("substitute", "delete"):
            errors += chunk.ref[1] - chunk.ref[0]
        elif chunk.kind == "insert":
            errors += chunk.hyp[1] - chunk.hyp[0]
        # a substitution of unequal lengths never happens in a word alignment: jiwer pairs words one to one
    return errors, len(ref)


def check_transcript(
    transcript: str, heard: str, reader: NumberReader, profile: QaProfile = DEFAULT_PROFILE
) -> tuple[TranscriptCheck, int, int]:
    """The check of ``transcript`` against what Whisper ``heard`` in the clip: the record, the word errors, and
    the transcript's words (the module docstring)."""
    errors, n_ref = word_errors(transcript, heard, reader)
    wer = errors / n_ref if n_ref else (1.0 if errors else 0.0)
    mismatch = wer > profile.wer_fail_above and errors >= profile.wer_fail_min_errors
    if n_ref == 0:
        mismatch = True  # nothing the number reader can read: no word of it can be confirmed
    return TranscriptCheck(heard=heard, wer=round(wer, DECIMALS), ok=not mismatch), errors, n_ref


def mismatch_error(
    check: TranscriptCheck, errors: int, n_ref: int, profile: QaProfile = DEFAULT_PROFILE
) -> NarrationError:
    """``REF_TEXT_MISMATCH`` for a transcript the clip does not say, with what Whisper heard."""
    return NarrationError(
        codes.REF_TEXT_MISMATCH,
        f"the transcript does not match the clip: {errors} word error(s) in {n_ref} words "
        f"(word error rate {check.wer:.3f}; refused above {profile.wer_fail_above} with at least "
        f"{profile.wer_fail_min_errors} errors)",
        field="voice.transcript",
        hint=(
            "Send the exact words spoken in the clip, as design_voice returned them; 'details.heard' is what the "
            "speech recogniser heard. Nothing was rendered."
        ),
        details={"heard": check.heard, "wer": check.wer, "word_errors": errors, "words": n_ref},
        retryable=False,
    )


__all__ = ["check_transcript", "mismatch_error", "word_errors"]
