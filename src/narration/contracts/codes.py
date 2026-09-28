"""Every error and flag code, with its retryability, severities and retake rule (design section 14).

The two tables of section 14 are copied here verbatim in meaning, plus design change DC-2 (plan.md
section 1.5): the retryable codes ``QUEUE_FULL`` and ``RATE_LIMITED``, and ``retry_after_s`` on every
retryable error.

Use the constants (``codes.VOICE_NOT_MEASURED``) or look a code up in ``ERRORS`` / ``FLAGS``; never type a
code as a string literal elsewhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

from .names import Severity

RetakeRule = Literal["always", "at_fail", "never"]
"""When a flag triggers an automatic retake (section 11.1, the "R" column of section 14):
``always`` at any severity, ``at_fail`` only at severity ``fail``, ``never``."""


@dataclass(frozen=True, slots=True)
class ErrorCode:
    """A tool execution error code (section 14).

    ``retryable`` is None for ``INTERNAL``, whose retryability is decided per instance ("maybe").
    ``hint`` is the default hint: what the caller should do next.
    """

    code: str
    retryable: bool | None
    meaning: str
    hint: str


@dataclass(frozen=True, slots=True)
class FlagCode:
    """A flag code reported inside results (section 14). ``severities`` lists every severity it may take."""

    code: str
    severities: tuple[Severity, ...]
    retake: RetakeRule
    meaning: str


# ---------------------------------------------------------------- error codes
INVALID_ARGUMENT: Final = "INVALID_ARGUMENT"
LIMIT_EXCEEDED: Final = "LIMIT_EXCEEDED"
NOT_FOUND: Final = "NOT_FOUND"
PATH_NOT_ALLOWED: Final = "PATH_NOT_ALLOWED"
UNSUPPORTED_AUDIO: Final = "UNSUPPORTED_AUDIO"
VOICE_FILE_MISMATCH: Final = "VOICE_FILE_MISMATCH"
VOICE_NOT_SYNTHETIC: Final = "VOICE_NOT_SYNTHETIC"
VOICE_NOT_MEASURED: Final = "VOICE_NOT_MEASURED"
REF_TEXT_MISMATCH: Final = "REF_TEXT_MISMATCH"
ENGINE_CHANGED: Final = "ENGINE_CHANGED"
CONTROL_UNSUPPORTED: Final = "CONTROL_UNSUPPORTED"
TEXT_REFUSED: Final = "TEXT_REFUSED"
ENGINE_DRIFT: Final = "ENGINE_DRIFT"
BACKEND_NOT_INSTALLED: Final = "BACKEND_NOT_INSTALLED"
DAEMON_UNAVAILABLE: Final = "DAEMON_UNAVAILABLE"
GPU_UNAVAILABLE: Final = "GPU_UNAVAILABLE"
STORE_FULL: Final = "STORE_FULL"
JOB_NOT_CANCELLABLE: Final = "JOB_NOT_CANCELLABLE"
INTERNAL: Final = "INTERNAL"
QUEUE_FULL: Final = "QUEUE_FULL"
RATE_LIMITED: Final = "RATE_LIMITED"

ERRORS: Final[dict[str, ErrorCode]] = {
    e.code: e
    for e in (
        ErrorCode(
            INVALID_ARGUMENT,
            False,
            "a schema or semantic failure: an unknown field, duplicate segment ids, text that is not the join of "
            "the cues, text_mode 'written', an exact span that cuts a word",
            "Fix the argument named in 'field' and send the request again.",
        ),
        ErrorCode(
            LIMIT_EXCEEDED,
            False,
            "the request exceeds a size limit (segments, cues, characters, hints)",
            "Split the request; get_server_status lists the limits.",
        ),
        ErrorCode(
            NOT_FOUND,
            False,
            "a job, design, take or measurement that does not exist (e.g. expired after the retention period)",
            "Send the original request again; work that left the cache is done again.",
        ),
        ErrorCode(
            PATH_NOT_ALLOWED,
            False,
            "a path that is not absolute, resolves to a network share or a device path, or is not a regular file",
            "Give the absolute path of a regular file on a local drive.",
        ),
        ErrorCode(
            UNSUPPORTED_AUDIO,
            False,
            "not a WAV the service reads, or longer than 30 s / larger than 20 MB for a voice clip",
            "Send a mono or stereo WAV; a voice clip must be at most 30 s and 20 MB.",
        ),
        ErrorCode(
            VOICE_FILE_MISMATCH,
            False,
            "the file's sha256 is not the one sent; the clip changed or the path is wrong",
            "Check the clip's path, and send the sha256 of the file that is there.",
        ),
        ErrorCode(
            VOICE_NOT_SYNTHETIC,
            False,
            "the clip is neither one the service designed nor one the operator allowlisted",
            "Use a synthetic clip the operator has allowlisted (or, where design_voice is available, one it "
            "designed), or ask the operator to allow this clip. Only a person allows a clip, "
            "at this machine, after confirming that it is synthetic; a calling agent must not run the command. "
            "The operator runs <venv_python> -m narration.admin --config <service_root>/narration.toml voices "
            "allow <clip.wav>, which adds its sha256 to [voices] allow_sha256. The daemon and the MCP server read "
            "that list only when they start, so the operator then stops the daemon first (the same command with "
            "daemon stop; with [daemon] autostart on, the next submission starts it again, else daemon start) "
            "and then reconnects the client to the server (in Claude Code, /mcp).",
        ),
        ErrorCode(
            VOICE_NOT_MEASURED,
            False,
            "no measurement for this voice under the current engine",
            "Measure the clip first with measure_voice (a heavy GPU job; check get_server_status first).",
        ),
        ErrorCode(
            REF_TEXT_MISMATCH,
            False,
            "measuring found that the transcript does not match the clip",
            "Send the exact words spoken in the clip, copied from the record kept with it (where design_voice is "
            "available, the candidate's transcript), never retyped.",
        ),
        ErrorCode(
            ENGINE_CHANGED,
            False,
            "expect_engine_profile differs from the service's engine profile",
            "Accept the new engine profile hash, or ask the operator to restore the old one.",
        ),
        ErrorCode(
            CONTROL_UNSUPPORTED,
            False,
            "a control no current engine supports: pace, context_before, context_after",
            "Remove the control; Qwen Base takes no pace or unspoken context (design section 3.3).",
        ),
        ErrorCode(
            TEXT_REFUSED,
            False,
            "markup characters in the text, or strict_text with text warnings left; every offender is listed",
            "Remove the characters listed in 'details', or reword the flagged tokens as spoken words.",
        ),
        ErrorCode(
            ENGINE_DRIFT,
            False,
            "the worker's fingerprint does not match the pinned engine profile, or the canary is below threshold",
            "Ask the operator to restore the pinned environment or re-pin the engine (narration-admin engine).",
        ),
        ErrorCode(
            BACKEND_NOT_INSTALLED,
            False,
            "model weights or a worker environment are missing",
            "Ask the operator to run narration-admin install and narration-admin doctor.",
        ),
        ErrorCode(
            DAEMON_UNAVAILABLE,
            True,
            "the daemon cannot be started detached (breakaway refused), or it is stopping",
            "Run 'narration-admin daemon start' in a terminal, then send the request again.",
        ),
        ErrorCode(
            GPU_UNAVAILABLE,
            True,
            "the wait for free GPU memory timed out",
            "Wait at least retry_after_s, then send the identical request again.",
        ),
        ErrorCode(
            STORE_FULL,
            True,
            "free disk space is below the minimum",
            "Free disk space (or ask the operator to run narration-admin gc), then send the request again.",
        ),
        ErrorCode(
            JOB_NOT_CANCELLABLE,
            False,
            "the job is already finished, failed or cancelled",
            "Nothing to cancel; read the job with get_job or get_results.",
        ),
        ErrorCode(
            INTERNAL,
            None,
            "a bug in the service; the log path is included",
            "Report it with the log path in 'details'; retry only if 'retryable' is true.",
        ),
        ErrorCode(
            QUEUE_FULL,
            True,
            "the job queue is full (DC-2)",
            "Wait at least retry_after_s, add your own jitter, then send the identical request again.",
        ),
        ErrorCode(
            RATE_LIMITED,
            True,
            "too many submissions in the rate window (DC-2)",
            "Wait at least retry_after_s, add your own jitter, then send the identical request again.",
        ),
    )
}

# ---------------------------------------------------------------- flag codes
WRITTEN_FORM_TOKEN: Final = "WRITTEN_FORM_TOKEN"
TERM_SPLIT_ACROSS_CUES: Final = "TERM_SPLIT_ACROSS_CUES"
SEGMENT_TOO_LONG: Final = "SEGMENT_TOO_LONG"
WER_HIGH: Final = "WER_HIGH"
EXACT_SPAN_MISMATCH: Final = "EXACT_SPAN_MISMATCH"
TERM_UNVERIFIED: Final = "TERM_UNVERIFIED"
SPK_SIM_LOW: Final = "SPK_SIM_LOW"
SPK_OUTLIER: Final = "SPK_OUTLIER"
PACE_FAST: Final = "PACE_FAST"
PACE_SLOW: Final = "PACE_SLOW"
HEAD_INSERTION: Final = "HEAD_INSERTION"
END_INSERTION: Final = "END_INSERTION"
SILENCE_LONG: Final = "SILENCE_LONG"
CLIPPING: Final = "CLIPPING"
SIGNAL_INVALID: Final = "SIGNAL_INVALID"
TOKEN_CAP_HIT: Final = "TOKEN_CAP_HIT"
CUE_UNALIGNED: Final = "CUE_UNALIGNED"
CUE_LOW_CONFIDENCE: Final = "CUE_LOW_CONFIDENCE"
CUE_ALIGNMENT_DISAGREE: Final = "CUE_ALIGNMENT_DISAGREE"
CUE_BOUNDARY_NO_PAUSE: Final = "CUE_BOUNDARY_NO_PAUSE"
ALIGNMENT_ERROR: Final = "ALIGNMENT_ERROR"
FIT_TIGHT: Final = "FIT_TIGHT"
OVER_SCENE: Final = "OVER_SCENE"
CANARY_MISMATCH: Final = "CANARY_MISMATCH"
CLIP_TOO_LONG: Final = "CLIP_TOO_LONG"
LOUDNESS_UNDER_TARGET: Final = "LOUDNESS_UNDER_TARGET"
GAIN_HIGH: Final = "GAIN_HIGH"
RETAKEN: Final = "RETAKEN"
RENDER_FAILED: Final = "RENDER_FAILED"
WORKER_CRASHED: Final = "WORKER_CRASHED"
GPU_OOM: Final = "GPU_OOM"
QA_UNAVAILABLE: Final = "QA_UNAVAILABLE"
CANCELLED: Final = "CANCELLED"

FLAGS: Final[dict[str, FlagCode]] = {
    f.code: f
    for f in (
        FlagCode(
            WRITTEN_FORM_TOKEN,
            ("warn", "info"),
            "never",
            "text check: a digit, symbol or unit-like token in spoken text; info for a lone letter",
        ),
        FlagCode(
            TERM_SPLIT_ACROSS_CUES,
            ("warn",),
            "never",
            "text check: a term would match only across a cue boundary, so no hint was applied",
        ),
        FlagCode(
            SEGMENT_TOO_LONG,
            ("warn",),
            "never",
            "the segment is longer than the voice's reliable length; it is still rendered",
        ),
        FlagCode(WER_HIGH, ("warn", "fail"), "at_fail", "wer_adj above the threshold, with the word-count rule"),
        FlagCode(
            EXACT_SPAN_MISMATCH,
            ("fail",),
            "always",
            "a different value or different words inside a span the caller marked exact",
        ),
        FlagCode(TERM_UNVERIFIED, ("warn",), "never", "ASR did not match a hinted term"),
        FlagCode(
            SPK_SIM_LOW,
            ("warn", "fail"),
            "at_fail",
            "similarity to the voice's anchor below the measured warn threshold / the floor",
        ),
        FlagCode(
            SPK_OUTLIER,
            ("info",),
            "never",
            "a suggested take stands apart from the rest of the request (a report, never a verdict)",
        ),
        FlagCode(PACE_FAST, ("warn", "fail"), "at_fail", "faster than the voice's pace curve at this length"),
        FlagCode(PACE_SLOW, ("warn", "fail"), "at_fail", "slower than the voice's pace curve at this length"),
        FlagCode(HEAD_INSERTION, ("warn", "fail"), "always", "words before cue 0, or reference bleed"),
        FlagCode(END_INSERTION, ("warn", "fail"), "at_fail", "words after the last cue"),
        FlagCode(SILENCE_LONG, ("warn", "fail"), "at_fail", "longest internal silence"),
        FlagCode(CLIPPING, ("warn",), "never", "raw samples at full scale"),
        FlagCode(
            SIGNAL_INVALID,
            ("warn", "fail"),
            "at_fail",
            "non-finite samples (fail) or a DC offset (warn) in the raw take (section 11.1 step 1; plan DC-5)",
        ),
        FlagCode(TOKEN_CAP_HIT, ("fail",), "always", "generation stopped at max_new_tokens"),
        FlagCode(CUE_UNALIGNED, ("warn",), "always", "cue not placed; times null; never interpolated"),
        FlagCode(CUE_LOW_CONFIDENCE, ("warn",), "never", "alignment posterior below threshold"),
        FlagCode(
            CUE_ALIGNMENT_DISAGREE, ("warn",), "never", "CTC vs Whisper boundary differ by more than the threshold"
        ),
        FlagCode(CUE_BOUNDARY_NO_PAUSE, ("info",), "never", "no silence gap at a cue boundary"),
        FlagCode(
            ALIGNMENT_ERROR,
            ("fail",),
            "always",
            "the aligner raised or could not run (frames < tokens + repeats)",
        ),
        FlagCode(FIT_TIGHT, ("warn",), "never", "only with scene_seconds: slack under 1.0 s; reported, not remedied"),
        FlagCode(OVER_SCENE, ("warn",), "never", "only with scene_seconds: over budget; reported, not remedied"),
        FlagCode(
            CANARY_MISMATCH,
            ("info",),
            "never",
            "canary hash differed but similarity passed (bit_exact tier only)",
        ),
        FlagCode(
            CLIP_TOO_LONG,
            ("fail",),
            "always",
            "a designed candidate is longer than [limits] max_clip_seconds, so measure_voice would refuse it",
        ),
        FlagCode(LOUDNESS_UNDER_TARGET, ("info",), "never", "the true-peak ceiling lowered the gain"),
        FlagCode(GAIN_HIGH, ("info",), "never", "gain above +12 dB"),
        FlagCode(RETAKEN, ("info",), "never", "an earlier attempt of this slot failed (listed)"),
        FlagCode(RENDER_FAILED, ("error",), "never", "segment-level execution problem: the render failed"),
        FlagCode(WORKER_CRASHED, ("error",), "never", "segment-level execution problem: a worker crashed"),
        FlagCode(GPU_OOM, ("error",), "never", "segment-level execution problem: out of GPU memory after one retry"),
        FlagCode(QA_UNAVAILABLE, ("error",), "never", "segment-level execution problem: QA could not run"),
        FlagCode(CANCELLED, ("error",), "never", "segment-level execution problem: the job was cancelled"),
    )
}


CUE_NO_ALIGNABLE_WORDS: Final = "no_alignable_words"
"""``CUE_UNALIGNED``'s ``details.reason`` for a cue whose text has no word the aligner can place (DC-12). The
only reason that is not a retake trigger: no retake can place such a cue."""
CUE_UNPLACED_LOW_CONFIDENCE: Final = "low_confidence"
"""``CUE_UNALIGNED``'s ``details.reason`` for a cue the aligner placed with a confidence under
``unplaced_below`` (section 11.2 step 7): its times are dropped, never interpolated."""
CUE_UNPLACED_ALIGNMENT_ERROR: Final = "alignment_error"
"""``CUE_UNALIGNED``'s ``details.reason`` for every cue of a take the aligner could not align at all (the take
gets ``ALIGNMENT_ERROR``, section 11.2 step 3)."""
CUE_NOT_PLACED: Final = "not_placed"
"""``CUE_UNALIGNED``'s ``details.reason`` when QA finds a cue without times and the aligner gave no
``CUE_UNALIGNED`` for it (QA's fallback, section 11.2 step 7): QA cannot say why, only that it is unplaced."""
CUE_UNALIGNED_REASONS: Final = frozenset(
    {CUE_NO_ALIGNABLE_WORDS, CUE_UNPLACED_LOW_CONFIDENCE, CUE_UNPLACED_ALIGNMENT_ERROR, CUE_NOT_PLACED}
)
"""Every ``details.reason`` a ``CUE_UNALIGNED`` may carry. That every ``CUE_UNALIGNED`` carries one of these
is this module's rule, from contracts 1.6.3 on; DC-12 names only ``CUE_NO_ALIGNABLE_WORDS``."""


def is_retake_trigger(code: str, severity: Severity, details: Mapping[str, object] | None = None) -> bool:
    """Whether a flag triggers an automatic retake (section 11.1).

    Any fail-severity flag does, and so do ``CUE_UNALIGNED`` and ``HEAD_INSERTION`` at any severity, except
    a ``CUE_UNALIGNED`` whose ``details.reason`` is ``no_alignable_words`` (DC-12): the cue's text gives the
    aligner nothing to place, so every retake would fail the same way.
    """
    if severity == "fail":
        return True
    if code == CUE_UNALIGNED and details is not None and details.get("reason") == CUE_NO_ALIGNABLE_WORDS:
        return False
    flag = FLAGS.get(code)
    return flag is not None and flag.retake == "always"


def error_code(code: str) -> ErrorCode:
    """Look up an error code; raises KeyError for a code the design does not define."""
    return ERRORS[code]


def flag_code(code: str) -> FlagCode:
    """Look up a flag code; raises KeyError for a code the design does not define."""
    return FLAGS[code]
