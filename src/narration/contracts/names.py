"""Names the design fixes: ids, schema ids, versions, tools, resources, prompts and enumerations.

Every other module takes these from here and never retypes them (AGENTS.md section 6). Section numbers
refer to ``docs/design.md``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Final, Literal, get_args

from narration_worker.protocol import WorkerRole as WorkerRole

# ---------------------------------------------------------------- the server (sections 5, 7)
SERVER_NAME: Final = "narration"
SPEC_REVISION: Final = "2026-07-28"

# ---------------------------------------------------------------- ids (sections 5, 6, 15)
RENDER_ID_PREFIX: Final = "rn_"
TAKE_ID_PREFIX: Final = "tk_"
ANALYSIS_ID_PREFIX: Final = "an_"
JOB_ID_PREFIX: Final = "job_"
ID_HEX_CHARS: Final = 16
"""A content id is its prefix plus the first 16 hex characters of its key's sha256 (section 6)."""
HASH_PREFIX: Final = "sha256:"
"""Keys and hashes the service reports carry this prefix (``voice_hash``, ``render_key``, …)."""

IdKind = Literal["job_id", "design_id", "render_id", "take_id", "analysis_id", "voice_hash"]
_ULID: Final = "[0-7][0-9A-HJKMNP-TV-Z]{25}"
ID_PATTERNS: Final[dict[IdKind, str]] = {
    "job_id": JOB_ID_PREFIX + _ULID,
    "design_id": _ULID,
    "render_id": RENDER_ID_PREFIX + f"[0-9a-f]{{{ID_HEX_CHARS}}}",
    "take_id": TAKE_ID_PREFIX + f"[0-9a-f]{{{ID_HEX_CHARS}}}",
    "analysis_id": ANALYSIS_ID_PREFIX + f"[0-9a-f]{{{ID_HEX_CHARS}}}",
    "voice_hash": HASH_PREFIX + "[0-9a-f]{64}",
}
"""The shape of every id a caller can send back (section 6), without anchors. Job and design ids are ULIDs
(Crockford base32, 26 characters, first one 0-7); content ids are a prefix plus 16 hex; hashes are
``sha256:`` plus 64 hex. The store, the front end and the tool schemas all use these, so an id is
refused at the edge before anything builds a path from it."""


def is_id(kind: IdKind, value: object) -> bool:
    """Whether ``value`` is a well-formed id of ``kind``: the whole string, so a trailing newline is refused."""
    return isinstance(value, str) and re.fullmatch(ID_PATTERNS[kind], value) is not None


def id_schema_pattern(kind: IdKind) -> str:
    """``ID_PATTERNS[kind]`` anchored for a JSON Schema ``pattern`` (ECMA-262: ``^…$``). Python's ``re.search``
    lets ``$`` match before a final newline, so code that must be exact uses ``is_id`` as well."""
    return "^" + ID_PATTERNS[kind] + "$"


# ---------------------------------------------------------------- schema ids of records and keys
VOICE_SCHEMA: Final = "narration.voice/v2"
RENDER_SCHEMA: Final = "narration.render/v1"
TAKE_SCHEMA: Final = "narration.take/v1"
ANALYSIS_SCHEMA: Final = "narration.analysis/v1"
MEASUREMENT_SCHEMA: Final = "narration.measurement/v1"
# Not named by the design; chosen here so every record and key carries a versioned schema id.
CANDIDATE_SCHEMA: Final = "narration.candidate/v1"
PROFILE_SCHEMA: Final = "narration.profile/v1"
ENGINE_PROFILE_SCHEMA: Final = "narration.engine-profile/v1"
ALIGNMENT_BENCHMARK_SCHEMA: Final = "narration.alignment-benchmark/v1"
JOB_SCHEMA: Final = "narration.job/v1"
REPORT_SCHEMA: Final = "narration.report/v1"
MEASUREMENT_KEY_SCHEMA: Final = "narration.measurement-key/v1"
DELIVERY_KEY_SCHEMA: Final = "narration.delivery-key/v1"
ANALYSIS_KEY_SCHEMA: Final = "narration.analysis-key/v1"
SEED_SCHEME: Final = "narration-seed/v1"

# ---------------------------------------------------------------- versions of the service's rules
TEXT_CHECKS_VERSION: Final = "text-1.1.0"
QA_PROFILE: Final = "default.v3"
NUMBER_READER: Final = "whisper-english-normalizer+nought@2"
"""The number reader of section 11.3. ``@2`` (plan.md DC-7): number words never merge across punctuation
(each side is read in phrases split at punctuation, so "two thousand, forty" is 2000 and 40, not 2040), and
’ ‘ ʼ are read as the straight apostrophe on both sides. ``@1`` was the vendored reader as-is."""
ALIGNMENT_METHOD: Final = "ctc-forced-align+silence-snap"
POST_RULES: Final = "narration.post/1"
"""The version of the delivery post-processing's own rules (section 13): framing, percentile, fade shape,
quantiser, gain rounding and the ceiling procedure. It is in the delivery key (``DeliveryTools.post``), so a
change to any of them that changes bytes needs a new version (WP13's review)."""
PROFILE_VERSION: Final = "profile-1"
"""The voice profile's measurement set (section 3.6 as changed by DC-1: pyin f0, Boersma HNR, CPPS)."""

# ---------------------------------------------------------------- engine profiles (section 6)
ENGINE_PROFILE_BASE: Final = "qwen3-base-1.7b.p1"
ENGINE_PROFILE_DESIGN: Final = "qwen3-design-1.7b.p1"
MODEL_QWEN_BASE: Final = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_QWEN_DESIGN: Final = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
MODEL_ASR: Final = "openai/whisper-large-v3"
MODEL_SV: Final = "microsoft/wavlm-base-plus-sv"
MODEL_ALIGNER: Final = "facebook/wav2vec2-large-960h-lv60-self"
MODEL_ALIGNER_ALTERNATIVE: Final = "Qwen/Qwen3-ForcedAligner-0.6B"
LANGUAGE: Final = "English"
"""The language passed to Qwen (section 10.2: part of the voice hash)."""

# ---------------------------------------------------------------- the service's own material
CORPUS: Final = "narration-en.v1"
BENCHMARK: Final = "alignment-en.v1"

# ---------------------------------------------------------------- tools, in tools/list order (section 7.1)
TOOL_NAMES: Final = (
    "get_server_status",
    "release_gpu",
    "get_job",
    "get_results",
    "cancel_job",
    "design_voice",
    "profile_voice",
    "measure_voice",
    "check_text",
    "audition_pronunciation",
    "submit_job",
)


@dataclass(frozen=True, slots=True)
class ResourceTemplate:
    """A ``narration://`` resource (section 7.7). ``ttl_ms`` is the running value for job resources."""

    uri_template: str
    name: str
    mime_type: str
    ttl_ms: int
    ttl_ms_terminal: int | None = None
    subscribable: bool = False


RESOURCES: Final = (
    ResourceTemplate("narration://status", "status", "application/json", 5_000),
    ResourceTemplate("narration://designs/{design_id}", "design", "application/json", 60_000),
    ResourceTemplate("narration://measurements/{voice_hash}", "measurement", "application/json", 60_000),
    ResourceTemplate("narration://jobs/{job_id}", "job", "application/json", 2_000, 86_400_000, subscribable=True),
    ResourceTemplate("narration://jobs/{job_id}/report", "job report", "text/markdown", 2_000, 86_400_000),
    ResourceTemplate("narration://takes/{take_id}", "take", "application/json", 86_400_000),
)
RESOURCE_CACHE_SCOPE: Final = "private"


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """A prompt (section 7.8) and its arguments, all required."""

    name: str
    arguments: tuple[str, ...]


PROMPTS: Final = (
    PromptSpec("narrate_script", ("voice_path",)),
    PromptSpec("resolve_flags", ("job_id",)),
    PromptSpec("design_narrator_voice", ("brief",)),
    PromptSpec("add_pronunciation", ("term",)),
)

# ---------------------------------------------------------------- enumerations
Severity = Literal["info", "warn", "fail", "error"]
Verdict = Literal["pass", "warn", "fail"]
JobKind = Literal["generate", "analyse", "design", "measure", "profile", "pronunciation"]
JobStatus = Literal["queued", "running", "cancelling", "completed", "failed", "cancelled"]
JobPhase = Literal[
    "waiting_for_gpu", "loading_model", "canary", "rendering", "postprocessing", "scoring", "retaking", "suggesting"
]
JobOutcome = Literal["all_passed", "needs_attention"]
SubmitStatus = Literal["queued", "running", "completed", "planned"]
SegmentState = Literal[
    "planned",
    "cached",
    "rendering",
    "rendered",
    "postprocessed",
    "scoring",
    "passed",
    "warned",
    "failed_qa",
    "error",
    "skipped",
]
Priority = Literal["batch", "interactive"]
DaemonState = Literal["stopped", "idle", "busy", "stopping"]
GpuHolder = Literal["qwen", "qa"]
DeterminismTier = Literal["bit_exact", "similar"]
ExactMatch = Literal["same", "different", "missing"]
BoundaryKind = Literal["pause", "no_pause", "segment_edge"]
TextWarningKind = Literal["digit", "symbol", "unit_like", "letter"]
CanaryStatus = Literal["hash_match", "similarity_pass", "not_run"]
"""A batch's canary outcome (section 10.1). A failed canary fails the job (``ENGINE_DRIFT``)."""
SuggestionTier = Literal[1, 2, 3, 4]
"""Section 8: 1 pass; 2 warn, every cue placed; 3 warn, a cue unplaced; 4 fail."""
EngineKind = Literal["base", "design"]
"""Which engine profile is meant: the Base clone path or VoiceDesign (section 6)."""
DaemonCommandKind = Literal["release_gpu", "stop", "stop_now"]
"""Requests the front-end and the operator CLI send the daemon through the store (sections 4.1, 7.6)."""
MaterialStatus = Literal["draft", "frozen"]
"""A material set is ``draft`` until gate H1 freezes it; a frozen set never changes (plan.md WP18)."""

SEVERITIES: Final = get_args(Severity)
JOB_STATUSES: Final = get_args(JobStatus)
TERMINAL_JOB_STATUSES: Final = ("completed", "failed", "cancelled")
JOB_PHASES: Final = get_args(JobPhase)
SEGMENT_STATES: Final = get_args(SegmentState)


# ---------------------------------------------------------------- the per-call generation cap (DC-4)
MAX_NEW_TOKENS_CEILING: Final = 8192
"""The pinned Qwen snapshots' ``generation_config.json`` value, which all the evidence used (plan.md section
1.3): the most any render may generate. The engine profile records the effective value; pass that."""


def max_new_tokens_for(text: str, *, per_char: float, floor: int, ceiling: int) -> int:
    """The cap one ``synthesize`` or ``design`` call passes (design section 10.1, DC-4).

    ``min(ceiling, max(floor, ceil(per_char * len(text))))``, where ``text`` is the text the call speaks
    (the engine text, or the design text) and ``per_char``/``floor`` are the engine's
    ``max_new_tokens_per_char``/``max_new_tokens_floor``, hashed into its engine profile. ``len`` counts
    code points of the NFC text. The cap only truncates (ADR 0003), so a render that ends under its cap is
    the same with any cap; a render that reaches it is ``TOKEN_CAP_HIT``.
    """
    if per_char <= 0 or floor < 1 or ceiling < 1:
        raise ValueError("per_char must be positive, and floor and ceiling at least 1")
    return min(ceiling, max(floor, math.ceil(per_char * len(text))))
