"""Names the design fixes: ids, schema ids, versions, tools, resources, prompts and enumerations.

Every other module takes these from here and never retypes them (AGENTS.md section 6). Section numbers
refer to ``docs/design.md``.
"""

from __future__ import annotations

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
NUMBER_READER: Final = "whisper-english-normalizer+nought@1"
ALIGNMENT_METHOD: Final = "ctc-forced-align+silence-snap"
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

SEVERITIES: Final = get_args(Severity)
JOB_STATUSES: Final = get_args(JobStatus)
TERMINAL_JOB_STATUSES: Final = ("completed", "failed", "cancelled")
JOB_PHASES: Final = get_args(JobPhase)
SEGMENT_STATES: Final = get_args(SegmentState)
