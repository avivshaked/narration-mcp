"""Typed records: the request fragments, the text echo, the sidecars and the results (design sections 6,
7 and Appendix B).

Field names are the design's JSON names, so ``serial.to_json(record)`` is the record's JSON form and
``serial.from_json(Record, data)`` reads it back. Records are frozen; collections are tuples. A ``dict``
field holds free-form data (``details``, generation settings) whose keys the design does not fix.

Where the design shows a field only in an example, its type here is the contract. Additions the design
does not show are marked "(added)" with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .names import (
    ALIGNMENT_BENCHMARK_SCHEMA,
    ANALYSIS_SCHEMA,
    CANDIDATE_SCHEMA,
    ENGINE_PROFILE_SCHEMA,
    JOB_SCHEMA,
    MEASUREMENT_SCHEMA,
    PROFILE_SCHEMA,
    RENDER_SCHEMA,
    SEED_SCHEME,
    TAKE_SCHEMA,
    CanaryStatus,
    DeterminismTier,
    ExactMatch,
    JobKind,
    JobOutcome,
    JobPhase,
    JobStatus,
    Priority,
    SegmentState,
    Severity,
    SuggestionTier,
    Verdict,
)

Details = dict[str, Any]

# ======================================================================== fragments (section 7.2)


@dataclass(frozen=True, slots=True, kw_only=True)
class Flag:
    """A flag inside results (section 7.2 Flag; codes in section 14)."""

    code: str
    severity: Severity
    message: str
    segment_id: str | None = None
    cue: int | None = None
    retake_trigger: bool | None = None
    details: Details | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Error:
    """A tool error (section 7.2 Error, with DC-2's ``retry_after_s``).

    ``retryable``: the same call may succeed later unchanged. ``retry_after_s`` (DC-2) is set on every
    retryable error: the server's minimum wait before resending the identical request.
    """

    code: str
    message: str
    retryable: bool
    hint: str | None = None
    field: str | None = None
    details: Details | None = None
    retry_after_s: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Voice:
    """A clip the caller keeps, by location (section 7.2 Voice). Never stored by the service."""

    path: str
    sha256: str
    transcript: str


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioRef:
    """Any audio file by location, e.g. ``profile_voice``'s ``audio`` (section 7.6)."""

    path: str
    sha256: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Hint:
    """A pronunciation hint for one term (section 7.2 Hint, section 9.1).

    ``note`` (added): section 9.1 lists it; it is informational and enters no key.
    """

    term: str
    respell: str | None = None
    align_as: str | None = None
    asr_aliases: tuple[str, ...] = ()
    note: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ExactSpan:
    """A span of a cue's text as sent that must be heard exactly (R14): code points, end exclusive."""

    start: int
    end: int


@dataclass(frozen=True, slots=True, kw_only=True)
class CueIn:
    """One cue of a segment as the caller sends it (section 7.2 Segment.cues)."""

    text: str
    at_s: float | None = None
    exact: tuple[ExactSpan, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class Fit:
    """Fit reporting margins (section 12); both default to 0."""

    lead_in_s: float = 0.0
    tail_s: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class SegmentIn:
    """A segment as the caller sends it (section 7.2 Segment). A text-only segment is one cue.

    ``controls`` is kept as sent: every property of it is refused by every current engine (section 3.3).
    """

    segment_id: str
    cues: tuple[CueIn, ...] = ()
    text: str | None = None
    attempts: tuple[int, ...] | None = None
    scene_seconds: float | None = None
    fit: Fit | None = None
    controls: Details | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SubmitOptions:
    """``submit_job``'s ``options`` (section 7.3), with the design's defaults."""

    dry_run: bool = False
    takes: int = 1
    max_retakes: int = 2
    strict_text: bool = False
    priority: Priority = "batch"
    idempotency_key: str | None = None


# ======================================================================== text echo (sections 7.5, 9.1)


@dataclass(frozen=True, slots=True, kw_only=True)
class HintApplied:
    """One application of a hint inside a cue: ``offset`` in code points into the cue's spoken text."""

    term: str
    respell: str | None
    offset: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ExactWords:
    """An exact span with its word range: ``words`` = [first word, end word), half-open (App. B)."""

    start: int
    end: int
    words: tuple[int, int]


@dataclass(frozen=True, slots=True, kw_only=True)
class CueText:
    """Per cue: received → spoken → engine, with spans in the joined texts (section 7.5, App. B)."""

    index: int
    received: str
    spoken: str
    engine: str
    spoken_span: tuple[int, int]
    engine_span: tuple[int, int]
    hints_applied: tuple[HintApplied, ...] = ()
    warnings: tuple[Flag, ...] = ()
    exact: tuple[ExactWords, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class TextChecksInfo:
    """The text rules applied (section 9: a version and a rules hash go into every sidecar)."""

    version: str
    rules_sha256: str


@dataclass(frozen=True, slots=True, kw_only=True)
class SegmentText:
    """A segment after the text pipeline (section 9.1): what the engine speaks and what QA checks against.

    ``warnings`` holds the segment-level text flags (``TERM_SPLIT_ACROSS_CUES``); each cue's own warnings
    are in its ``CueText``. ``spoken_chars`` is the spoken join's length in code points (section 7.2).
    """

    segment_id: str
    cues: tuple[CueText, ...]
    spoken_text: str
    engine_text: str
    spoken_chars: int
    warnings: tuple[Flag, ...] = ()
    text_checks: TextChecksInfo | None = None


# ======================================================================== audio records (sections 13, 15)


@dataclass(frozen=True, slots=True, kw_only=True)
class RawAudio:
    """The render layer's audio: WAV float32 mono at the model's rate, untrimmed (section 13)."""

    path: str
    sha256: str
    sample_rate: int
    samples: int
    format: str = "WAV FLOAT mono"


@dataclass(frozen=True, slots=True, kw_only=True)
class DeliveryAudio:
    """The delivery file: 48 kHz PCM_24 mono (section 13). ``duration_s`` = samples / sample_rate."""

    path: str
    sha256: str
    sample_rate: int
    samples: int
    duration_s: float
    format: str = "WAV PCM_24 mono"


@dataclass(frozen=True, slots=True, kw_only=True)
class Trim:
    """The trim record (section 13 step 1). ``rule`` (from App. B) states the rule in words."""

    head_s: float
    tail_s: float
    pad_s: float
    rule: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Loudness:
    """The loudness record (section 13 steps 3 and 6). ``target_lufs`` appears in App. B's take.json."""

    measured_lufs: float
    gain_db: float
    true_peak_dbtp: float
    ceiling_applied: bool
    target_lufs: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class DeliveryTools:
    """The resampler's and loudness meter's names and versions: part of the delivery key (section 10.2)."""

    resampler: str
    loudness_meter: str


# ======================================================================== render.json (App. B)


@dataclass(frozen=True, slots=True, kw_only=True)
class RenderVoice:
    voice_hash: str
    clip_sha256: str
    x_vector_only_mode: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class RenderEngine:
    """The engine as it rendered: pinned values, plus ``observed`` (GPU, driver, CUDA, cuDNN; not hashed)."""

    engine_profile_id: str
    engine_profile_hash: str
    model_repo: str
    model_revision: str
    non_streaming_mode: bool
    generation: Details
    observed: Details = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryRecord:
    batch_status: CanaryStatus


@dataclass(frozen=True, slots=True, kw_only=True)
class Licence:
    """Licences recorded with every render and analysis (section 18). Keys depend on the record."""

    generation_model: str | None = None
    voice_clip: str | None = None
    aligner: str | None = None
    asr: str | None = None
    sv: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class RenderRecord:
    """``renders/<ab>/rn_<16hex>/render.json`` (App. B). Immutable once written."""

    schema: str = RENDER_SCHEMA
    render_id: str
    render_key: str
    voice: RenderVoice
    engine: RenderEngine
    engine_text: str
    seed: int
    seed_scheme: str = SEED_SCHEME
    attempt: int
    raw: RawAudio
    hit_token_cap: bool
    gen_s: float
    rtf: float
    canary: CanaryRecord
    licence: Licence


# ======================================================================== take.json (App. B)


@dataclass(frozen=True, slots=True, kw_only=True)
class TakeRecord:
    """``takes/<ab>/tk_<16hex>/take.json`` (App. B): the delivery layer; ``take_id`` names it.

    ``flags`` (added): the post-processing flags of this take (``LOUDNESS_UNDER_TARGET``, ``GAIN_HIGH``),
    which depend only on the raw audio and the delivery profile, so they belong to the take.
    """

    schema: str = TAKE_SCHEMA
    take_id: str
    delivery_key: str
    render_id: str
    delivery: DeliveryAudio
    trim: Trim
    loudness: Loudness
    tools: DeliveryTools
    post_stretched: bool = False
    flags: tuple[Flag, ...] = ()


# ======================================================================== analysis (App. B, section 11)


@dataclass(frozen=True, slots=True, kw_only=True)
class WordTiming:
    """A word in delivery-file seconds; null times for a word that could not be placed."""

    text: str
    start_s: float | None
    end_s: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class CueTiming:
    """A cue's times in delivery-file seconds (section 11.2). Null times: unplaced, never interpolated."""

    index: int
    start_s: float | None
    end_s: float | None
    confidence: float | None
    words: tuple[WordTiming, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class ErrorStats:
    """Measured cue-boundary error: p50 and p95 in seconds, over n boundaries."""

    p50_s: float
    p95_s: float
    n: int


@dataclass(frozen=True, slots=True, kw_only=True)
class MeasuredError:
    """The published error of the alignment method (R1, section 11.2). Null until a benchmark exists."""

    p50_s: float | None
    p95_s: float | None
    n: int | None
    benchmark: str | None
    by_kind: dict[str, ErrorStats] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class CrossCheck:
    model: str
    max_disagreement_s: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class Alignment:
    """A take's cue alignment (section 11.2 step 8; App. B ``alignment``)."""

    method: str
    model: str
    revision: str
    device: str
    cross_check: CrossCheck
    measured_error: MeasuredError | None
    cues: tuple[CueTiming, ...]
    flags: tuple[Flag, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class ExactResult:
    """One exact span's outcome (section 11.3): ``expected`` and ``heard`` in normalised form."""

    cue: int
    start: int
    end: int
    expected: str
    heard: str | None
    match: ExactMatch


@dataclass(frozen=True, slots=True, kw_only=True)
class TermResult:
    """One occurrence of a hinted term (section 11.1 step 6)."""

    term: str
    cue: int
    heard: str | None
    ok: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class QaMetrics:
    """QA's numbers (App. B ``qa.metrics``). ``spoken_cps`` and ``clipping_fraction`` (added): section
    11.1 steps 1 and 9 name them."""

    wer_raw: float | None
    wer_adj: float | None
    word_errors: int | None
    exact_ok: bool
    spk_sim_anchor: float | None
    spoken_wpm: float | None
    expected_spoken_wpm: float | None
    head_insertion_words: int
    end_insertion_words: int
    longest_silence_s: float | None
    spoken_cps: float | None = None
    clipping_fraction: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class QaThresholds:
    """The thresholds this verdict used, from the voice's measurement and the QA profile."""

    spk_warn: float | None
    spk_fail: float
    pace_tol: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class QaResult:
    """A take's QA (App. B ``qa``). The verdict depends only on this take and the request's inputs."""

    verdict: Verdict
    transcript: str | None
    exact: tuple[ExactResult, ...]
    terms: tuple[TermResult, ...]
    metrics: QaMetrics
    thresholds: QaThresholds
    flags: tuple[Flag, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalysisText:
    cues: tuple[CueText, ...]
    text_checks: TextChecksInfo
    hints_used: tuple[Hint, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalysisVersions:
    qa_profile: str
    asr: str
    sv: str
    aligner_method: str
    number_reader: str
    measurement: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalysisRecord:
    """``takes/<ab>/tk_<16hex>/analyses/an_<16hex>.json`` (App. B). One per analysis key."""

    schema: str = ANALYSIS_SCHEMA
    analysis_id: str
    analysis_key: str
    take_id: str
    text: AnalysisText
    versions: AnalysisVersions
    alignment: Alignment
    qa: QaResult
    licence: Licence


# ======================================================================== measurement (section 3.2, App. B)


@dataclass(frozen=True, slots=True, kw_only=True)
class EngineRef:
    """An engine profile by id and hash, as results report it."""

    id: str
    hash: str


@dataclass(frozen=True, slots=True, kw_only=True)
class TranscriptCheck:
    heard: str
    wer: float
    ok: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class SimilarityBaseline:
    """The measured similarity distribution (section 3.2); QA thresholds come from its p5 (section 11.1)."""

    anchor_p5: float
    anchor_p50: float
    consistency_p5: float


@dataclass(frozen=True, slots=True, kw_only=True)
class PaceTrend:
    """Pace against length, fitted over the rungs up to ``band_max_chars`` (section 3.2)."""

    intercept_wpm: float
    per_100_chars: float
    band_max_chars: int


@dataclass(frozen=True, slots=True, kw_only=True)
class PacePoint:
    chars: int
    wpm: float


@dataclass(frozen=True, slots=True, kw_only=True)
class Pace:
    trend: PaceTrend
    tol: float
    curve: tuple[PacePoint, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class LadderSeed:
    """One take of a ladder rung (section 7.6: per rung and seed: spoken wpm, wer_adj, similarity, verdict)."""

    seed: int
    attempt: int
    take_id: str | None
    wpm: float | None
    wer_adj: float | None
    sim: float | None
    verdict: Verdict
    duration_s: float | None = None
    flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class LadderRung:
    chars: int
    seeds: tuple[LadderSeed, ...]
    passes: bool
    paragraph_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Anchor:
    """The voice's anchor: the centroid embedding over the clip and the calibration takes (section 3.2)."""

    model: str
    dim: int
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class CalibrationTake:
    paragraph_id: str
    seed: int
    attempt: int
    take_id: str
    sim_anchor: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class MeasurementRecord:
    """``measurements/<voice_hash>/<engine_profile_id>/measurement.json`` (App. B). Returned in full."""

    schema: str = MEASUREMENT_SCHEMA
    voice_hash: str
    clip_sha256: str
    engine_profile: EngineRef
    measurement_key: str
    transcript_check: TranscriptCheck
    corpus: str
    similarity: SimilarityBaseline
    pace: Pace
    max_segment_chars: int | None
    max_segment_seconds: float | None
    ladder: tuple[LadderRung, ...]
    anchor: Anchor
    calibration: tuple[CalibrationTake, ...]
    measured_at: str


# ======================================================================== design and profile (3.1, 3.5, 3.6)


@dataclass(frozen=True, slots=True, kw_only=True)
class LintFinding:
    """A negation the positive-only word list found (section 3.5)."""

    phrase: str
    offset: int
    suggestion: str


@dataclass(frozen=True, slots=True, kw_only=True)
class LintResult:
    policy: str
    findings: tuple[LintFinding, ...]
    note: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ProfileMeasurements:
    """The voice profile's numbers (section 3.6 as changed by DC-1: no jitter or shimmer; CPPS instead).

    ``speaking_rate_wpm`` is null when no transcript is known for the audio.
    """

    duration_s: float
    pitch_median_hz: float | None
    pitch_p10_hz: float | None
    pitch_p90_hz: float | None
    pitch_range_st: float | None
    speaking_rate_wpm: float | None
    pause_ratio: float
    loudness_lufs: float | None
    spectral_centroid_hz: float
    hnr_db: float | None
    cpps_db: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ProfilePictures:
    spectrogram: str
    pitch: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ProfileRecord:
    """``profiles/<ab>/<sha256>/profile.json`` (section 15): measurements and picture paths."""

    schema: str = PROFILE_SCHEMA
    audio_sha256: str
    profile_version: str
    measurements: ProfileMeasurements
    pictures: ProfilePictures


@dataclass(frozen=True, slots=True, kw_only=True)
class Candidate:
    """A designed voice candidate (sections 3.1, 6): ``designs/<design_id>/<cand>/candidate.json``."""

    schema: str = CANDIDATE_SCHEMA
    design_id: str
    index: int
    clip: AudioRef
    transcript: str
    transcript_check: TranscriptCheck | None
    description: str
    description_sha256: str
    design_text: str
    seed: int
    engine_profile: EngineRef
    lint: LintResult
    profile: ProfileRecord | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ProvenanceEntry:
    """One line of ``provenance.jsonl`` (section 17.4): append-only, never pruned."""

    clip_sha256: str
    design_id: str
    date: str


# ======================================================================== engine profile (sections 6, 10.1)


@dataclass(frozen=True, slots=True, kw_only=True)
class Determinism:
    """The determinism switches pinned in an engine profile (section 10.1)."""

    attn_implementation: str
    tf32: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    deterministic_algorithms: str
    cublas_workspace_config: str


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryPin:
    """The service's canary under one engine profile on this machine (section 10.1, DC-3).

    Created by ``narration-admin engine pin`` on the installing machine; not part of the profile's hash.
    """

    material: str
    seed: int
    raw_sha256: str
    embedding: tuple[float, ...]
    threshold: float
    pinned_at: str


@dataclass(frozen=True, slots=True, kw_only=True)
class EngineProfile:
    """``engines/<engine_profile_id>.json`` (sections 6, 15).

    ``hash`` covers every field except ``hash``, ``snapshot_dir`` (a local path), ``observed`` (GPU,
    driver, CUDA, cuDNN: recorded, not hashed) and ``canary`` (made on the installing machine, DC-3).
    """

    schema: str = ENGINE_PROFILE_SCHEMA
    engine_profile_id: str
    hash: str
    model_repo: str
    model_revision: str
    snapshot_dir: str
    weights: dict[str, str]
    worker_project: str
    uv_lock_sha256: str
    packages: dict[str, str]
    dtype: str
    determinism: Determinism
    settings: Details
    capabilities: Details
    licence: str
    vram_need_mb: int
    tier: DeterminismTier | None = None
    observed: Details = field(default_factory=dict)
    canary: CanaryPin | None = None


# ======================================================================== alignment benchmark (section 11.2)


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkRef:
    id: str
    sha256: str
    description: str


@dataclass(frozen=True, slots=True, kw_only=True)
class AlignmentBenchmark:
    """``alignment/<method_id>.json`` (sections 6, 11.2): the method's measured error, published."""

    schema: str = ALIGNMENT_BENCHMARK_SCHEMA
    method_id: str
    model: str
    revision: str
    snap: Details
    benchmark: BenchmarkRef
    measured_error: ErrorStats
    by_kind: dict[str, ErrorStats]
    measured_at: str


# ======================================================================== jobs and results (sections 7.4, 7.5, 8)


@dataclass(frozen=True, slots=True, kw_only=True)
class Progress:
    """Job progress (section 7.4); ``total_s`` may grow with retakes."""

    done_s: float
    total_s: float
    fraction: float
    segments_done: int
    segments_total: int


@dataclass(frozen=True, slots=True, kw_only=True)
class JobRecord:
    """``jobs/<job_id>/job.json`` and the job row (section 6). The request is kept by value."""

    schema: str = JOB_SCHEMA
    job_id: str
    kind: JobKind
    request: Details
    request_sha256: str
    label: str | None
    priority: Priority
    status: JobStatus
    phase: JobPhase | None
    round: int
    progress: Progress
    outcome: JobOutcome | None
    error: Error | None
    idempotency_key: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Suggestion:
    """The suggested take's tier and why (section 8, V2). Advice; the caller chooses."""

    tier: SuggestionTier
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class PaceValue:
    """A pace in spoken words per minute (section 7.5 ``qa.pace`` / ``qa.pace_expected``)."""

    spoken_wpm: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class FitReport:
    """Fit for one take, only when the segment gave ``scene_seconds`` (section 12)."""

    scene_seconds: float
    budget_s: float
    duration_s: float
    slack_s: float
    overrun_s: float
    flags: tuple[Flag, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class TakeQa:
    """A take's QA as ``get_results`` shows it (section 7.5 ``qa``)."""

    verdict: Verdict
    flags: tuple[Flag, ...]
    wer_raw: float | None
    wer_adj: float | None
    exact_ok: bool
    exact: tuple[ExactResult, ...]
    terms: tuple[TermResult, ...]
    spk_sim_anchor: float | None
    pace: PaceValue
    pace_expected: PaceValue
    transcript: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class TakeResult:
    """One take in ``get_results`` (section 7.5)."""

    take_id: str
    render_id: str
    attempt: int
    seed: int
    fresh: bool
    delivery: DeliveryAudio
    trim: Trim
    loudness: Loudness
    analysis_id: str | None
    cues: tuple[CueTiming, ...]
    alignment: Alignment | None
    qa: TakeQa | None
    fit: FitReport | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SegmentResultText:
    spoken_chars: int
    cues: tuple[CueText, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class SegmentResult:
    """One segment in ``get_results`` (section 7.5)."""

    segment_id: str
    status: SegmentState
    suggested_take_id: str | None
    suggestion: Suggestion | None
    text: SegmentResultText
    takes: tuple[TakeResult, ...]
    flags: tuple[Flag, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class Consistency:
    """Similarity across the suggested takes of one request (section 11.1): a report, never a verdict."""

    min: float | None
    median: float | None
    outliers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class ListenFirstItem:
    """One entry of ``listen_first`` (section 11.1), in the design's priority order."""

    segment_id: str
    take_id: str | None
    cue: int | None
    reason: str
    from_s: float | None
    to_s: float | None
