"""Typed records: the request fragments, the text echo, the sidecars and the results (design sections 6,
7 and Appendix B).

Field names are the design's JSON names, so ``serial.to_json(record)`` is the record's JSON form and
``serial.from_json(Record, data)`` reads it back. Records are frozen; collections are tuples. A ``dict``
field holds free-form data (``details``, generation settings) whose keys the design does not fix.

**Absent versus null.** A field typed ``X | None = None`` is *optional*: when unset it is left out of the
JSON (``serial.to_json``). A field typed ``X | None`` with no default is *nullable*: it is always present,
and null says "not known" or "not there" (an unplaced cue's times, a take with no analysis yet). The
published output schemas follow the same rule (``tests/contracts`` validates records against them).

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
    DaemonCommandKind,
    DaemonState,
    DeterminismTier,
    ExactMatch,
    GpuHolder,
    JobKind,
    JobOutcome,
    JobPhase,
    JobStatus,
    MaterialStatus,
    Priority,
    SegmentState,
    Severity,
    SuggestionTier,
    Verdict,
    WorkerRole,
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
    """An exact span with its word range (App. B).

    ``start`` and ``end`` are code-point offsets into the cue as sent, echoed as the caller gave them.
    ``words`` = [first word, end word), half-open, counting the cue's spoken words as
    ``narration.text.words`` defines them (a token of punctuation only is not a word). QA and the aligner
    use the same indices.
    """

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
    """The loudness record (section 13 steps 3 and 6). ``target_lufs`` appears in App. B's take.json.

    ``measured_lufs`` and ``true_peak_dbtp`` are null for a take with no defined loudness: every block under
    BS.1770-4's absolute gate (-70 LUFS), or an all-zero file. JSON cannot carry minus infinity, and a floor
    value would be a measurement nobody made.
    """

    measured_lufs: float | None
    gain_db: float
    true_peak_dbtp: float | None
    ceiling_applied: bool
    target_lufs: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class DeliveryTools:
    """What the delivery key names besides the delivery profile (section 10.2): the resampler's and loudness
    meter's names and versions, with their pinned parameters, and ``post``, the version of the post-processing's
    own rules (``names.POST_RULES``)."""

    resampler: str
    loudness_meter: str
    post: str


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
    """A take's cue alignment (section 11.2 step 8; App. B ``alignment``).

    ``model`` and ``revision`` name the aligner. The service sets them on every analysis it writes, and
    ``get_results`` shows them (``TakeAlignment``). They are optional only because App. B's example keeps the
    aligner in ``versions.aligner_method`` instead, and a sidecar in that shape must still load.
    """

    method: str
    model: str | None = None
    revision: str | None = None
    device: str
    cross_check: CrossCheck
    measured_error: MeasuredError | None
    cues: tuple[CueTiming, ...]
    flags: tuple[Flag, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class ExactResult:
    """One exact span's outcome (section 11.3): ``expected`` and ``heard`` in normalised form.

    ``words`` is the span's word range, [first, end) as in ``ExactWords``: the part the analysis key covers.
    ``start`` and ``end`` are the offsets of a request, which the key does not cover, so an analysis reused
    by another request may hold another request's offsets. The job assembler (WP36) restamps them from the
    current request's ``ExactWords``, matched by ``cue`` and ``words``.
    """

    cue: int
    start: int
    end: int
    words: tuple[int, int]
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
    """``takes/<ab>/tk_<16hex>/analyses/an_<16hex>.json`` (App. B). One per analysis key.

    ``embedding`` (added): the take's speaker embedding (``versions.sv``), a raw output kept so the per-job
    consistency report and ``SPK_OUTLIER`` work for cached takes without loading a model (section 11.1).
    """

    schema: str = ANALYSIS_SCHEMA
    analysis_id: str
    analysis_key: str
    take_id: str
    text: AnalysisText
    versions: AnalysisVersions
    alignment: Alignment
    qa: QaResult
    licence: Licence
    embedding: tuple[float, ...] | None = None


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
    ``clip`` is the designed canary clip (kept under ``engines/``, immutable, never collected) and
    ``transcript`` its text; ``seed`` is the gate render's seed; ``raw_sha256`` and ``embedding`` are the
    pinned gate render's.
    """

    material: str
    clip: AudioRef
    transcript: str
    seed: int
    raw_sha256: str
    embedding: tuple[float, ...]
    threshold: float
    pinned_at: str


@dataclass(frozen=True, slots=True, kw_only=True)
class EngineProfile:
    """``engines/<engine_profile_id>.json`` (sections 6, 15).

    ``hash`` covers every field except ``hash``, ``snapshot_dir`` (a local path), ``observed`` (GPU,
    driver, CUDA, cuDNN: recorded, not hashed), ``tier`` (the outcome of the repeat test on this machine,
    section 10.1: an observation, set after the pin) and ``canary`` (made on the installing machine, DC-3).
    Setting ``tier`` or ``canary`` after the pin therefore changes no render key.
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
class JobAttempt:
    """One attempt of one segment slot in a job (section 8): what was produced and whether this job made it.

    ``fresh`` is true when this job rendered it, false when it came from the cache (section 7.5, R11); it
    cannot be recomputed later, so the job keeps it. ``round`` is the round that produced it (0 first).
    """

    attempt: int
    seed: int
    round: int
    render_key: str
    render_id: str | None
    take_id: str | None
    analysis_id: str | None
    fresh: bool
    verdict: Verdict | None
    flags: tuple[Flag, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class JobSegment:
    """A job's work on one segment (section 6 ``items[]``; section 7.4 ``segments[]``).

    ``flags`` are the segment's execution and text flags (``RENDER_FAILED``, ``GPU_OOM``, ``CANCELLED``,
    ``SEGMENT_TOO_LONG``, text warnings). A take's own flags are in its analysis and its ``JobAttempt``.
    """

    segment_id: str
    state: SegmentState
    attempts: tuple[JobAttempt, ...] = ()
    takes_ok: int = 0
    retakes_used: int = 0
    flags: tuple[Flag, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class JobRecord:
    """``jobs/<job_id>/job.json`` and the job row (section 6). The request is kept by value.

    ``request`` is the tool's arguments as a JSON object, stored after they passed the tool's input
    schema (``schemas.TOOLS_BY_NAME[tool].input_schema``: ``submit_job`` for ``generate`` and ``analyse``,
    ``design_voice``, ``measure_voice``, ``profile_voice``, ``audition_pronunciation``). There is no typed
    record for it: the input schema fixes its layout, and it is read as a dict. ``items`` is the per-segment
    state the job engine writes and ``get_job``/``get_results`` read. ``result`` is the handle of a
    non-generate job's result (e.g. ``design_id``; the measurement key; the audition's take ids), and
    ``message`` the human-readable progress line of section 7.4.
    """

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
    items: tuple[JobSegment, ...] = ()
    result: Details | None = None
    message: str | None = None


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
class TakeAlignment:
    """A take's alignment as ``get_results`` shows it (section 7.5 ``alignment``).

    Built from the analysis's ``Alignment`` (App. B): ``cross_check`` names the cross-check in words (for
    example "whisper-large-v3 word timestamps"), ``max_disagreement_s`` is lifted beside it, and the cue
    times appear once, at the take's level (``TakeResult.cues``). ``measured_error`` is null until an
    alignment benchmark exists.
    """

    method: str
    model: str
    revision: str
    cross_check: str
    max_disagreement_s: float | None
    measured_error: MeasuredError | None
    flags: tuple[Flag, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class TakeResult:
    """One take in ``get_results`` (section 7.5).

    ``alignment`` and ``qa`` are null for a take with no analysis (a failed or cancelled job's unfinished
    work). ``fit`` is left out unless the segment gave ``scene_seconds`` (section 12).

    ``flags`` (added): this take's flags outside its cached verdict, never part of ``qa.verdict``: the
    delivery flags of its ``TakeRecord`` (``LOUDNESS_UNDER_TARGET``, ``GAIN_HIGH``) and the per-job flags
    (``SPK_OUTLIER`` from the consistency report, ``CANARY_MISMATCH`` from the canary gate, ``RETAKEN``).
    """

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
    alignment: TakeAlignment | None
    qa: TakeQa | None
    fit: FitReport | None = None
    flags: tuple[Flag, ...] = ()


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


# ======================================================================== audition (section 7.6)


@dataclass(frozen=True, slots=True, kw_only=True)
class AuditionVariantResult:
    """One respelling variant of ``audition_pronunciation``: its takes and what the ASR heard in each.

    Whoever owns the text decides by ear; the service records no choice (section 7.6).
    """

    label: str
    respell: str
    engine_text: str
    takes: tuple[TakeResult, ...]
    heard: tuple[str | None, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class AuditionResult:
    """An ``audition_pronunciation`` job's result (``get_results`` ``audition``)."""

    term: str
    carrier: str | None
    variants: tuple[AuditionVariantResult, ...]


# ======================================================================== the daemon (sections 4, 4.1, 7.6)


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkerInfo:
    role: WorkerRole
    pid: int


@dataclass(frozen=True, slots=True, kw_only=True)
class CurrentJob:
    job_id: str
    kind: JobKind
    label: str | None
    phase: JobPhase | None
    started_at: str


@dataclass(frozen=True, slots=True, kw_only=True)
class GpuStatus:
    """The GPU as the daemon sees it (sections 4, 7.6; DC-2 ``admission.gpu``). ``need_mb`` is by model
    group (``qwen``, ``qa``); ``waiting_since`` is set while a job waits for free VRAM."""

    name: str | None
    total_mb: int | None
    free_mb: int | None
    in_use: bool
    holder: GpuHolder | None
    unload_in_s: float | None
    need_mb: dict[str, int] = field(default_factory=dict)
    waiting_since: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class DaemonStatus:
    """``run/daemon.json`` (section 15): written by the daemon on start and on every phase change, read by
    the front-end for ``get_server_status`` and DC-2's ``admission`` and ``poll_after_s``.

    ``est_drain_s`` is the daemon's estimate of how long the queue will take (null when unknown).
    """

    state: DaemonState
    pid: int | None
    started_at: str | None
    workers: tuple[WorkerInfo, ...]
    current_job: CurrentJob | None
    gpu: GpuStatus
    est_drain_s: float | None
    updated_at: str


@dataclass(frozen=True, slots=True, kw_only=True)
class DaemonCommand:
    """A request to the daemon through the store (there are no sockets, section 4).

    ``release_gpu``: unload an idle model now (section 7.6); ``stop``: finish the in-flight segment, then
    exit; ``stop_now``: re-queue it and exit (section 4.1). The daemon completes each with a ``result``
    (for ``release_gpu``: ``{released, holder_before, busy_job}``).
    """

    command_id: str
    kind: DaemonCommandKind
    requested_at: str
    done_at: str | None
    result: Details | None


# ======================================================================== the service's own material (WP18)


@dataclass(frozen=True, slots=True, kw_only=True)
class MaterialBoundary:
    """A cue boundary in the alignment benchmark: whether a pause is expected after cue ``after_cue``."""

    after_cue: int
    pause: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class MaterialParagraph:
    """One paragraph of the service's own material, as a Segment a request can carry (section 7.2)."""

    segment_id: str
    cues: tuple[CueIn, ...]
    spoken_chars: int
    target_spoken_chars: int | None = None
    boundaries: tuple[MaterialBoundary, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class MaterialSet:
    """A material set (``material/<kind>/<set-id>/``) checked against its manifest.

    ``sha256`` is the sha256 of the set's ``manifest.json`` bytes, which list every file's sha256, so it
    names the set's exact content (the benchmark's ``sha256`` in section 11.2; the corpus version in the
    measurement key). For the calibration corpus, ``paragraphs`` are the calibration paragraphs and
    ``ladder`` the length-ladder paragraphs, one per rung.
    """

    set_id: str
    kind: str
    version: int
    status: MaterialStatus
    sha256: str
    paragraphs: tuple[MaterialParagraph, ...]
    ladder: tuple[MaterialParagraph, ...] = ()
    hints: tuple[Hint, ...] = ()
    invented_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryMaterial:
    """The canary's text (DC-3): the description, design text and seed its clip is designed from, and the
    gate's fixed text and seed."""

    set_id: str
    status: MaterialStatus
    sha256: str
    description: str
    design_text: str
    design_seed: int
    gate_text: str
    gate_seed: int
