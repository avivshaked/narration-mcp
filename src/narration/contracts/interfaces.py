"""The seams between work packages, as ``Protocol``s (plan.md WP01).

Each protocol names the design sections it implements and the work package that builds it. A work package
may add methods to its own implementation; changing what is written here needs a contract-change request
(AGENTS.md section 3). Everything is typed with the records of ``narration.contracts.models``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from narration.config import DeliveryConfig, MeasurementConfig

from .models import (
    Alignment,
    AlignmentBenchmark,
    AnalysisRecord,
    Anchor,
    AudioRef,
    CanaryMaterial,
    Candidate,
    Consistency,
    CueTiming,
    DaemonCommand,
    DaemonStatus,
    DeliveryTools,
    EngineProfile,
    Flag,
    Hint,
    JobRecord,
    LintResult,
    ListenFirstItem,
    Loudness,
    MaterialSet,
    MeasuredError,
    MeasurementRecord,
    Pace,
    ProfileRecord,
    ProvenanceEntry,
    QaResult,
    RenderRecord,
    SegmentIn,
    SegmentResult,
    SegmentText,
    SimilarityBaseline,
    Suggestion,
    TakeRecord,
    TextChecksInfo,
    Trim,
    Verdict,
)
from .names import DaemonCommandKind, EngineKind, JobStatus, WorkerRole
from .worker import AlignReply, AsrWord, HelloReply

# ======================================================================== WP10: text (sections 3.5, 7.2, 9)


@runtime_checkable
class TextPlanner(Protocol):
    """The text pipeline of section 9.1 (WP10): sanitise, canonical form, join, hints, checks, exact spans.

    The text is spoken as sent: only whitespace and Unicode form change; hints apply per cue and never
    across a cue boundary; warnings never change the text.
    """

    @property
    def checks_info(self) -> TextChecksInfo:
        """``text_checks_version`` (``names.TEXT_CHECKS_VERSION``) and the sha256 of the rules in force."""
        ...

    def plan_segment(self, segment: SegmentIn, hints: Sequence[Hint]) -> SegmentText:
        """One segment through section 9.1 steps 1–5.

        Raises ``NarrationError``: ``TEXT_REFUSED`` for markup or control characters (every offender in
        ``details``); ``INVALID_ARGUMENT`` (with ``field``) when ``text`` differs from the join of the
        cues, or an exact span is out of range, overlapping, or cuts a word.
        """
        ...

    def plan_request(
        self, segments: Sequence[SegmentIn], hints: Sequence[Hint], *, strict_text: bool
    ) -> tuple[SegmentText, ...]:
        """Every segment of a request, in order. Duplicate segment ids are ``INVALID_ARGUMENT``; with
        ``strict_text``, any text warning left makes the request ``TEXT_REFUSED`` (every offender listed)."""
        ...


@runtime_checkable
class DescriptionLinter(Protocol):
    """The positive-only check of voice descriptions (section 3.5, WP10). It warns, never refuses."""

    def lint(self, description: str) -> LintResult:
        """Whole-word, case-insensitive word-list findings, each with its phrase, offset and suggestion."""
        ...


# ======================================================================== WP12: keys and store (10.2, 10.3, 15, 17.2)


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalysisKeyInputs:
    """What the analysis key covers (section 10.2), and nothing else.

    ``cue_spans`` are each cue's [start, end) in the spoken text; ``exact_spans`` are (cue, first word,
    end word); ``hints_qa`` are the QA inputs of the hints used: (term, asr_aliases, align_as).

    ``measurement_key`` is what makes the speaker and pace thresholds part of the key: it covers the ladder
    settings (``sim_warn_margin``, ``sim_fail_floor``, ``pace_tol_min`` among them). A take scored against a
    measurement passes that measurement's key. A ladder take, scored while its measurement is being built
    (with the calibration ``anchor`` and ``similarity``), passes the key of the measurement being built,
    which is known before the ladder runs. Only work with no speaker or pace check at all
    (``audition_pronunciation``) passes None.
    """

    delivery_sha256: str
    spoken_text: str
    cue_spans: tuple[tuple[int, int], ...]
    exact_spans: tuple[tuple[int, int, int], ...]
    hints_qa: tuple[tuple[str, tuple[str, ...], str | None], ...]
    qa_profile: str
    text_checks_version: str
    number_reader: str
    asr_model: str
    sv_model: str
    aligner_method_id: str
    measurement_key: str | None


@runtime_checkable
class KeyBuilder(Protocol):
    """Keys, ids and seeds (sections 10.2, 10.3; WP12). All keys: ``"sha256:"`` + 64 hex over RFC 8785
    canonical JSON of an object that carries a ``schema`` member (``names.*_SCHEMA``).

    Ids: ``rn_``/``tk_``/``an_`` + the first 16 hex characters of the render/delivery/analysis key.
    Job and design ids are ULIDs (``job_`` + ULID for jobs).

    **Seed** (section 10.3, pinned here byte for byte)::

        parts = ["narration-seed/v1", voice_hash, sha256_hex(engine_text as UTF-8), str(attempt)]
        digest = sha256(b"\\x00".join(p.encode("utf-8") for p in parts))
        seed = int.from_bytes(digest[0:4], "big") & 0x7FFFFFFF

    ``voice_hash`` here is the full reported form (``"sha256:"`` + 64 hex). The segment id is never an
    input.
    """

    def canonical_json(self, value: Any) -> bytes:
        """RFC 8785 canonical JSON of ``value``."""
        ...

    def voice_hash(
        self, *, model: str, clip_sha256: str, transcript: str, language: str, x_vector_only_mode: bool
    ) -> str:
        """H({schema: "narration.voice/v2", model, clip_sha256, transcript (NFC), language, x_vector_only_mode}).

        ``model`` is always ``names.MODEL_QWEN_BASE``, the cloning model's repo id: never a revision, an
        engine profile id or a profile hash. A voice's hash must survive a re-pin, because measurements are
        kept per voice and then per engine profile (sections 10.1, 15). ``language`` is ``names.LANGUAGE``.
        This method applies NFC to ``transcript`` itself, so callers pass the transcript as received.
        """
        ...

    def measurement_key(
        self, *, voice_hash: str, engine_profile_hash: str, corpus_version: str, settings: MeasurementConfig
    ) -> str:
        """H({schema, voice_hash, engine_profile_hash, corpus version, ladder settings}).

        ``corpus_version`` names the corpus set and its content: ``"<set id>@sha256:<hex>"`` from the
        ``MaterialSet``. The ladder settings are every field of ``settings`` except ``corpus``: ``seeds``,
        ``length_ladder_spoken_chars``, ``trend_band_max_chars``, ``pace_tol_min``, ``sim_warn_margin``,
        ``sim_fail_floor``.
        """
        ...

    def render_key(self, *, engine_profile_hash: str, voice_hash: str, engine_text: str, seed: int) -> str:
        """H({schema: "narration.render/v1", engine_profile_hash, voice_hash, engine_text, seed})."""
        ...

    def delivery_key(self, *, raw_sha256: str, profile: DeliveryProfile, tools: DeliveryTools) -> str:
        """H({schema, raw_sha256, delivery profile (every field of ``profile``), the resampler's and meter's
        names and versions, stretch: null})."""
        ...

    def analysis_key(self, inputs: AnalysisKeyInputs) -> str:
        """H({schema, every field of ``inputs``})."""
        ...

    def seed(self, *, voice_hash: str, engine_text: str, attempt: int) -> int:
        """The seed of section 10.3, exactly as in this class's docstring."""
        ...

    def render_id(self, render_key: str) -> str: ...

    def take_id(self, delivery_key: str) -> str: ...

    def analysis_id(self, analysis_key: str) -> str: ...

    def new_job_id(self) -> str:
        """``job_`` + a new ULID."""
        ...

    def new_design_id(self) -> str:
        """A new ULID."""
        ...


class Lease(Protocol):
    """A claim on one key (section 4 item 6): while held, nobody else produces that key."""

    @property
    def key(self) -> str: ...

    def renew(self, ttl_s: float) -> None: ...

    def release(self) -> None: ...


ClaimResult = Literal["claimed", "exists", "in_flight"]


@runtime_checkable
class Store(Protocol):
    """SQLite (WAL) rows plus content-addressed files under ``store_root`` (sections 15, 17.2; WP12).

    Every path is built by the store from validated ids and confined under the root (``realpath`` stays
    under it; Windows reserved names and reparse points are refused, through ``Platform.check_store_path``).
    Every file is written to a temp name and renamed; immutable files are made read-only.

    The store is the only channel between processes (the front-ends, the daemon, the operator CLI): there
    are no sockets (section 4). Its methods are synchronous; async callers run them in a thread
    (``asyncio.to_thread``). Stateless (sections 0.2, 2): it holds caches of work done and the service's
    own records, never a caller's script, choice, approval, pronunciation list or voice.
    """

    @property
    def root(self) -> Path: ...

    # ---- layout (section 15)
    def render_dir(self, render_id: str) -> Path: ...

    def take_dir(self, take_id: str) -> Path: ...

    def analysis_path(self, take_id: str, analysis_id: str) -> Path: ...

    def measurement_dir(self, voice_hash: str, engine_profile_id: str) -> Path: ...

    def design_dir(self, design_id: str, index: int) -> Path: ...

    def profile_dir(self, audio_sha256: str) -> Path: ...

    def job_dir(self, job_id: str) -> Path: ...

    def scratch_path(self, *parts: str) -> Path:
        """A path under ``scratch/`` for worker I/O; its parent directory exists."""
        ...

    # ---- the three cache layers and the service's own records
    def get_render(self, render_key: str) -> RenderRecord | None: ...

    def get_render_by_id(self, render_id: str) -> RenderRecord | None: ...

    def put_render(self, record: RenderRecord, raw_audio: Path) -> RenderRecord:
        """Publish ``raw_audio`` (moved into place) and ``render.json``; returns the record with its path."""
        ...

    def get_take(self, delivery_key: str) -> TakeRecord | None: ...

    def get_take_by_id(self, take_id: str) -> TakeRecord | None: ...

    def put_take(self, record: TakeRecord, delivery_audio: Path) -> TakeRecord: ...

    def get_analysis(self, analysis_key: str) -> AnalysisRecord | None: ...

    def get_analysis_by_id(self, analysis_id: str) -> AnalysisRecord | None: ...

    def analyses_of(self, take_id: str) -> tuple[AnalysisRecord, ...]:
        """Every analysis of a take (``narration://takes/{take_id}``)."""
        ...

    def put_analysis(self, record: AnalysisRecord) -> AnalysisRecord: ...

    def get_measurement(self, voice_hash: str, engine_profile_id: str) -> MeasurementRecord | None: ...

    def put_measurement(self, record: MeasurementRecord) -> MeasurementRecord: ...

    def measurements_of(self, voice_hash: str) -> tuple[MeasurementRecord, ...]:
        """A voice's measurements, one per engine profile (``narration://measurements/{voice_hash}``)."""
        ...

    def get_profile(self, audio_sha256: str, profile_version: str) -> ProfileRecord | None: ...

    def put_profile(self, record: ProfileRecord, pictures_dir: Path | None) -> ProfileRecord:
        """Publish ``profile.json`` and move the pictures in ``pictures_dir`` into place; the returned
        record has the pictures' final paths."""
        ...

    def put_candidate(self, candidate: Candidate, clip_audio: Path) -> Candidate:
        """Move the designed clip into ``designs/<design_id>/<cand>/`` and publish ``candidate.json``; the
        returned record has ``clip.path`` and ``clip.sha256`` filled in."""
        ...

    def get_design(self, design_id: str) -> tuple[Candidate, ...]: ...

    def add_provenance(self, entry: ProvenanceEntry) -> None:
        """Append to ``provenance.jsonl`` (never pruned) and index it."""
        ...

    def is_provenance(self, clip_sha256: str) -> bool: ...

    # ---- engine profiles and the canary (sections 6, 10.1, 15; DC-3): never collected
    def get_engine_profile(self, engine_profile_id: str) -> EngineProfile | None: ...

    def put_engine_profile(self, profile: EngineProfile) -> EngineProfile:
        """Write ``engines/<engine_profile_id>.json``. A pinned profile's hashed fields never change; a new
        pin is a new id (section 10.1)."""
        ...

    def list_engine_profiles(self) -> tuple[EngineProfile, ...]: ...

    def current_engine_profile(self, kind: EngineKind) -> EngineProfile | None:
        """The profile in use for ``base`` or ``design`` work, or None before ``engine pin``."""
        ...

    def set_current_engine_profile(self, kind: EngineKind, engine_profile_id: str) -> None: ...

    def put_canary_clip(self, engine_profile_id: str, audio: Path) -> AudioRef:
        """Move the designed canary clip to ``engines/<engine_profile_id>/canary.wav`` (immutable, never
        collected, re-hashed by ``verify``) and return its location and sha256."""
        ...

    # ---- the alignment benchmark (sections 6, 11.2): never collected
    def get_alignment_benchmark(self, method_id: str) -> AlignmentBenchmark | None: ...

    def put_alignment_benchmark(self, record: AlignmentBenchmark) -> AlignmentBenchmark: ...

    def current_alignment_benchmark(self) -> AlignmentBenchmark | None:
        """The benchmark of the aligner method in use, whose measured error every take reports."""
        ...

    # ---- claim-time re-check and in-flight waiting (section 4 item 6)
    def claim(self, key: str, holder: str, *, ttl_s: float) -> tuple[ClaimResult, Lease | None]:
        """``exists`` if the key's result is published; ``in_flight`` if another holder has a live lease;
        otherwise ``claimed`` with a lease. A lease past its TTL is free to claim."""
        ...

    def wait_for(self, key: str, *, timeout_s: float) -> bool:
        """Wait until the key's result exists (True) or its lease lapses or the timeout passes (False)."""
        ...

    # ---- jobs and the queue
    def create_job(self, record: JobRecord) -> tuple[JobRecord, bool]:
        """Insert a job, or return the active job with the same ``request_sha256``; the bool says whether
        it is new.

        ``idempotency_key`` deduplicates retries, and a retry is the same request. An active job with the
        same kind and key but a different ``request_sha256`` is refused: ``NarrationError(INVALID_ARGUMENT,
        field="idempotency_key")``, hint "use a new key for a different request" (plan.md DC-6).
        """
        ...

    def get_job(self, job_id: str) -> JobRecord | None: ...

    def update_job(self, job_id: str, *, expect_status: JobStatus | None = None, **changes: Any) -> JobRecord | None:
        """Change ``JobRecord`` fields (``changes`` are field names) and bump ``updated_at``. With
        ``expect_status``, the change is made only if the job is still in that status (compare-and-set);
        otherwise nothing changes and the result is None, so a blind write never reverts a cancel."""
        ...

    def queued_jobs(self) -> tuple[JobRecord, ...]:
        """Queued and running jobs, by priority (``interactive`` first) then FIFO."""
        ...

    def claim_job(self, job_id: str, holder: str) -> JobRecord | None:
        """Atomically move this job from ``queued`` to ``running`` for ``holder``; None if it is no longer
        queued. The scheduler chooses which (grouping by engine profile, affinity to the resident model)."""
        ...

    def claim_next_job(self, holder: str) -> JobRecord | None:
        """Atomically move the next queued job (priority, then FIFO) to ``running`` for the daemon."""
        ...

    def jobs_created_since(self, iso_time: str) -> int:
        """How many jobs were created at or after ``iso_time``, whatever their status now: the submit rate
        cap across every front-end process (``RATE_LIMITED``, DC-2's ``admission.rate``)."""
        ...

    # ---- the daemon's status and commands (sections 4, 4.1, 7.6): the store is the channel
    def get_daemon_status(self) -> DaemonStatus | None:
        """``run/daemon.json``, or None when no daemon has written one."""
        ...

    def put_daemon_status(self, status: DaemonStatus) -> None: ...

    def post_command(self, kind: DaemonCommandKind) -> DaemonCommand: ...

    def pending_commands(self) -> tuple[DaemonCommand, ...]: ...

    def commands_since(self, requested_at: str) -> tuple[DaemonCommand, ...]:
        """Every command posted at or after ``requested_at``, pending or done, in the order they were posted.

        ``requested_at`` is a store time (``2026-09-26T21:50:44.123Z``): fixed-width UTC to the millisecond,
        so text order is time order, and anything else raises ``ValueError``. The store never prunes
        commands, so one answered long ago is still listed. A daemon reads it on start for the stops posted
        after its launch that another daemon has already answered (section 4.1; contracts 1.6.4, WP30).
        """
        ...

    def complete_command(self, command_id: str, result: Mapping[str, Any] | None) -> DaemonCommand: ...

    def wait_for_command(self, command_id: str, *, timeout_s: float) -> DaemonCommand | None:
        """The command once the daemon has completed it, or None at the timeout."""
        ...

    # ---- retention (section 15)
    def touch(self, kind: str, ident: str) -> None:
        """Record a use, which restarts the item's retention period."""
        ...

    def gc(self, *, dry_run: bool = True) -> Mapping[str, Any]: ...

    def verify(self) -> Mapping[str, Any]:
        """Re-hash every immutable file; report mismatches."""
        ...


# ======================================================================== WP13: delivery (section 13)


DeliveryProfile = DeliveryConfig
"""The delivery settings (``[delivery]``, section 16): one class, so every field enters the delivery key."""


@dataclass(frozen=True, slots=True, kw_only=True)
class DeliveryOutput:
    """What post-processing produced: the file is at the ``out_path`` given."""

    samples: int
    sample_rate: int
    duration_s: float
    trim: Trim
    loudness: Loudness
    flags: tuple[Flag, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class SignalStats:
    """Signal facts QA needs (section 11.1 step 1), measured by the server from the audio files.

    ``voiced_start_s`` / ``voiced_end_s`` bound the speech in the delivery file (for pace, section 11.1
    step 9); ``longest_internal_silence_s`` is the longest pause strictly inside them. Speech here uses the
    trim's rule (section 13, DC-10: frame RMS with the mean removed, against max(p95 - 40 dB, -70 dBFS)).

    These rules feed QA verdicts, and the analysis key covers them only through ``names.QA_PROFILE``. A
    change to any of them (the frame, the percentile, the thresholds, what counts as clipping) needs a new
    QA profile version, or cached verdicts go stale (WP13's review).
    """

    raw_samples: int
    raw_sample_rate: int
    raw_clipping_fraction: float
    raw_nonfinite: bool
    raw_dc_offset: float
    delivery_duration_s: float
    voiced_start_s: float | None
    voiced_end_s: float | None
    longest_internal_silence_s: float


@runtime_checkable
class DeliveryProcessor(Protocol):
    """Deterministic post-processing on the CPU, thread-capped (section 13; WP13): relative trim, pinned
    48 kHz resampler, static gain to the target loudness (``DeliveryConfig.target_lufs``, default -23 LUFS;
    plan.md DC-8), fades, PCM_24, true peak on the final file."""

    @property
    def tools(self) -> DeliveryTools:
        """The resampler's and loudness meter's names and versions (they enter the delivery key)."""
        ...

    def process(self, raw_path: Path, out_path: Path, profile: DeliveryProfile) -> DeliveryOutput:
        """Write ``out_path`` (temp name then rename). The same input gives the same bytes, every time."""
        ...

    def signal_stats(self, raw_path: Path, delivery_path: Path) -> SignalStats: ...


# ======================================================================== WP14: QA logic (sections 8, 11.1, 11.3)


@dataclass(frozen=True, slots=True, kw_only=True)
class QaInputs:
    """Everything one take's verdict may depend on (section 11.1): the take and the request's inputs for it.

    The scorer computes the take's similarity to the anchor from ``embedding``. When ``measurement`` is
    given, it supplies the anchor, the similarity baseline and the pace curve, and ``anchor``,
    ``similarity`` and ``pace`` stay None. Work without a finished measurement passes what exists instead:
    the measurement's own ladder takes pass the calibration set's ``anchor`` and ``similarity`` (and no
    ``pace``); ``audition_pronunciation`` passes none, so no speaker or pace check applies.
    """

    segment: SegmentText
    hints: tuple[Hint, ...]
    """The hints used in this segment: the terms in ``segment``'s ``hints_applied``, the same set that
    ``AnalysisText.hints_used`` records and ``AnalysisKeyInputs.hints_qa`` covers. Never the request's whole
    hint list: a hint the key does not cover must not change a cached verdict (section 11.1)."""
    voice_transcript: str
    asr_text: str
    asr_words: tuple[AsrWord, ...]
    embedding: tuple[float, ...] | None
    alignment: Alignment
    signal: SignalStats
    hit_token_cap: bool
    measurement: MeasurementRecord | None
    anchor: Anchor | None = None
    similarity: SimilarityBaseline | None = None
    pace: Pace | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoredTake:
    """What the suggestion needs of one take (section 8)."""

    take_id: str
    attempt: int
    verdict: Verdict
    flags: tuple[Flag, ...]
    cues: tuple[CueTiming, ...]


@runtime_checkable
class QaScorer(Protocol):
    """QA logic, plain Python (plan.md P1; WP14): verdicts, flags, suggestions, consistency, listen-first."""

    @property
    def profile_version(self) -> str:
        """``names.QA_PROFILE``."""
        ...

    def score(self, inputs: QaInputs) -> QaResult:
        """The take's verdict (section 11.1, thresholds ``default.v3``): signal, text match with the word-count
        rule, exact spans (11.3), terms, insertions, speaker, pace; cue-alignment flags are included.

        The result is cached under the analysis key and reused by any request with the same key, so it must hold
        nothing the key does not cover: every flag has ``segment_id`` None, and each ``ExactResult`` carries
        its ``words``. The job assembler (WP36) stamps the current request's segment id on the flags and its
        offsets on the exact results, and adds ``start``/``end`` to each ``EXACT_SPAN_MISMATCH`` flag's
        ``details`` (the flag is cached with {cue, words, expected, heard, match}; §11.3 shows the caller
        {cue, start, end, expected, heard}).

        Raises ``errors.QaUnavailable`` when a check the inputs call for cannot run; never passes a take
        without it.
        """
        ...

    def suggest(self, takes: Sequence[ScoredTake]) -> tuple[str | None, Suggestion | None]:
        """Section 8's tiers, lowest attempt within the first tier that has a take."""
        ...

    def consistency(
        self, suggested: Sequence[tuple[str, tuple[float, ...]]], measurement: MeasurementRecord | None
    ) -> tuple[Consistency, tuple[Flag, ...]]:
        """Similarity of each suggested take to the set's centroid; outliers below the consistency baseline
        (p5 − 0.01) get ``SPK_OUTLIER`` (info). A report only, never a verdict."""
        ...

    def listen_first(self, segments: Sequence[SegmentResult]) -> tuple[ListenFirstItem, ...]:
        """Section 11.1's order: fails; exact-span and term flags; cue alignment; insertion, similarity, pace
        and consistency; text warnings and over-long segments. It reads each take's ``qa.flags``,
        ``alignment.flags`` and ``flags`` (the delivery and per-job flags) and each segment's ``flags``."""
        ...

    def report_md(self, results: Mapping[str, Any]) -> str:
        """``report.md`` from the assembled ``get_results`` object (its output schema fixes the layout):
        every flag, including replaced attempts, and each cue's received → engine text."""
        ...


# ======================================================================== WP15: cue alignment (section 11.2)


@dataclass(frozen=True, slots=True, kw_only=True)
class AlignTranscript:
    """The aligner's transcript (section 11.2 step 1): tokens in the model's alphabet, and for each token the
    (cue, word) it belongs to, or None for a word separator. A ``*`` token is the wildcard for a run of words
    the alphabet cannot spell (plan.md DC-11); it maps to its run's first word. ``dropped`` lists every word
    that cannot be spelled.

    ``cue_count`` is the segment's number of cues, so a cue with no alignable word still gets its entry
    (null times, ``CUE_UNALIGNED``). ``term_words`` are the (cue, word) pairs a hinted term covers; the
    Whisper cross-check leaves them out (section 11.2 step 6: names excluded).
    """

    tokens: tuple[str, ...]
    token_words: tuple[tuple[int, int] | None, ...]
    words: tuple[tuple[int, int, str], ...]
    cue_count: int
    term_words: tuple[tuple[int, int], ...] = ()
    dropped: tuple[tuple[int, int, str], ...] = ()


@runtime_checkable
class AlignerCore(Protocol):
    """The pure part of cue alignment (WP15): the transcript, the guard, snapping, confidence, cross-check.

    Cue times are never interpolated: an unplaceable cue has null times and ``CUE_UNALIGNED``.
    """

    @property
    def method_id(self) -> str: ...

    @property
    def model(self) -> str:
        """The pinned aligner model's repo id, set at construction (needed even when the worker failed)."""
        ...

    @property
    def revision(self) -> str: ...

    @property
    def device(self) -> str: ...

    def build_transcript(self, segment: SegmentText, hints: Sequence[Hint]) -> AlignTranscript: ...

    def guard(self, transcript: AlignTranscript, num_frames: int) -> bool:
        """``T ≥ L + R``: frames at least tokens plus repeats."""
        ...

    def guard_details(self, transcript: AlignTranscript, num_frames: int) -> dict[str, Any]:
        """The facts behind a failed guard, as the worker reports them: {reason, frames, tokens, repeats}."""
        ...

    def resolve(
        self,
        transcript: AlignTranscript,
        reply: AlignReply | None,
        audio: npt.NDArray[np.float32],
        sample_rate: int,
        asr_words: Sequence[AsrWord],
        measured_error: MeasuredError | None,
        *,
        error: Mapping[str, Any] | None = None,
    ) -> Alignment:
        """Turn the worker's token spans into cue and word times in delivery-file seconds: snap boundaries
        into pauses, score confidence, cross-check with Whisper.

        ``reply`` None means the worker reported ``ALIGNMENT_ERROR`` (pass its ``details`` as ``error``), or
        the caller's own guard failed (pass ``guard_details``): every cue unaligned and the take gets
        ``ALIGNMENT_ERROR`` (fail). An ``ok: false`` reply with any other code raises ``WorkerFailure``: it is
        no verdict on the take, and the caller records ``QA_UNAVAILABLE``."""
        ...


# ======================================================================== WP16: workers (Appendix A, section 4)


@runtime_checkable
class WorkerClient(Protocol):
    """The daemon's side of one worker process (Appendix A; WP16).

    ``request`` raises ``WorkerFailure`` for ``ok: false``, ``WorkerCrashed`` when the process exits or
    breaks the protocol, and ``WorkerTimeout`` when no reply comes in time (the worker is then stopped).
    """

    @property
    def role(self) -> WorkerRole: ...

    @property
    def pid(self) -> int | None: ...

    @property
    def hello(self) -> HelloReply | None:
        """The worker's ``hello`` reply (its role, backend and fingerprint) once started, else None."""
        ...

    def start(self) -> HelloReply:
        """Start the process and exchange ``hello``."""
        ...

    def request(self, op: str, payload: Mapping[str, Any], *, timeout_s: float) -> dict[str, Any]: ...

    def is_alive(self) -> bool: ...

    def close(self, *, timeout_s: float = 10.0) -> None:
        """``shutdown``, then terminate if it does not exit in time."""
        ...


# ======================================================================== WP19: platform (sections 4.1, 17.2)


@runtime_checkable
class Platform(Protocol):
    """Every OS-specific mechanism (WP19). Windows only in v1; on another OS each call raises
    ``errors.UnsupportedPlatform`` (``DAEMON_UNAVAILABLE``, not retryable), and ``doctor`` says so."""

    def singleton(self, store_root: Path) -> AbstractContextManager[bool]:
        """Hold the daemon's singleton lock keyed on the store path; the value says whether it was acquired."""
        ...

    def spawn_detached(self, argv: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> int:
        """Start a detached process (section 4.1) and return its pid; raises ``NarrationError``
        (``DAEMON_UNAVAILABLE``) when breakaway is refused. Never falls back to a non-detached start."""
        ...

    def kill_on_close_group(self) -> AbstractContextManager[Callable[[int], None]]:
        """A group (a Job Object) whose processes die with the daemon; yields a function that adds a pid."""
        ...

    def set_below_normal_priority(self, pid: int) -> None: ...

    def check_readable_path(self, path: str) -> Path:
        """Section 17.3: absolute, on a local drive, resolving to a regular file; else ``PATH_NOT_ALLOWED``."""
        ...

    def check_store_path(self, path: Path, root: Path) -> Path:
        """Section 17.2: ``realpath`` under ``root``, no reserved names, no reparse points."""
        ...

    def free_disk_bytes(self, path: Path) -> int: ...


# ======================================================================== WP17 / WP36: the backend the front-end calls


ProgressCallback = Callable[[float, float | None, str | None], Awaitable[None]]
"""``progress(progress, total, message)``: sends ``notifications/progress`` for the in-flight request."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceContent:
    uri: str
    mime_type: str
    text: str
    ttl_ms: int
    extra: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    """What the MCP front-end calls (WP17 builds the front-end against a fake; WP36 wires the real one).

    Each tool method takes the arguments **already validated** against the tool's input schema and returns
    the tool's structured result (a JSON object matching its output schema). A tool error is raised as
    ``NarrationError``; the front-end turns it into ``isError: true``. Cancelling the MCP request cancels
    only the awaiting call (``get_job``'s wait), never the job. Store calls run in a thread
    (``asyncio.to_thread``); ``get_job``'s long-poll re-reads the job until it changes or ``wait_s`` ends.
    """

    async def get_server_status(self, args: Mapping[str, Any]) -> dict[str, Any]: ...

    async def release_gpu(self, args: Mapping[str, Any]) -> dict[str, Any]: ...

    async def get_job(self, args: Mapping[str, Any], progress: ProgressCallback | None) -> dict[str, Any]: ...

    async def get_results(self, args: Mapping[str, Any]) -> dict[str, Any]: ...

    async def cancel_job(self, args: Mapping[str, Any]) -> dict[str, Any]: ...

    async def design_voice(self, args: Mapping[str, Any]) -> dict[str, Any]: ...

    async def profile_voice(self, args: Mapping[str, Any]) -> dict[str, Any]: ...

    async def measure_voice(self, args: Mapping[str, Any]) -> dict[str, Any]: ...

    async def check_text(self, args: Mapping[str, Any]) -> dict[str, Any]: ...

    async def audition_pronunciation(self, args: Mapping[str, Any]) -> dict[str, Any]: ...

    async def submit_job(self, args: Mapping[str, Any]) -> dict[str, Any]: ...

    async def read_resource(self, uri: str) -> ResourceContent:
        """A ``narration://`` resource (section 7.7); raises ``NarrationError(NOT_FOUND)`` for a missing one,
        which the front-end reports as JSON-RPC -32602."""
        ...


# ======================================================================== WP18: the service's own material


@runtime_checkable
class MaterialLoader(Protocol):
    """Reads the service's own material (``material/``, WP18) and checks every file against its manifest.

    Raises ``errors.MaterialError`` when a set is missing, malformed, or its bytes do not match the manifest.
    The sets are the service's own text, never a caller's (sections 3.2, 10.1, 11.2).
    """

    def corpus(self, set_id: str) -> MaterialSet:
        """The calibration corpus (``narration-en.v1``): calibration paragraphs and the length ladder."""
        ...

    def benchmark(self, set_id: str) -> MaterialSet:
        """The alignment benchmark (``alignment-en.v1``), each paragraph with its expected boundaries."""
        ...

    def demo(self, set_id: str) -> MaterialSet:
        """The Phase 4 demo script, with its hints."""
        ...

    def canary(self, set_id: str) -> CanaryMaterial: ...
