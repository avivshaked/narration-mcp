"""Records for the store's tests, built with the service's own key functions so ids and keys agree.

Audio here is a few bytes standing in for a WAV: the store never parses audio, it moves and hashes it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from narration import keys
from narration.config import DeliveryConfig, MeasurementConfig
from narration.contracts import names
from narration.contracts.interfaces import AnalysisKeyInputs
from narration.contracts.models import (
    Alignment,
    AlignmentBenchmark,
    AnalysisRecord,
    AnalysisText,
    AnalysisVersions,
    Anchor,
    AudioRef,
    BenchmarkRef,
    CalibrationTake,
    CanaryPin,
    CanaryRecord,
    Candidate,
    CrossCheck,
    DaemonStatus,
    DeliveryAudio,
    DeliveryTools,
    Determinism,
    EngineProfile,
    EngineRef,
    ErrorStats,
    GpuStatus,
    JobRecord,
    LadderRung,
    LadderSeed,
    Licence,
    LintResult,
    Loudness,
    MeasurementRecord,
    Pace,
    PaceTrend,
    ProfileMeasurements,
    ProfilePictures,
    ProfileRecord,
    Progress,
    QaMetrics,
    QaResult,
    QaThresholds,
    RawAudio,
    RenderEngine,
    RenderRecord,
    RenderVoice,
    SimilarityBaseline,
    TakeRecord,
    TextChecksInfo,
    TranscriptCheck,
    Trim,
)
from narration.contracts.names import JobKind, JobStatus, Priority
from narration.store import NarrationStore

ENGINE_HASH = "sha256:" + "9e" * 32
CLIP_SHA = "5b" * 32
TOOLS = DeliveryTools(resampler="soxr 0.5.0", loudness_meter="pyloudnorm 0.1.1", post=names.POST_RULES)
VOICE_HASH = keys.voice_hash(
    model=names.MODEL_QWEN_BASE,
    clip_sha256=CLIP_SHA,
    transcript="Good bread asks for patience.",
    language=names.LANGUAGE,
    x_vector_only_mode=False,
)
T0 = "2026-09-26T12:00:00.000Z"
METHOD_ID = "ctc-snap/wav2vec2@abc"
"""The aligner method the test store is configured with (``alignment_method_id``)."""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audio_bytes(label: str, size: int = 4096) -> bytes:
    """Deterministic stand-in audio for ``label``."""
    seed = label.encode("utf-8")
    out = bytearray()
    counter = 0
    while len(out) < size:
        out += hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        counter += 1
    return bytes(out[:size])


def scratch_file(store: NarrationStore, name: str, data: bytes) -> Path:
    path = store.scratch_path("test", name)
    path.write_bytes(data)
    return path


def render_record(engine_text: str = "Before dawn, the reef belongs to the shrimp.", attempt: int = 0) -> RenderRecord:
    seed = keys.seed(voice_hash=VOICE_HASH, engine_text=engine_text, attempt=attempt)
    key = keys.render_key(engine_profile_hash=ENGINE_HASH, voice_hash=VOICE_HASH, engine_text=engine_text, seed=seed)
    return RenderRecord(
        render_id=keys.render_id(key),
        render_key=key,
        voice=RenderVoice(voice_hash=VOICE_HASH, clip_sha256=CLIP_SHA),
        engine=RenderEngine(
            engine_profile_id=names.ENGINE_PROFILE_BASE,
            engine_profile_hash=ENGINE_HASH,
            model_repo=names.MODEL_QWEN_BASE,
            model_revision="0" * 40,
            non_streaming_mode=False,
            generation={"do_sample": True, "max_new_tokens": 8192},
        ),
        engine_text=engine_text,
        seed=seed,
        attempt=attempt,
        raw=RawAudio(path="", sha256="", sample_rate=24000, samples=2048),
        hit_token_cap=False,
        gen_s=1.5,
        rtf=2.3,
        canary=CanaryRecord(batch_status="hash_match"),
        licence=Licence(generation_model="Apache-2.0"),
    )


def take_record(raw_sha256: str, render_id: str) -> TakeRecord:
    key = keys.delivery_key(raw_sha256=raw_sha256, profile=DeliveryConfig(), tools=TOOLS)
    return TakeRecord(
        take_id=keys.take_id(key),
        delivery_key=key,
        render_id=render_id,
        delivery=DeliveryAudio(path="", sha256="", sample_rate=48000, samples=4096, duration_s=4096 / 48000),
        trim=Trim(head_s=0.1, tail_s=0.2, pad_s=0.08),
        loudness=Loudness(measured_lufs=-16.0, gain_db=5.4, true_peak_dbtp=-2.3, ceiling_applied=False),
        tools=TOOLS,
    )


def analysis_record(take: TakeRecord, *, qa_profile: str = names.QA_PROFILE) -> AnalysisRecord:
    inputs = AnalysisKeyInputs(
        delivery_sha256=take.delivery.sha256,
        spoken_text="Before dawn, the reef belongs to the shrimp.",
        cue_spans=((0, 44),),
        exact_spans=(),
        hints_qa=(),
        qa_profile=qa_profile,
        text_checks_version=names.TEXT_CHECKS_VERSION,
        number_reader=names.NUMBER_READER,
        asr_model=names.MODEL_ASR + "@" + "1" * 40,
        sv_model=names.MODEL_SV + "@" + "2" * 40,
        aligner_method_id=names.ALIGNMENT_METHOD,
        measurement_key=None,
    )
    key = keys.analysis_key(inputs)
    return AnalysisRecord(
        analysis_id=keys.analysis_id(key),
        analysis_key=key,
        take_id=take.take_id,
        text=AnalysisText(
            cues=(), text_checks=TextChecksInfo(version=names.TEXT_CHECKS_VERSION, rules_sha256="0" * 64)
        ),
        versions=AnalysisVersions(
            qa_profile=qa_profile,
            asr=inputs.asr_model,
            sv=inputs.sv_model,
            aligner_method=names.ALIGNMENT_METHOD,
            number_reader=names.NUMBER_READER,
            measurement=None,
        ),
        alignment=Alignment(
            method=names.ALIGNMENT_METHOD,
            model=names.MODEL_ALIGNER,
            revision="3" * 40,
            device="cpu",
            cross_check=CrossCheck(model=names.MODEL_ASR, max_disagreement_s=0.06),
            measured_error=None,
            cues=(),
        ),
        qa=QaResult(
            verdict="pass",
            transcript="before dawn the reef belongs to the shrimp",
            exact=(),
            terms=(),
            metrics=QaMetrics(
                wer_raw=0.0,
                wer_adj=0.0,
                word_errors=0,
                exact_ok=True,
                spk_sim_anchor=0.98,
                spoken_wpm=150.0,
                expected_spoken_wpm=148.0,
                head_insertion_words=0,
                end_insertion_words=0,
                longest_silence_s=0.4,
            ),
            thresholds=QaThresholds(spk_warn=0.97, spk_fail=0.9, pace_tol=0.17),
        ),
        licence=Licence(aligner="apache-2.0"),
    )


def measurement_record(
    *,
    corpus_hex: str = "c0" * 32,
    engine_id: str = names.ENGINE_PROFILE_BASE,
    calibration_takes: tuple[str, ...] = (),
    ladder_takes: tuple[str | None, ...] = (),
) -> MeasurementRecord:
    """A measurement; ``calibration_takes`` and ``ladder_takes`` are the take ids it names."""
    engine_hash = ENGINE_HASH if engine_id == names.ENGINE_PROFILE_BASE else keys.hash_key({"schema": engine_id})
    key = keys.measurement_key(
        voice_hash=VOICE_HASH,
        engine_profile_hash=engine_hash,
        corpus_version=f"{names.CORPUS}@sha256:{corpus_hex}",
        settings=MeasurementConfig(),
    )
    return MeasurementRecord(
        voice_hash=VOICE_HASH,
        clip_sha256=CLIP_SHA,
        engine_profile=EngineRef(id=engine_id, hash=engine_hash),
        measurement_key=key,
        transcript_check=TranscriptCheck(heard="good bread asks for patience", wer=0.0, ok=True),
        corpus=names.CORPUS,
        similarity=SimilarityBaseline(anchor_p5=0.98, anchor_p50=0.987, consistency_p5=0.983),
        pace=Pace(
            method=names.PACE_METHOD,
            trend=PaceTrend(intercept_cps=16.2, per_100_chars=0.4, band_max_chars=300),
            tol=0.17,
            curve=(),
            speaking_share=0.9,
        ),
        max_segment_chars=450,
        max_segment_seconds=31.5,
        ladder=(
            (
                LadderRung(
                    chars=80,
                    seeds=tuple(
                        LadderSeed(seed=i, attempt=0, take_id=t, cps=16.5, wer_adj=0.0, sim=0.98, verdict="pass")
                        for i, t in enumerate(ladder_takes)
                    ),
                    passes=True,
                ),
            )
            if ladder_takes
            else ()
        ),
        anchor=Anchor(model=names.MODEL_SV, dim=3, embedding=(0.1, 0.2, 0.3)),
        calibration=tuple(
            CalibrationTake(paragraph_id=f"cal-{i}", seed=i, attempt=0, take_id=t)
            for i, t in enumerate(calibration_takes)
        ),
        measured_at=T0,
    )


def profile_record(audio_sha256: str, *, version: str = names.PROFILE_VERSION, pictures: bool = True) -> ProfileRecord:
    return ProfileRecord(
        audio_sha256=audio_sha256,
        profile_version=version,
        measurements=ProfileMeasurements(
            duration_s=6.2,
            pitch_median_hz=110.0,
            pitch_p10_hz=90.0,
            pitch_p90_hz=140.0,
            pitch_range_st=7.6,
            speaking_rate_wpm=None,
            pause_ratio=0.2,
            loudness_lufs=-20.0,
            spectral_centroid_hz=1500.0,
            hnr_db=15.0,
            cpps_db=6.0,
        ),
        pictures=ProfilePictures(
            spectrogram="spectrogram.png" if pictures else "", pitch="pitch.png" if pictures else ""
        ),
    )


def candidate(design_id: str, index: int, profile: ProfileRecord | None = None) -> Candidate:
    return Candidate(
        design_id=design_id,
        index=index,
        clip=AudioRef(path="", sha256=""),
        transcript="Good bread asks for patience.",
        transcript_check=None,
        description="A calm, warm narrator with an unhurried pace.",
        description_sha256=sha(b"A calm, warm narrator with an unhurried pace."),
        design_text="Good bread asks for patience.",
        seed=2001 + index,
        engine_profile=EngineRef(id=names.ENGINE_PROFILE_DESIGN, hash="sha256:" + "d1" * 32),
        lint=LintResult(policy="warn", findings=()),
        profile=profile,
    )


def job_record(
    job_id: str,
    *,
    request_sha256: str = "a" * 64,
    kind: JobKind = "generate",
    status: JobStatus = "queued",
    priority: Priority = "batch",
    idempotency_key: str | None = None,
) -> JobRecord:
    return JobRecord(
        job_id=job_id,
        kind=kind,
        request={"segments": [{"segment_id": "p01", "text": "Before dawn."}]},
        request_sha256=request_sha256,
        label=None,
        priority=priority,
        status=status,
        phase=None,
        round=0,
        progress=Progress(done_s=0.0, total_s=0.0, fraction=0.0, segments_done=0, segments_total=1),
        outcome=None,
        error=None,
        idempotency_key=idempotency_key,
        created_at=T0,
        updated_at=T0,
    )


def engine_profile(engine_id: str = names.ENGINE_PROFILE_BASE, *, hash_: str = ENGINE_HASH) -> EngineProfile:
    return EngineProfile(
        engine_profile_id=engine_id,
        hash=hash_,
        model_repo=names.MODEL_QWEN_BASE,
        model_revision="0" * 40,
        snapshot_dir="<models_root>/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/" + "0" * 40,
        weights={"model.safetensors": "e" * 64},
        worker_project="workers/qwen3tts",
        uv_lock_sha256="f" * 64,
        packages={"qwen-tts": "0.1.1"},
        dtype="bfloat16",
        determinism=Determinism(
            attn_implementation="sdpa",
            tf32=False,
            cudnn_deterministic=True,
            cudnn_benchmark=False,
            deterministic_algorithms="warn_only",
            cublas_workspace_config=":4096:8",
        ),
        settings={"non_streaming_mode": False},
        capabilities={"controls": {"pace": False}},
        licence="Apache-2.0",
        vram_need_mb=7000,
    )


def canary_pin(clip: AudioRef) -> CanaryPin:
    return CanaryPin(
        material="canary-en.v1",
        clip=clip,
        transcript="The canary sings the same song.",
        seed=7,
        raw_sha256="c" * 64,
        embedding=(0.5, 0.5),
        threshold=0.95,
        pinned_at=T0,
    )


def alignment_benchmark(method_id: str = METHOD_ID, *, p50: float = 0.03) -> AlignmentBenchmark:
    return AlignmentBenchmark(
        method_id=method_id,
        model=names.MODEL_ALIGNER,
        revision="3" * 40,
        snap={"frame_s": 0.02},
        benchmark=BenchmarkRef(id=names.BENCHMARK, sha256="b" * 64, description="the service's own benchmark"),
        measured_error=ErrorStats(p50_s=p50, p95_s=0.09, n=30),
        by_kind={"pause": ErrorStats(p50_s=0.02, p95_s=0.05, n=15)},
        measured_at=T0,
    )


def daemon_status() -> DaemonStatus:
    return DaemonStatus(
        state="idle",
        pid=1234,
        started_at=T0,
        workers=(),
        current_job=None,
        gpu=GpuStatus(name=None, total_mb=None, free_mb=None, in_use=False, holder=None, unload_in_s=None),
        est_drain_s=None,
        updated_at=T0,
    )
