"""Planning a generation request over the three cache layers (design sections 7.3, 10.2, 10.3 and 12).

Every attempt's keys follow from the request and the service's pins alone, exactly as the job engine
computes them (``narration.jobs.stages``): the seed and the ``render_key`` from the voice hash, the engine
text and the attempt number; the ``delivery_key`` from the raw audio's sha256, the delivery profile and the
post-processing tools; the ``analysis_key`` from the delivery's sha256 and the request's inputs for it. So
the plan walks the layers the daemon will walk: a render hit with a delivery miss is post-processing only; a
take hit with an analysis miss is scoring only.

The plan covers round 0, the requested attempts (a segment's ``attempts``, or ``0..takes-1``). Retakes depend
on verdicts not made yet, so they are not planned; a cached take that is a retake trigger adds its cached
retakes when the daemon walks them.

The analysis key names the QA models and the aligner's method id (``AnalysisPins``), which the installation
pins (WP32 assembles them). Without them, the analysis layer cannot be looked up here, and every attempt
counts as needing its analysis: the plan is then an upper estimate, and the daemon still finds what is cached.

Estimates are advice, from the voice's pace curve (``narration.jobs.plan.estimated_audio_s``) and DC-2's
throughput figures (``narration.jobs.admission``): never a promise and never part of a key.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from narration import keys
from narration.config import DeliveryConfig
from narration.contracts.interfaces import AnalysisKeyInputs, Store
from narration.contracts.models import (
    AnalysisRecord,
    DeliveryTools,
    EngineProfile,
    Hint,
    MeasurementRecord,
    RenderRecord,
    SegmentIn,
    SegmentText,
    TakeRecord,
)
from narration.jobs.admission import MODEL_LOAD_S, WALL_PER_AUDIO_S
from narration.jobs.plan import analysis_key_inputs as engine_key_inputs
from narration.jobs.plan import estimated_audio_s, hints_used, requested_attempts

POST_SHARE: float = 1 / 3
"""The share of an attempt's wall time that each layer takes (the job engine's progress counts a third per
layer made)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalysisPins:
    """What the analysis key names of the QA group and the aligner (section 10.2): the ASR and speaker models
    as ``repo@revision`` (``ModelPin.name``), the aligner's method id, and the QA profile and number reader
    versions. The installation pins them (WP32).

    ``aligner_revision`` is the aligner model's pinned revision, which its method id already names. It enters no
    key: ``get_server_status`` reports it as ``alignment.revision`` before the alignment benchmark has run."""

    asr_model: str
    sv_model: str
    aligner_method_id: str
    qa_profile: str
    number_reader: str
    aligner_revision: str | None = None


@dataclass(slots=True, kw_only=True)
class AttemptPlan:
    """One requested attempt and what the cache holds for it."""

    segment_id: str
    attempt: int
    seed: int
    render_key: str
    render: RenderRecord | None = None
    take: TakeRecord | None = None
    analysis: AnalysisRecord | None = None
    est_audio_s: float = 0.0


@dataclass(slots=True, kw_only=True)
class Plan:
    """What a request needs: per layer, how many attempts the cache does not answer, and the estimates."""

    attempts: list[AttemptPlan] = field(default_factory=list)
    segments_total: int = 0
    segments_cached: int = 0

    @property
    def renders_needed(self) -> int:
        """Attempts with no render in the cache."""
        return sum(1 for a in self.attempts if a.render is None)

    @property
    def deliveries_needed(self) -> int:
        """Attempts with no take (delivery) in the cache."""
        return sum(1 for a in self.attempts if a.take is None)

    @property
    def analyses_needed(self) -> int:
        """Attempts with no analysis the plan could find (all of them without ``AnalysisPins``)."""
        return sum(1 for a in self.attempts if a.analysis is None)

    @property
    def est_audio_s(self) -> float:
        """Seconds of audio still to render (at the voice's pace curve)."""
        return round(sum(a.est_audio_s for a in self.attempts if a.render is None), 1)

    def work_wall_s(self, wall_per_audio_s: float = WALL_PER_AUDIO_S) -> float:
        """Wall seconds of this request's own work: each layer a third of an attempt's time, plus a load of
        each model group the plan needs."""
        audio = 0.0
        for a in self.attempts:
            layers = (a.render is None) + (a.take is None) + (a.analysis is None)
            audio += a.est_audio_s * layers * POST_SHARE
        loads = (1 if self.renders_needed else 0) + (1 if self.analyses_needed else 0)
        return round(audio * wall_per_audio_s + loads * MODEL_LOAD_S, 1)

    def as_json(self, *, queue_position: int | None, wait_s: float) -> dict[str, Any]:
        """``submit_job``'s ``plan`` (section 7.3); ``est_wall_s`` includes the wait for the jobs ahead."""
        out: dict[str, Any] = {
            "segments_total": self.segments_total,
            "segments_cached": self.segments_cached,
            "renders_needed": self.renders_needed,
            "deliveries_needed": self.deliveries_needed,
            "analyses_needed": self.analyses_needed,
            "est_audio_s": self.est_audio_s,
            "est_wall_s": round(wait_s + self.work_wall_s(), 1),
        }
        if queue_position is not None:
            out["queue_position"] = queue_position
        return out


def analysis_key_inputs(
    *,
    take: TakeRecord,
    text: SegmentText,
    hints: Sequence[Hint],
    pins: AnalysisPins,
    measurement: MeasurementRecord,
) -> AnalysisKeyInputs:
    """The analysis key's inputs (section 10.2), built by the job engine's own builder
    (``narration.jobs.plan.analysis_key_inputs``, which ``narration.jobs.stages.Stages.key_inputs`` uses) from
    ``pins``. ``hints`` are the hints the segment uses."""
    return engine_key_inputs(
        take=take,
        text=text,
        hints=hints,
        qa_profile=pins.qa_profile,
        text_checks_version=text.text_checks.version if text.text_checks is not None else "",
        number_reader=pins.number_reader,
        asr_model=pins.asr_model,
        sv_model=pins.sv_model,
        aligner_method_id=pins.aligner_method_id,
        measurement_key=measurement.measurement_key,
    )


def plan_request(
    store: Store,
    *,
    segments: Sequence[SegmentIn],
    texts: Sequence[SegmentText],
    hints: Sequence[Hint],
    takes: int,
    voice_hash: str,
    profile: EngineProfile,
    measurement: MeasurementRecord,
    delivery: DeliveryConfig,
    tools: DeliveryTools,
    pins: AnalysisPins | None,
) -> Plan:
    """Walk the cache for every requested attempt of every segment (see the module docstring)."""
    plan = Plan(segments_total=len(texts))
    for segment, text in zip(segments, texts, strict=True):
        used = hints_used(text, hints)
        est = estimated_audio_s(text, measurement)
        complete = True
        for number in requested_attempts(segment, takes):
            seed = keys.seed(voice_hash=voice_hash, engine_text=text.engine_text, attempt=number)
            render_key = keys.render_key(
                engine_profile_hash=profile.hash, voice_hash=voice_hash, engine_text=text.engine_text, seed=seed
            )
            attempt = AttemptPlan(
                segment_id=text.segment_id, attempt=number, seed=seed, render_key=render_key, est_audio_s=est
            )
            attempt.render = store.get_render(render_key)
            if attempt.render is not None:
                delivery_key = keys.delivery_key(raw_sha256=attempt.render.raw.sha256, profile=delivery, tools=tools)
                attempt.take = store.get_take(delivery_key)
            if attempt.take is not None and pins is not None and text.text_checks is not None:
                inputs = analysis_key_inputs(
                    take=attempt.take, text=text, hints=used, pins=pins, measurement=measurement
                )
                attempt.analysis = store.get_analysis(keys.analysis_key(inputs))
            complete = complete and attempt.analysis is not None
            plan.attempts.append(attempt)
        if complete:
            plan.segments_cached += 1
    return plan


__all__ = ["AnalysisPins", "AttemptPlan", "Plan", "analysis_key_inputs", "plan_request"]
