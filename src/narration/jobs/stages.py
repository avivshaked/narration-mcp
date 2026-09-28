"""The three stages of a round (design section 4 items 4 and 6, sections 10 and 11): render on Qwen,
post-process on the CPU, score on the QA group, each answered by the cache first.

Every attempt's keys follow from the request and the service's pins alone (section 10.2): the seed and the
``render_key`` from the voice, the engine text and the attempt number; the ``delivery_key`` from the raw
audio's sha256 and the delivery profile; the ``analysis_key`` from the delivery's sha256 and the request's
inputs for it. Each layer is looked up when an attempt is planned, and again at claim time. A result that
exists is used. One that another holder is producing is waited for (its lease) and never produced twice.
Otherwise this engine takes the lease, makes the layer and publishes it.

A stage raises what its worker raises (``WorkerFailure``, ``WorkerCrashed``, ``WorkerTimeout``); the
engine's failure handling (``failures``) turns that into a retry, a flag or a job error.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import soundfile

from narration import keys
from narration.align import has_letters
from narration.contracts import codes, names
from narration.contracts.errors import NarrationError, WorkerCrashed, WorkerFailure, WorkerTimeout
from narration.contracts.interfaces import AnalysisKeyInputs, QaInputs, WorkerClient
from narration.contracts.models import (
    AnalysisRecord,
    AnalysisText,
    AnalysisVersions,
    Anchor,
    CanaryRecord,
    CueText,
    DeliveryAudio,
    Licence,
    MeasurementRecord,
    Pace,
    RawAudio,
    RenderEngine,
    RenderRecord,
    RenderVoice,
    SimilarityBaseline,
    TakeRecord,
)
from narration.contracts.names import GpuHolder
from narration.contracts.worker import AlignReply, AsrWord, HelloReply

from .core import LOST_STATE, ActiveRun, EngineCore, worker_code
from .gpu import GroupNeed, Readiness
from .host import GROUP_ROLES, ResidencyError, RunnerHost
from .pins import call_cap, generation, qwen_load_payload
from .state import Attempt, JobRun, Outcome, SegmentWork, label

log = logging.getLogger(__name__)

RENDER_LEASE_S: Final = 300.0
POST_LEASE_S: Final = 300.0
SCORE_LEASE_S: Final = 300.0
"""A lease's TTL. It is renewed every ``LEASE_RENEW_S`` while its work runs (the engine's ``LeaseKeeper``), so
the TTL only bounds how long a lease outlives a daemon that died; its successor, under the same holder name,
may claim the key again at once."""
LEASE_RENEW_S: Final = 60.0
PREPARE_TIMEOUT_S: Final = 300.0
QA_TIMEOUT_S: Final = 900.0
FRAMES_PER_SECOND: Final = 12.5
"""Qwen3-TTS-12Hz's codec: a call capped at N tokens makes at most N / 12.5 s of audio."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ScoringFacts:
    """What one segment's takes are judged against (``QaInputs``): the voice's finished measurement, or, for work
    without one, whatever exists of it (an anchor and a similarity baseline, a pace curve). Missing facts are
    not checked (``narration.qa.scorer.voice_facts``)."""

    measurement: MeasurementRecord | None = None
    anchor: Anchor | None = None
    similarity: SimilarityBaseline | None = None
    pace: Pace | None = None


class Stages:
    """Render, post-process and score one attempt each, answered by the cache first (see the module
    docstring), and the cache lookups and model loads they need."""

    def __init__(self, core: EngineCore) -> None:
        self.core = core

    # ------------------------------------------------------------------ keys and the cache
    def plan_attempt(
        self, host: RunnerHost, run: JobRun, seg: SegmentWork, slot: int, number: int, round_: int
    ) -> Attempt:
        """A new attempt, its keys from the request alone, looked up in the cache layer by layer."""
        engine_text = seg.text.engine_text
        seed = keys.seed(voice_hash=run.voice_hash, engine_text=engine_text, attempt=number)
        render_key = keys.render_key(
            engine_profile_hash=run.profile.hash, voice_hash=run.voice_hash, engine_text=engine_text, seed=seed
        )
        attempt = Attempt(segment=seg.index, slot=slot, attempt=number, round=round_, seed=seed, render_key=render_key)
        found = host.store.get_render(render_key)
        if found is not None:
            host.store.touch("render", found.render_id)
            attempt.render = found
            self.find_take(host, run, seg, attempt)
        return attempt

    def delivery_key(self, render: RenderRecord) -> str:
        """The take's key: the raw audio's sha256 and the delivery profile (section 10.2)."""
        core = self.core
        return keys.delivery_key(
            raw_sha256=render.raw.sha256, profile=core.config.delivery, tools=core.parts.delivery.tools
        )

    def find_take(self, host: RunnerHost, run: JobRun, seg: SegmentWork, attempt: Attempt) -> None:
        """Look up the attempt's take in the cache, and then its analysis."""
        assert attempt.render is not None
        take = host.store.get_take(self.delivery_key(attempt.render))
        if take is None:
            return
        attempt.take = take
        self.find_analysis(host, run, seg, attempt)

    def find_analysis(self, host: RunnerHost, run: JobRun, seg: SegmentWork, attempt: Attempt) -> None:
        """Set the attempt's analysis key, and look its analysis up in the cache."""
        assert attempt.take is not None
        attempt.analysis_key = keys.analysis_key(self.key_inputs(run, seg, attempt.take))
        found = host.store.get_analysis(attempt.analysis_key)
        if found is not None:
            host.store.touch("analysis", found.analysis_id)  # also restarts its take's retention
            attempt.analysis = found
        else:
            host.store.touch("take", attempt.take.take_id)

    def measurement_key(self, run: JobRun, seg: SegmentWork) -> str | None:
        """The measurement key a segment's analyses name (``AnalysisKeyInputs.measurement_key``, section 10.2).

        ``generate`` and ``analyse``: the key of the voice's finished measurement. A kind that scores takes
        without one overrides this, with ``scoring_facts``: ``measure_voice`` names the key of the measurement
        it is building for its ladder takes, and None for its calibration takes, which no speaker or pace
        check judges (the contract's rule).
        """
        if run.measurement is None:
            raise RuntimeError(f"job {run.job_id} scores takes but has no measurement to judge them against")
        return run.measurement.measurement_key

    def scoring_facts(self, run: JobRun, seg: SegmentWork) -> ScoringFacts:
        """What a segment's takes are judged against (``QaInputs``): for ``generate`` and ``analyse``, the voice's
        finished measurement, which supplies the anchor, the similarity baseline and the pace curve. A kind that
        scores takes without one overrides this (see ``measurement_key``)."""
        if run.measurement is None:
            raise RuntimeError(f"job {run.job_id} scores takes but has no measurement to judge them against")
        return ScoringFacts(measurement=run.measurement)

    def key_inputs(self, run: JobRun, seg: SegmentWork, take: TakeRecord) -> AnalysisKeyInputs:
        """The analysis key's inputs (section 10.2): the take, the request's inputs for it, the service's pins."""
        parts, text = self.core.parts, seg.text
        return AnalysisKeyInputs(
            delivery_sha256=take.delivery.sha256,
            spoken_text=text.spoken_text,
            cue_spans=tuple(c.spoken_span for c in text.cues),
            exact_spans=tuple((c.index, e.words[0], e.words[1]) for c in text.cues for e in c.exact),
            hints_qa=tuple((h.term, h.asr_aliases, h.align_as) for h in seg.hints),
            qa_profile=parts.scorer.profile_version,
            text_checks_version=(text.text_checks or parts.text.checks_info).version,
            number_reader=parts.scorer.number_reader,
            asr_model=parts.qa_pins.asr.name,
            sv_model=parts.qa_pins.sv.name,
            aligner_method_id=parts.aligner.method_id,
            measurement_key=self.measurement_key(run, seg),
        )

    # ------------------------------------------------------------------ making a model group ready
    def qwen_need(self, run: ActiveRun) -> GroupNeed:
        """What rendering needs loaded: the job's Qwen engine profile on the Qwen group (Base for a clone; a
        design job's run names VoiceDesign)."""
        profile = run.profile
        return GroupNeed(
            group="qwen",
            key=profile.hash,
            label=profile.engine_profile_id,
            payload=qwen_load_payload(profile, self.core.config.gpu.device),
            need_mb=profile.vram_need_mb,
            cublas_workspace_config=profile.determinism.cublas_workspace_config,
        )

    def qa_need(self) -> GroupNeed:
        """What scoring needs loaded: the QA group's pinned models."""
        pins = self.core.parts.qa_pins
        return GroupNeed(
            group="qa",
            key=pins.key,
            label="the QA models",
            payload=pins.load_payload(self.core.config.gpu.device),
            need_mb=pins.vram_need_mb,
        )

    def ready(self, host: RunnerHost, run: ActiveRun, need: GroupNeed) -> bool:
        """Make the group ready; False while waiting for free VRAM. After a Qwen load, the engine guard runs.

        A load the worker refuses is a job-level failure (the models as pinned cannot run), except running out
        of memory, which is retried (``failures``). A load the pool refuses because it holds another group
        on the GPU (``ResidencyError``: its bookkeeping and the residency's disagree) clears the GPU and is
        tried once more; a second refusal fails the job with ``INTERNAL``, never a take."""
        core = self.core
        try:
            state = self._ensure(host, run, need)
        except ResidencyError as exc:
            log.warning(
                "job %s: the pool refused to load %s (%s); clearing the GPU, once more", run.job_id, need.label, exc
            )
            self._clear(host)
            try:
                state = self._ensure(host, run, need)
            except ResidencyError as again:
                raise NarrationError(
                    codes.INTERNAL,
                    f"the worker pool refused to load {need.label} twice, with the GPU cleared in between",
                    details={"group": need.group, "exception": type(again).__name__},
                    retryable=False,
                    hint="This is a bug in the service; the daemon's log has the details. Nothing was rendered.",
                ) from again
        if state == "waiting":
            run.message = f"waiting for free GPU memory to load {need.label}"
            return False
        if state == "loaded" and need.group == "qwen":
            run.prepared = None
            client = host.workers.client("qwen", cublas_workspace_config=need.cublas_workspace_config)
            hello = client.hello
            core.phase(host, run, "canary")
            try:
                core.canary = core.parts.guard.after_load(host, run.profile, hello)
            except Exception:
                # A load the guard refused is never used: not by this job, and not by the next, which loads
                # again and is checked again.
                core.canary = "not_run"
                self.drop(host, "qwen")
                raise
        return True

    def _ensure(self, host: RunnerHost, run: ActiveRun, need: GroupNeed) -> Readiness:
        core = self.core
        try:
            return core.residency.ensure(host, need, phase=lambda p: core.phase(host, run, p))
        except WorkerFailure as exc:
            worker = worker_code(exc)
            if worker == codes.GPU_OOM:
                raise
            code = codes.BACKEND_NOT_INSTALLED if worker == codes.BACKEND_NOT_INSTALLED else codes.INTERNAL
            raise NarrationError(
                code,
                f"the {need.group} worker could not load {need.label}: {exc.message}",
                details={"worker_code": exc.code, **exc.details},
                retryable=False,
                hint="Run narration-admin doctor; the models as pinned could not be loaded.",
            ) from exc

    def _clear(self, host: RunnerHost) -> None:
        """Unload whatever group the pool says is on the GPU, and forget every group the residency knew."""
        holder = host.workers.gpu_holder
        if holder is not None:
            self.drop(host, holder)
        for group in GROUP_ROLES:
            self.core.residency.forget(group)

    def drop(self, host: RunnerHost, group: GpuHolder) -> None:
        """Unload the group through the pool, and forget it was loaded, even if the unload fails."""
        try:
            self.core.residency.unload(host, group)
        except (WorkerFailure, WorkerCrashed, WorkerTimeout) as exc:
            log.warning("unloading the %s group failed: %s", group, exc)
        finally:
            self.core.residency.forget(group)

    # ------------------------------------------------------------------ render
    def render(self, host: RunnerHost, run: JobRun, attempt: Attempt) -> Outcome:
        """Make the attempt's render (section 10), or take it from the cache or from another holder."""
        core, store, seg = self.core, host.store, run.segments[attempt.segment]
        found = store.get_render(attempt.render_key)
        if found is None:
            # The key is claimed before the model group is loaded: work in flight elsewhere never costs a swap.
            status, lease = store.claim(attempt.render_key, host.holder, ttl_s=RENDER_LEASE_S)
            if status == "in_flight":
                attempt.deferred_until = core.parts.clock() + core.parts.defer_s
                return "worked"
            if status == "claimed" and lease is not None:
                try:
                    with core.leases.kept(lease, ttl_s=RENDER_LEASE_S, every_s=LEASE_RENEW_S):
                        need = self.qwen_need(run)
                        if not self.ready(host, run, need):
                            return "waited" if not host.should_stop() else "stopped"
                        found = self._synthesize(host, run, seg, attempt, need)
                finally:
                    lease.release()
            else:
                found = store.get_render(attempt.render_key)
                if found is None:
                    return "worked"  # published and gone again (collected): looked up afresh next step
        attempt.render = found
        self.find_take(host, run, seg, attempt)  # a render made before may already have its take
        return "worked"

    def _synthesize(
        self, host: RunnerHost, run: JobRun, seg: SegmentWork, attempt: Attempt, need: GroupNeed
    ) -> RenderRecord:
        core = self.core
        config = core.config
        client = host.workers.client("qwen", cublas_workspace_config=need.cublas_workspace_config)
        if run.prepared is not client:  # a new worker holds no prepared voice, whatever its pid
            self._prepare(run, client)
        core.phase(host, run, "rendering" if attempt.round == 0 else "retaking")
        run.message = f"round {attempt.round}: rendering {label(run, attempt)}"
        engine_text = seg.text.engine_text
        cap = call_cap(run.profile, engine_text)
        render_id = keys.render_id(attempt.render_key)
        out = run.scratch / f"{render_id}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        started = core.parts.clock()
        reply = client.request(
            "synthesize",
            {
                "voice_hash": run.voice_hash,
                "engine_text": engine_text,
                "language": names.LANGUAGE,
                "seed": attempt.seed,
                "max_new_tokens": cap,
                "out_path": str(out),
            },
            timeout_s=max(300.0, cap / FRAMES_PER_SECOND * 5.0),
        )
        samples, rate, gen_s = int(reply["samples"]), int(reply["sample_rate"]), float(reply["gen_s"])
        audio_s = samples / rate if rate else 0.0
        hello = client.hello
        record = RenderRecord(
            render_id=render_id,
            render_key=attempt.render_key,
            voice=RenderVoice(
                voice_hash=run.voice_hash,
                clip_sha256=run.request.voice.sha256,
                x_vector_only_mode=config.engines.qwen3_base.x_vector_only_mode,
            ),
            engine=RenderEngine(
                engine_profile_id=run.profile.engine_profile_id,
                engine_profile_hash=run.profile.hash,
                model_repo=run.profile.model_repo,
                model_revision=run.profile.model_revision,
                non_streaming_mode=bool(run.profile.settings["non_streaming_mode"]),
                generation={**generation(run.profile), "max_new_tokens": int(reply.get("max_new_tokens", cap))},
                observed=_observed(hello),
            ),
            engine_text=engine_text,
            seed=attempt.seed,
            attempt=attempt.attempt,
            raw=RawAudio(path=str(out), sha256="", sample_rate=rate, samples=samples),
            hit_token_cap=bool(reply["hit_token_cap"]),
            gen_s=gen_s,
            rtf=round(gen_s / audio_s, 4) if audio_s else 0.0,
            canary=CanaryRecord(batch_status=core.canary),
            licence=Licence(generation_model=run.profile.licence, voice_clip="synthetic"),
        )
        published = host.store.put_render(record, out)
        run.fresh_keys.add(attempt.render_key)
        core.throughput.record(core.parts.clock() - started, seg.est_s / 3)
        return published

    def _prepare(self, run: JobRun, client: WorkerClient) -> None:
        """Prepare the job's voice in the Qwen worker (App. A ``prepare_voice``).

        A clip the worker cannot prepare fails the job (``UNSUPPORTED_AUDIO``, section 14): every take needs
        the voice, so no take is tried. Running out of memory, a worker that is not installed, and one that
        lost its models go on to the failure handling like any worker failure (``failures``)."""
        try:
            client.request(
                "prepare_voice",
                {
                    "voice_hash": run.voice_hash,
                    "ref_wav": str(run.clip),
                    "ref_text": run.request.voice.transcript,
                    "x_vector_only_mode": self.core.config.engines.qwen3_base.x_vector_only_mode,
                },
                timeout_s=PREPARE_TIMEOUT_S,
            )
        except WorkerFailure as exc:
            code = worker_code(exc)
            if code in (codes.GPU_OOM, codes.BACKEND_NOT_INSTALLED) or code in LOST_STATE:
                raise
            raise NarrationError(
                codes.UNSUPPORTED_AUDIO,
                f"the Qwen worker could not prepare the voice from its clip: {exc.message}",
                field="voice",
                details={"worker_code": code, **exc.details},
                retryable=False,
                hint="Send a WAV the service designed (at most 30 s and 20 MB); nothing was rendered.",
            ) from exc
        run.prepared = client

    # ------------------------------------------------------------------ post-process
    def post(self, host: RunnerHost, run: JobRun, attempt: Attempt) -> Outcome:
        """Make the attempt's take from its render (section 10.4), or take it from the cache or another holder."""
        core, store, seg, render = self.core, host.store, run.segments[attempt.segment], attempt.render
        assert render is not None
        delivery_key = self.delivery_key(render)
        found = store.get_take(delivery_key)
        if found is None:
            status, lease = store.claim(delivery_key, host.holder, ttl_s=POST_LEASE_S)
            if status == "in_flight":
                attempt.deferred_until = core.parts.clock() + core.parts.defer_s
                return "worked"
            if status == "claimed" and lease is not None:
                try:
                    with core.leases.kept(lease, ttl_s=POST_LEASE_S, every_s=LEASE_RENEW_S):
                        found = self._deliver(host, run, seg, attempt, render, delivery_key)
                finally:
                    lease.release()
            else:
                found = store.get_take(delivery_key)
                if found is None:
                    return "worked"
        attempt.take = found
        self.find_analysis(host, run, seg, attempt)
        return "worked"

    def _deliver(
        self, host: RunnerHost, run: JobRun, seg: SegmentWork, attempt: Attempt, render: RenderRecord, delivery_key: str
    ) -> TakeRecord:
        core = self.core
        core.phase(host, run, "postprocessing")
        run.message = f"round {attempt.round}: post-processing {label(run, attempt)}"
        take_id = keys.take_id(delivery_key)
        out = run.scratch / f"{take_id}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        started = core.parts.clock()
        made = core.parts.delivery.process(Path(render.raw.path), out, core.config.delivery)
        record = TakeRecord(
            take_id=take_id,
            delivery_key=delivery_key,
            render_id=render.render_id,
            delivery=DeliveryAudio(
                path=str(out), sha256="", sample_rate=made.sample_rate, samples=made.samples, duration_s=made.duration_s
            ),
            trim=made.trim,
            loudness=made.loudness,
            tools=core.parts.delivery.tools,
            flags=made.flags,
        )
        published = host.store.put_take(record, out)
        core.throughput.record(core.parts.clock() - started, seg.est_s / 3)
        return published

    # ------------------------------------------------------------------ score
    def score(self, host: RunnerHost, run: JobRun, attempt: Attempt) -> Outcome:
        """Score the attempt's take (section 11.1), or take its analysis from the cache or another holder."""
        core, store, seg = self.core, host.store, run.segments[attempt.segment]
        take, render, key = attempt.take, attempt.render, attempt.analysis_key
        assert take is not None and render is not None and key is not None
        found = store.get_analysis(key)
        if found is None:
            status, lease = store.claim(key, host.holder, ttl_s=SCORE_LEASE_S)  # claimed before QA is loaded
            if status == "in_flight":
                attempt.deferred_until = core.parts.clock() + core.parts.defer_s
                return "worked"
            if status == "claimed" and lease is not None:
                try:
                    with core.leases.kept(lease, ttl_s=SCORE_LEASE_S, every_s=LEASE_RENEW_S):
                        if not self.ready(host, run, self.qa_need()):
                            return "waited" if not host.should_stop() else "stopped"
                        core.phase(host, run, "scoring")
                        run.message = f"round {attempt.round}: scoring {label(run, attempt)}"
                        started = core.parts.clock()
                        found = store.put_analysis(self._analyse(host, run, seg, take, render, key))
                        core.throughput.record(core.parts.clock() - started, seg.est_s / 3)
                finally:
                    lease.release()
            else:
                found = store.get_analysis(key)
                if found is None:
                    return "worked"
        attempt.analysis = found
        return "worked"

    def _analyse(
        self, host: RunnerHost, run: JobRun, seg: SegmentWork, take: TakeRecord, render: RenderRecord, key: str
    ) -> AnalysisRecord:
        """Section 11.1 on the delivery file: ASR, the speaker embedding, the cue alignment, the signal facts,
        then the verdict. The worker returns raw outputs only; every verdict is the scorer's (plan.md P1)."""
        parts = self.core.parts
        client = host.workers.client("qa")
        wav = take.delivery.path
        heard = client.request(
            "transcribe",
            {"wav": wav, "language": names.LANGUAGE, "word_timestamps": True, "long_form": True},
            timeout_s=QA_TIMEOUT_S,
        )
        # App. A's embed takes "cuda" or "cpu" (EmbedRequest.device). "cuda" means the GPU the QA group was
        # loaded on, which the load's payload names ([gpu] device, e.g. "cuda:1"); the worker embeds there. No
        # device index goes in the request, and the protocol is unchanged (the lead's decision, WP31).
        device = "cpu" if self.core.config.gpu.device == "cpu" else "cuda"
        embedded = client.request("embed", {"wav": wav, "device": device}, timeout_s=QA_TIMEOUT_S)
        aligner = parts.aligner
        transcript = aligner.build_transcript(seg.text, seg.hints)
        reply: AlignReply | None = None
        error: dict[str, Any] | None = None
        if has_letters(transcript):  # no letter: nothing to place, and resolve says so (DC-12)
            try:
                reply = cast(
                    AlignReply,
                    client.request("align", {"wav": wav, "tokens": list(transcript.tokens)}, timeout_s=QA_TIMEOUT_S),
                )
            except WorkerFailure as exc:
                if worker_code(exc) != codes.ALIGNMENT_ERROR:
                    raise
                error = dict(exc.details or {})
                log.info("job %s: the aligner could not align %s: %s", run.job_id, take.take_id, exc.message)
        audio, rate = soundfile.read(wav, dtype="float32", always_2d=True)
        samples = np.asarray(audio, dtype=np.float32).mean(axis=1).astype(np.float32)
        asr_words = tuple(cast(list[AsrWord], heard.get("words") or []))
        alignment = aligner.resolve(transcript, reply, samples, int(rate), asr_words, run.measured_error, error=error)
        signal = parts.delivery.signal_stats(Path(render.raw.path), Path(wav))
        embedding = tuple(float(v) for v in embedded["embedding"])
        facts = self.scoring_facts(run, seg)
        qa = parts.scorer.score(
            QaInputs(
                segment=seg.text,
                hints=seg.hints,
                voice_transcript=run.request.voice.transcript,
                asr_text=str(heard.get("text") or ""),
                asr_words=asr_words,
                embedding=embedding,
                alignment=alignment,
                signal=signal,
                hit_token_cap=render.hit_token_cap,
                measurement=facts.measurement,
                anchor=facts.anchor,
                similarity=facts.similarity,
                pace=facts.pace,
            )
        )
        pins = parts.qa_pins
        return AnalysisRecord(
            analysis_id=keys.analysis_id(key),
            analysis_key=key,
            take_id=take.take_id,
            text=AnalysisText(
                cues=tuple(_request_free(c) for c in seg.text.cues),
                text_checks=seg.text.text_checks or parts.text.checks_info,
                hints_used=seg.hints,
            ),
            versions=AnalysisVersions(
                qa_profile=parts.scorer.profile_version,
                asr=pins.asr.name,
                sv=pins.sv.name,
                aligner_method=aligner.method_id,
                number_reader=parts.scorer.number_reader,
                measurement=self.measurement_key(run, seg),
            ),
            alignment=alignment,
            qa=qa,
            licence=pins.licence,
            embedding=embedding,
        )


def _request_free(cue: CueText) -> CueText:
    """A cue's text echo as a cached analysis keeps it: nothing the analysis key does not cover. ``received``
    becomes the spoken form, and warnings carry no segment id. The job assembler echoes the request's own
    text (``get_results``), never a cached analysis's."""
    return dataclasses.replace(
        cue,
        received=cue.spoken,
        warnings=tuple(dataclasses.replace(w, segment_id=None) for w in cue.warnings),
    )


def _observed(hello: HelloReply | None) -> dict[str, Any]:
    """What the worker observed of its machine, recorded with the render and never hashed."""
    if hello is None:
        return {}
    fingerprint = cast(Mapping[str, Any], hello.get("fingerprint") or {})
    return {k: fingerprint[k] for k in ("gpu", "driver", "cuda", "cudnn") if fingerprint.get(k) is not None}


__all__ = [
    "FRAMES_PER_SECOND",
    "LEASE_RENEW_S",
    "POST_LEASE_S",
    "PREPARE_TIMEOUT_S",
    "QA_TIMEOUT_S",
    "RENDER_LEASE_S",
    "SCORE_LEASE_S",
    "ScoringFacts",
    "Stages",
]
