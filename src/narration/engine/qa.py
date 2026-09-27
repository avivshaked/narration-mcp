"""The QA group as this installation pins it, and the cue aligner (design sections 4, 10.2, 11 and 18).

The analysis key names the ASR and speaker models by repo and revision, and the aligner by its method id
(section 10.2), so these come from the service's pins (``models``), never from a request:

- **Whisper-large-v3** (``asr``) and **WavLM-base-plus-sv** (``sv``) on the GPU;
- the **CTC aligner** (``[alignment] model``, the wav2vec2 model by default) on the CPU, which WP15's
  ``CtcAligner`` drives with the thresholds of ``[alignment]``.

``QA_VRAM_NEED_MB`` is the VRAM the group needs (section 4 item 2), and ``QA_MEASURED`` the configuration it
was measured with; see their docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from narration.align import CtcAligner
from narration.config import Config
from narration.contracts.models import Licence
from narration.jobs.pins import ModelPin, QaPins

from .models import WAVLM_SV, WHISPER, PinnedModel, pinned
from .profile import EngineSetupError

QA_VRAM_NEED_MB: Final = 11_500
"""The VRAM the QA group needs on the GPU, in MB (section 4 item 2: a load waits for this plus the margin).

KNOW (WP22's spike h, ``spikes/h-i-qa-load/README.md`` and ``results.json``; the lead's decision): it covers the
whole group, the CUDA context included, for a take of any length, as ``QA_MEASURED`` describes it: Whisper in
float16 with five beams and word times, and WavLM-SV embedding in 60 s windows (DC-15), with the aligner on the
CPU. The worker returns the allocator's cached memory after each op, so the ops' footprints do not add up; the
peak is transcription's, about 10.9 GB with the context, and 11 500 adds about 5 % for the allocator. Both
peaks are bounded (Whisper's 30 s window, the embedding's 60 s windows), so one number holds for any take.

It is not the ``load`` reply's ``vram_mb`` (3606 in the spike): that is the resident models alone, before any
op runs. Re-run spike h and change this with ``QA_MEASURED`` when any part of that configuration changes."""


@dataclass(frozen=True, slots=True, kw_only=True)
class QaMeasured:
    """The QA group's configuration as spike h measured ``QA_VRAM_NEED_MB`` with it. The models are written out
    here, not taken from ``models``, so that a new pin there fails ``tests/engine/test_qa_need.py`` until the
    spike is run again."""

    asr_repo: str
    asr_revision: str
    asr_dtype: str
    """Whisper's dtype on the GPU (float32 on the CPU)."""
    asr_attn_implementation: str
    asr_num_beams: int
    word_timestamps: bool
    sv_repo: str
    sv_revision: str
    sv_dtype: str
    sv_window_s: int
    """WavLM-SV embeds a take longer than this in windows of this length (DC-15 as amended)."""
    aligner_device: str


QA_MEASURED: Final = QaMeasured(
    asr_repo="openai/whisper-large-v3",
    asr_revision="06f233fe06e710322aca913c1bc4249a0d71fce1",
    asr_dtype="float16",
    asr_attn_implementation="eager",
    asr_num_beams=5,
    word_timestamps=True,
    sv_repo="microsoft/wavlm-base-plus-sv",
    sv_revision="feb593a6c23c1cc3d9510425c29b0a14d2b07b1e",
    sv_dtype="float32",
    sv_window_s=60,
    aligner_device="cpu",
)
"""What spike h ran (``spikes/h-i-qa-load``, 2026-09-27): see ``QA_VRAM_NEED_MB``."""


def model_pin(config: Config, model: PinnedModel) -> ModelPin:
    """A pinned model as the job engine loads it: its snapshot folder under ``[server] models_root``, which must
    exist (``BACKEND_NOT_INSTALLED`` otherwise)."""
    folder = model.snapshot_dir(config.server.models_root)
    if not folder.is_dir():
        raise EngineSetupError(
            f"no snapshot of {model.repo} at revision {model.revision} under the models root",
            hint="Ask the operator to run narration-admin install; nothing was rendered.",
            details={"repo": model.repo, "revision": model.revision, "snapshot_dir": str(folder)},
        )
    return ModelPin(repo=model.repo, revision=model.revision, snapshot_dir=str(folder))


def aligner_model(config: Config) -> PinnedModel:
    """The pin of ``[alignment] model``; a model the service does not pin is ``BACKEND_NOT_INSTALLED``."""
    try:
        return pinned(config.alignment.model)
    except KeyError as exc:
        raise EngineSetupError(
            str(exc.args[0]),
            hint="Set [alignment] model to the service's aligner (section 16), then run narration-admin install.",
        ) from exc


def qa_pins(config: Config) -> QaPins:
    """The QA group's pinned models, the VRAM it needs and the licences every analysis records (section 18)."""
    aligner = aligner_model(config)
    return QaPins(
        asr=model_pin(config, WHISPER),
        sv=model_pin(config, WAVLM_SV),
        aligner=model_pin(config, aligner),
        vram_need_mb=QA_VRAM_NEED_MB,
        licence=Licence(aligner=aligner.licence, asr=WHISPER.licence, sv=WAVLM_SV.licence),
    )


def aligner(config: Config) -> CtcAligner:
    """The cue aligner ``[alignment]`` configures, at the pinned revision of its model (WP15)."""
    model = aligner_model(config)
    try:
        return CtcAligner.from_config(config.alignment, revision=model.revision)
    except ValueError as exc:
        raise EngineSetupError(
            f"the aligner cannot run as configured: {exc}",
            hint="Set [alignment] as design section 16 shows; nothing was rendered.",
        ) from exc


def aligner_method_id(config: Config) -> str:
    """The configured aligner's method id, which the store needs to find its alignment benchmark
    (``NarrationStore(alignment_method_id=...)``) and every analysis key names."""
    return aligner(config).method_id


__all__ = [
    "QA_MEASURED",
    "QA_VRAM_NEED_MB",
    "QaMeasured",
    "aligner",
    "aligner_method_id",
    "aligner_model",
    "model_pin",
    "qa_pins",
]
