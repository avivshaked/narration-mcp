"""The QA group as this installation pins it, and the cue aligner (design sections 4, 10.2, 11 and 18).

The analysis key names the ASR and speaker models by repo and revision, and the aligner by its method id
(section 10.2), so these come from the service's pins (``models``), never from a request:

- **Whisper-large-v3** (``asr``) and **WavLM-base-plus-sv** (``sv``) on the GPU;
- the **CTC aligner** (``[alignment] model``, the wav2vec2 model by default) on the CPU, which WP15's
  ``CtcAligner`` drives with the thresholds of ``[alignment]``.

``QA_VRAM_MB`` is the VRAM the group needs (section 4 item 2): WP22 measured transcription with word times
peaking at 10.6 GB reserved, flat from 30 s, and the embedding of longer takes is windowed (DC-15), so its
peak stays under that for a take of any length. WP22 proposed 11500 and the lead set it (KNOW:
``status/WP22.md`` and ``spikes/h-i-qa-load`` on ``wp/22-qa-worker``).
"""

from __future__ import annotations

from typing import Final

from narration.align import CtcAligner
from narration.config import Config
from narration.contracts.models import Licence
from narration.jobs.pins import ModelPin, QaPins

from .models import WAVLM_SV, WHISPER, PinnedModel, pinned
from .profile import EngineSetupError

QA_VRAM_MB: Final = 11500


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
        vram_need_mb=QA_VRAM_MB,
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


__all__ = ["QA_VRAM_MB", "aligner", "aligner_method_id", "aligner_model", "model_pin", "qa_pins"]
