"""The ``qa`` worker role: the protocol handler over Whisper, WavLM-SV, the CTC aligner and the voice measures
(design section 4, QA worker; sections 3.6, 10.1, 11.1 and 11.2; Appendix A; plan.md WP22).

Ops: ``hello`` (from ``WorkerHandler``), ``load``, ``unload``, ``shutdown``, ``transcribe``, ``embed``, ``f0``,
``align`` and ``profile``. The worker is a model runner (plan.md P1): it returns transcripts with word times,
embeddings, pitch tracks, token spans and profile numbers, and never a verdict.

- ``load`` takes ``device`` and ``models`` {asr?, sv?, aligner?}, each a ``ModelRef`` {repo, revision,
  snapshot_dir}; ``determinism`` is optional. The references are checked as the ``qwen3`` worker and the fake
  check theirs (``snapshots``): well formed with an absolute ``snapshot_dir``, then every directory present
  (``BACKEND_NOT_INSTALLED``) before anything else, then named by its 40-hex revision; then ``device``; then that
  each snapshot holds the model its use pins (``config.json``; else ``BACKEND_NOT_INSTALLED``). Only a ``load``
  that passes every check releases the models loaded before, applies the determinism switches (the request's,
  or section 10.1's: TF32 off, cuDNN deterministic, benchmark off, deterministic algorithms warn-only) and
  loads: Whisper and WavLM on ``device``, the aligner always on the CPU (section 11.2). ``models: {}`` loads no
  model (``f0`` and ``profile`` then run). A load that fails part-way leaves nothing loaded. ``vram_mb`` is the
  CUDA memory torch has reserved after the load, and null when no model went to a GPU.
- ``transcribe`` (``asr``): Whisper-large-v3 as ``asr`` pins it. ``language`` is a name or code (``"English"``).
- ``embed`` (``sv``): an L2-normalised x-vector, on ``cuda`` (the GPU the model was loaded on) or ``cpu`` (a CPU
  copy made on first use, so the canary check never swaps models; section 10.1). Audio of up to 30 s is embedded
  in one pass; longer audio in equal windows of at most 30 s, whose embeddings' mean is normalised again (DC-15).
- ``f0``: a ``librosa.pyin`` track between ``fmin_hz`` and ``fmax_hz``, every 10 ms at 16 kHz.
- ``align`` (``aligner``): WP15's CTC alignment (``align.AlignOp``).
- ``profile``: the voice profile's numbers (``models.ProfileMeasurements``) and two PNG pictures written to
  ``out_dir`` through temporary names (``voice``).

``f0`` and ``profile`` use no model, but like every model op they reply ``NOT_LOADED`` until a ``load`` succeeds
and after ``unload`` (the worker contract; they need only that some load succeeded). An op whose model the last
``load`` did not name replies ``NOT_LOADED`` too, saying which model to name.

The worker forces ``HF_HUB_OFFLINE`` and ``TRANSFORMERS_OFFLINE`` on before anything imports the Hugging Face
libraries (section 17.7), and points matplotlib's and numba's caches into the store
(``<store_root>/scratch/qa-worker/``), so nothing is written outside it. It writes no audio: its outputs are the
JSON replies and the profile's PNG files.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Callable
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from narration_worker.determinism import apply_determinism, parse_determinism
from narration_worker.errors import OpError
from narration_worker.handler import (
    DEVICE_PATTERN,
    Request,
    WorkerContext,
    WorkerHandler,
    require_bool,
    require_number,
    require_one_of,
    require_str,
)
from narration_worker.protocol import DeterminismSettings

from . import voice
from .align import AlignOp, QaError, check_ctc_config, op_error, read_audio, snapshot_vocabulary, to_mono_16k
from .asr import ARCHITECTURE as ASR_ARCHITECTURE
from .asr import MODEL_TYPE as ASR_MODEL_TYPE
from .asr import WhisperAsr, whisper_language
from .snapshots import Snapshot, check_architecture, check_use, parse_models
from .sv import ARCHITECTURE as SV_ARCHITECTURE
from .sv import DEVICES as EMBED_DEVICES
from .sv import MODEL_TYPE as SV_MODEL_TYPE
from .sv import WavLmSv

OFFLINE_ENV: Final = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
CACHE_DIR: Final = ("scratch", "qa-worker")
"""Where, under the store root, the worker's libraries keep their caches (matplotlib's font list, numba's
compiled functions)."""
QA_DETERMINISM: Final[DeterminismSettings] = {
    "tf32": False,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "deterministic_algorithms": "warn_only",
}
"""The switches a ``load`` without ``determinism`` applies: section 10.1's."""


class QaHandler(WorkerHandler):
    """The ``qa`` role (see the module docstring)."""

    role = "qa"
    fingerprint_packages = (
        "narration-worker-qa",
        "transformers",
        "tokenizers",
        "safetensors",
        "torch",
        "torchaudio",
        "numpy",
        "scipy",
        "soundfile",
        "librosa",
        "soxr",
        "numba",
        "pyloudnorm",
        "matplotlib",
    )

    def __init__(self, context: WorkerContext) -> None:
        super().__init__(context)
        os.environ.update(OFFLINE_ENV)
        cache = context.store_root.joinpath(*CACHE_DIR)
        os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
        os.environ["NUMBA_CACHE_DIR"] = str(cache / "numba")
        self.aligner = AlignOp(self)
        self._asr: WhisperAsr | None = None
        self._sv: WavLmSv | None = None
        self._loaded = False

    # ------------------------------------------------------------------ load / unload / shutdown
    def op_load(self, request: Request) -> dict[str, Any]:
        """Load the models ``models`` names from their pinned snapshots (see the module docstring)."""
        snapshots = _qa(parse_models, request)
        for snapshot in snapshots:
            _qa(check_use, snapshot)
        device = require_str(request, "device")
        if DEVICE_PATTERN.fullmatch(device) is None:
            raise OpError("INVALID_REQUEST", "device must be cpu, cuda or cuda:<n>", {"field": "device"})
        determinism = parse_determinism(request["determinism"]) if "determinism" in request else QA_DETERMINISM
        by_use = {snapshot.use: snapshot for snapshot in snapshots}
        for snapshot in snapshots:
            _qa(_check_model, snapshot)
        torch = self.torch()
        if torch is None:
            raise OpError(
                "BACKEND_NOT_INSTALLED",
                "torch is not installed in the QA worker's environment: sync workers/qa",
                {"package": "torch"},
            )
        if device.startswith("cuda") and ({"asr", "sv"} & set(by_use)) and not _cuda_usable(torch):
            raise OpError(
                "BACKEND_NOT_INSTALLED",
                "no CUDA device is available to this worker: load with device cpu, or run it where a GPU is",
                {"field": "device", "device": device},
            )
        self.shutdown()  # every check passed: from here the load goes ahead
        apply_determinism(determinism, torch)
        load_s = 0.0
        try:
            if "asr" in by_use:
                ref = by_use["asr"]
                asr = WhisperAsr(torch)
                load_s += self._run(lambda: asr.load(ref.repo, ref.revision, ref.path, device))
                self._asr = asr
            if "sv" in by_use:
                ref = by_use["sv"]
                sv = WavLmSv(torch)
                load_s += self._run(lambda: sv.load(ref.repo, ref.revision, ref.path, device))
                self._sv = sv
            if "aligner" in by_use:
                aligner_ref = request["models"]["aligner"]
                load_s += self._run(lambda: self.aligner.load(aligner_ref, "models.aligner"))
        except BaseException:
            self.shutdown()
            raise
        self._loaded = True
        on_gpu = device.startswith("cuda") and bool({"asr", "sv"} & set(by_use))
        return {"load_s": round(load_s, 3), "vram_mb": _reserved_mb(torch, device) if on_gpu else None}

    def op_unload(self, request: Request) -> dict[str, Any]:
        """Release every model (idempotent)."""
        self.shutdown()
        return {}

    def shutdown(self) -> None:
        self._loaded = False
        self.aligner.unload()
        for model in (self._asr, self._sv):
            if model is not None:
                model.unload()
        self._asr = None
        self._sv = None
        gc.collect()
        torch = self._torch
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and cuda.is_available() and cuda.is_initialized():
            cuda.empty_cache()

    # ------------------------------------------------------------------ the model ops
    def op_transcribe(self, request: Request) -> dict[str, Any]:
        """Transcribe a WAV with Whisper as pinned (``asr``): ``protocol.TranscribeReply``."""
        asr = self._need(self._asr, "transcribe", "asr")
        language = require_str(request, "language")
        _qa(whisper_language, language)
        word_timestamps = require_bool(request, "word_timestamps")
        long_form = require_bool(request, "long_form")
        audio = self._audio_16k(request)
        return self._run(lambda: asr.transcribe(audio, language, word_timestamps=word_timestamps, long_form=long_form))

    def op_embed(self, request: Request) -> dict[str, Any]:
        """The WavLM-SV x-vector of a WAV, L2-normalised: ``protocol.EmbedReply``.

        ``device`` ``cuda`` runs on the GPU the model was loaded on, ``cpu`` on a CPU copy (the canary's path). Audio
        of up to 30 s is embedded in one pass, exactly as the bake-off's evidence was; longer audio is cut into
        ``ceil(length / 30 s)`` equal windows, and the reply is the mean of their L2-normalised embeddings,
        normalised again (DC-15), which bounds the memory at a 30 s pass's. KNOW (``spikes/h-i-qa-load``): its
        cosine to a one-pass embedding of the same audio is 0.999 at 60 s and 0.994 at 119 s.
        """
        sv = self._need(self._sv, "embed", "sv")
        device = require_one_of(request, "device", EMBED_DEVICES)
        audio = self._audio_16k(request)
        return self._run(lambda: sv.embed(audio, device))

    def op_f0(self, request: Request) -> dict[str, Any]:
        """A pyin pitch track of a WAV: ``protocol.F0Reply`` (null Hz where a frame is not voiced)."""
        self._need_loaded("f0")
        fmin = require_number(request, "fmin_hz")
        fmax = require_number(request, "fmax_hz")
        if not voice.F0_MIN_HZ <= fmin < voice.F0_MAX_HZ:
            raise OpError(
                "INVALID_REQUEST",
                f"fmin_hz must be from {voice.F0_MIN_HZ:g} and below {voice.F0_MAX_HZ:g}",
                {"field": "fmin_hz"},
            )
        if not fmin < fmax <= voice.F0_MAX_HZ:
            raise OpError(
                "INVALID_REQUEST",
                f"fmax_hz must be above fmin_hz and at most {voice.F0_MAX_HZ:g}",
                {"field": "fmax_hz"},
            )
        audio = self._audio_16k(request)
        track = self._run(lambda: voice.pitch_track(audio, fmin, fmax))
        f0, probability = voice.track_reply(track)
        return {
            "hop_s": track.hop_s,
            "f0_hz": f0,
            "voiced_probability": probability,
            "method": voice.pyin_method(fmin, fmax),
        }

    def op_align(self, request: Request) -> dict[str, Any]:
        """CTC forced alignment of the server's tokens (WP15): ``protocol.AlignReply``."""
        return self._run(lambda: self.aligner.handle(request))

    def op_profile(self, request: Request) -> dict[str, Any]:
        """The voice profile of a WAV: ``protocol.ProfileReply``, with the pictures written to ``out_dir``."""
        self._need_loaded("profile")
        transcript = request.get("transcript")
        if transcript is not None and not isinstance(transcript, str):
            raise OpError("INVALID_REQUEST", "transcript must be a string or null", {"field": "transcript"})
        path = self.input_file(request, "wav")
        audio, rate = _qa(read_audio, path)
        out_dir = self.output_dir(request, "out_dir")
        measurements, track, x = self._run(lambda: voice.measure(audio, rate, transcript))
        pictures = self._run(lambda: voice.draw_pictures(x, track, measurements, out_dir))
        return {"measurements": measurements, "pictures": pictures, "method": voice.method()}

    # ------------------------------------------------------------------ internals
    def _need_loaded(self, op: str) -> None:
        if not self._loaded:
            raise OpError("NOT_LOADED", f"{op} needs the QA worker loaded: send load first")

    def _need[T](self, model: T | None, op: str, use: str) -> T:
        self._need_loaded(op)
        if model is None:
            raise OpError(
                "NOT_LOADED",
                f"{op} needs models.{use}, which the last load did not name: send load with models.{use}",
                {"model": use},
            )
        return model

    def _audio_16k(self, request: Request) -> npt.NDArray[np.float32]:
        """The request's ``wav`` (inside the store), mixed to mono and resampled to 16 kHz."""
        path = self.input_file(request, "wav")
        audio, rate = _qa(read_audio, path)
        return to_mono_16k(audio, rate)

    def _run[T](self, call: Callable[[], T]) -> T:
        """Run a model call: a ``QaError`` keeps its code, running out of GPU memory is ``GPU_OOM``, and any other
        exception propagates (the loop reports it as ``INTERNAL``)."""
        try:
            return call()
        except QaError as exc:
            raise op_error(exc) from exc
        except OpError:
            raise
        except Exception as exc:
            classified = self.classify(exc)
            if classified is not None:
                raise classified from exc
            raise


def _check_model(snapshot: Snapshot) -> None:
    """Check that a snapshot holds the model its use pins, before anything loads (``BackendMissing``)."""
    if snapshot.use == "asr":
        check_architecture(snapshot.path, ASR_ARCHITECTURE, ASR_MODEL_TYPE, "ASR")
    elif snapshot.use == "sv":
        check_architecture(snapshot.path, SV_ARCHITECTURE, SV_MODEL_TYPE, "speaker-verification")
    else:
        path, vocab = snapshot_vocabulary(snapshot.revision, snapshot.path)
        check_ctc_config(path, vocab)


def _qa[T](call: Callable[..., T], *args: Any) -> T:
    """``call(*args)``, with a ``QaError`` turned into the protocol's ``OpError``."""
    try:
        return call(*args)
    except QaError as exc:
        raise op_error(exc) from exc


def _cuda_usable(torch: Any) -> bool:
    """Whether a CUDA device is visible. KNOW (torch 2.11 on Windows): with ``CUDA_VISIBLE_DEVICES`` empty,
    ``torch.cuda.is_available()`` is true while ``device_count()`` is 0, so both are asked."""
    return bool(torch.cuda.is_available()) and int(torch.cuda.device_count()) > 0


def _reserved_mb(torch: Any, device: str) -> int:
    """The CUDA memory torch has reserved on ``device``, in MiB (asked only after a model loaded there, so it
    never starts CUDA by itself)."""
    return int(torch.cuda.memory_reserved(device) // (1024 * 1024))
