"""The ``qwen3`` worker role: the protocol handler over ``engine.QwenEngine`` (design Appendix A, section 10.1).

Ops: ``hello`` (from ``WorkerHandler``), ``load``, ``prepare_voice``, ``synthesize``, ``design``, ``unload``,
``shutdown``. One model is resident at a time: Base for ``prepare_voice`` and ``synthesize``, VoiceDesign for
``design`` (section 4, GPU scheduler).

- ``load`` takes ``model`` {repo, revision, snapshot_dir}, ``device``, ``dtype``, ``attn_implementation``,
  ``determinism`` and ``settings``, all required. The snapshot folder must exist (else
  ``BACKEND_NOT_INSTALLED``, checked first), be an absolute path, be named by its 40-hex revision (section 4)
  and hold Qwen3-TTS Base or VoiceDesign (``engine.check_snapshot``). Only a ``load`` that passes every check
  applies the determinism switches, before the model loads, so a refused ``load`` changes nothing. Every
  audio-changing setting comes from ``settings``, none from the library's defaults, and
  ``settings.generation.max_new_tokens`` is the ceiling of the calls' own caps. The checks on ``dtype``,
  ``attn_implementation`` and ``settings`` are ``narration_worker.qwen_settings``, which the fake worker
  shares, so it refuses the same loads. Every ``load`` clears the prepared voices.
- ``prepare_voice`` keeps at most ``engine.MAX_PREPARED_VOICES`` voices, evicting the least recently used,
  so ``VOICE_NOT_PREPARED`` from ``synthesize`` means "send ``prepare_voice`` again, then retry".
- ``synthesize`` and ``design`` take a required ``max_new_tokens``, the call's own generation cap (design
  section 10.1, DC-4), checked with the protocol's ``require_max_new_tokens``: an integer from 2 (qwen-tts's
  ``min_new_tokens``) to the loaded ceiling. A missing or out-of-range cap is ``INVALID_REQUEST``, never
  clamped, and like every argument error it comes before ``VOICE_NOT_PREPARED``. They seed ``random``,
  numpy, torch and torch.cuda with the request's seed immediately before generating (section 10.3), and
  write the raw render as a float32 mono WAV to ``out_path`` through a temp name. The reply echoes the cap
  applied (``max_new_tokens``).
  - The file is byte-reproducible: the same samples give the same bytes (``narration_worker.wav``, the
    writer every worker shares), so a render's sha256 is a fact about its audio.
  - The file keeps the samples as generated, NaN and infinity included: ``SIGNAL_INVALID`` is the
    server's verdict on the raw take (section 11.1, DC-5).
  - The reply is ``protocol.AudioReply``: ``new_tokens`` (codec frames decoded: the talker's steps less
    one, so at most ``max_new_tokens`` less one) and ``hit_token_cap`` (the last sampled token was not the
    end token). Whether a take that hit the cap fails is the server's decision (``TOKEN_CAP_HIT``, section
    11.1).

The worker forces ``HF_HUB_OFFLINE`` and ``TRANSFORMERS_OFFLINE`` on before anything imports the Hugging Face
libraries (section 17.7), whatever its environment says.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, TypeVar

import numpy as np
from narration_worker.determinism import apply_determinism, parse_determinism, seed_everything
from narration_worker.errors import OpError
from narration_worker.handler import (
    DEVICE_PATTERN,
    REVISION_PATTERN,
    Request,
    WorkerContext,
    WorkerHandler,
    require_bool,
    require_int,
    require_max_new_tokens,
    require_one_of,
    require_str,
)
from narration_worker.protocol import Controls, WorkerErrorCode
from narration_worker.qwen_settings import ATTN_IMPLEMENTATIONS, DTYPES, SettingsError, parse_settings
from narration_worker.wav import write_float32_mono

from .engine import EngineError, QwenEngine, Rendered, check_snapshot

OFFLINE_ENV: Final = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
MAX_SEED: Final = 0xFFFFFFFF
T = TypeVar("T")


class Qwen3Handler(WorkerHandler):
    """The ``qwen3`` role (see the module docstring)."""

    role = "qwen3"
    controls = Controls(pace=False, context=False, instruct=False)
    fingerprint_packages = (
        "narration-worker-qwen3tts",
        "qwen-tts",
        "transformers",
        "accelerate",
        "tokenizers",
        "torch",
        "torchaudio",
        "librosa",
        "soxr",
        "numpy",
        "soundfile",
    )

    def __init__(self, context: WorkerContext) -> None:
        super().__init__(context)
        os.environ.update(OFFLINE_ENV)
        self._engine: QwenEngine | None = None

    # ------------------------------------------------------------------ load / unload / shutdown
    def op_load(self, request: Request) -> dict[str, Any]:
        """Load Base or VoiceDesign from a pinned snapshot, with the determinism switches and settings given."""
        snapshot = _snapshot_dir(request)
        device = require_str(request, "device")
        if DEVICE_PATTERN.fullmatch(device) is None:
            raise OpError("INVALID_REQUEST", "device must be cpu, cuda or cuda:<n>", {"field": "device"})
        dtype = require_one_of(request, "dtype", DTYPES)
        attn = require_one_of(request, "attn_implementation", ATTN_IMPLEMENTATIONS)
        if "determinism" not in request:
            raise OpError(
                "INVALID_REQUEST",
                "the qwen3 worker needs the determinism switches (section 10.1)",
                {"field": "determinism"},
            )
        determinism = parse_determinism(request["determinism"])
        if "settings" not in request:
            raise OpError(
                "INVALID_REQUEST", "every audio-changing setting must be passed (section 10.1)", {"field": "settings"}
            )
        try:
            settings = parse_settings(request["settings"])
        except SettingsError as exc:
            raise OpError("INVALID_REQUEST", exc.message, {"field": exc.field}) from exc
        try:
            check_snapshot(snapshot)
        except EngineError as exc:
            raise _op_error(exc) from exc
        torch = self.torch()
        if torch is None:
            raise OpError("BACKEND_NOT_INSTALLED", "torch is not installed in this worker's venv: sync it")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise OpError(
                "BACKEND_NOT_INSTALLED",
                "no CUDA device is available to this worker; the Qwen models need an NVIDIA GPU",
                {"field": "device", "device": device},
            )
        apply_determinism(determinism, torch)  # every check passed: from here the load goes ahead
        engine = self._engine or QwenEngine(torch)
        self._engine = engine
        result = self._run(
            lambda: engine.load(
                snapshot_dir=snapshot, device=device, dtype=dtype, attn_implementation=attn, settings=settings
            ),
            None,
        )
        return {"load_s": round(result.load_s, 3), "vram_mb": result.vram_mb}

    def op_unload(self, request: Request) -> dict[str, Any]:
        """Drop the resident model and its prepared voices (idempotent)."""
        self.shutdown()
        return {}

    def shutdown(self) -> None:
        if self._engine is not None:
            self._engine.unload()

    # ------------------------------------------------------------------ Base
    def op_prepare_voice(self, request: Request) -> dict[str, Any]:
        """Build a clip's voice-clone prompt (ICL: the clip and its exact transcript), named by ``voice_hash``."""
        engine = self._loaded("prepare_voice")
        voice_hash = require_str(request, "voice_hash")
        ref_text = require_str(request, "ref_text", allow_empty=True)
        x_vector_only_mode = require_bool(request, "x_vector_only_mode")
        ref_wav = self.input_file(request, "ref_wav")
        try:
            engine.prepare_voice(voice_hash, ref_wav, ref_text, x_vector_only_mode)
        except EngineError as exc:
            raise _op_error(exc) from exc
        return {}

    def op_synthesize(self, request: Request) -> dict[str, Any]:
        """Render ``engine_text`` in a prepared voice with the request's seed; write the raw WAV."""
        engine = self._loaded("synthesize")
        voice_hash = require_str(request, "voice_hash")
        text = require_str(request, "engine_text")
        language = require_str(request, "language")
        seed = require_int(request, "seed", minimum=0, maximum=MAX_SEED)
        cap = _call_cap(request, engine)
        out = self.output_file(request, "out_path")
        if not engine.prepared(voice_hash):
            raise OpError(
                "VOICE_NOT_PREPARED",
                "send prepare_voice for this voice_hash, then retry (prepared voices are evicted, and every load "
                "clears them)",
                {"voice_hash": voice_hash},
            )
        seed_everything(seed, torch=self.torch())
        rendered = self._render(lambda: engine.synthesize(voice_hash, text, language, cap))
        return _write(out, rendered)

    # ------------------------------------------------------------------ VoiceDesign
    def op_design(self, request: Request) -> dict[str, Any]:
        """Design a voice from ``description`` speaking ``design_text`` with the request's seed; write the WAV."""
        engine = self._loaded("design")
        description = require_str(request, "description")
        text = require_str(request, "design_text")
        language = require_str(request, "language")
        seed = require_int(request, "seed", minimum=0, maximum=MAX_SEED)
        cap = _call_cap(request, engine)
        out = self.output_file(request, "out_path")
        seed_everything(seed, torch=self.torch())
        rendered = self._render(lambda: engine.design(description, text, language, cap))
        return _write(out, rendered)

    # ------------------------------------------------------------------ internals
    def _loaded(self, op: str) -> QwenEngine:
        engine = self._engine
        if engine is None or not engine.loaded:
            raise OpError("NOT_LOADED", f"{op} needs a loaded model: send load first")
        return engine

    def _render(self, call: Callable[[], Rendered]) -> Rendered:
        return self._run(call, "RENDER_FAILED")

    def _run(self, call: Callable[[], T], otherwise: WorkerErrorCode | None) -> T:
        """Run an engine call: its ``EngineError`` keeps its code, running out of GPU memory is ``GPU_OOM``,
        and any other exception is ``otherwise`` (``None``: re-raised, which the loop reports as ``INTERNAL``)."""
        try:
            return call()
        except EngineError as exc:
            raise _op_error(exc) from exc
        except OpError:
            raise
        except Exception as exc:
            classified = self.classify(exc)
            if classified is not None:
                raise classified from exc
            if otherwise is None:
                raise
            kind = type(exc)
            raise OpError(
                otherwise,
                f"generation raised {kind.__name__}: {exc}"[:2000],
                {"type": f"{kind.__module__}.{kind.__name__}"},
            ) from exc


def _snapshot_dir(request: Request) -> Path:
    """The ``model`` member's snapshot folder: present first (``BACKEND_NOT_INSTALLED``), then named by its
    revision (section 4)."""
    if "model" not in request:
        raise OpError("INVALID_REQUEST", "the qwen3 worker's load needs model", {"field": "model"})
    ref = request["model"]
    if not isinstance(ref, dict) or not all(isinstance(ref.get(k), str) for k in ("repo", "revision", "snapshot_dir")):
        raise OpError("INVALID_REQUEST", "model must have repo, revision and snapshot_dir", {"field": "model"})
    snapshot = Path(ref["snapshot_dir"])
    if not snapshot.is_absolute():
        raise OpError("INVALID_REQUEST", "model.snapshot_dir must be an absolute path", {"field": "model.snapshot_dir"})
    if not snapshot.is_dir():
        raise OpError(
            "BACKEND_NOT_INSTALLED",
            f"no snapshot of {ref['repo']} at {snapshot}; install the models (narration-admin install)",
            {"field": "model", "repo": ref["repo"], "snapshot_dir": str(snapshot)},
        )
    revision = ref["revision"]
    if REVISION_PATTERN.fullmatch(revision) is None:
        raise OpError("INVALID_REQUEST", "model.revision must be a 40-hex commit SHA", {"field": "model.revision"})
    if snapshot.name != revision:
        raise OpError(
            "INVALID_REQUEST",
            "the snapshot folder must be named by its revision (section 4)",
            {"field": "model.snapshot_dir", "revision": revision, "folder": snapshot.name},
        )
    return snapshot


def _call_cap(request: Request, engine: QwenEngine) -> int:
    """A ``synthesize`` or ``design`` call's own ``max_new_tokens``, checked against the loaded ceiling."""
    ceiling = engine.ceiling
    if ceiling is None:  # the op checked that a model is loaded
        raise OpError("NOT_LOADED", "send load first")
    return require_max_new_tokens(request, ceiling)


def _write(out: Path, rendered: Rendered) -> dict[str, Any]:
    """Write the raw render as float32 samples with the workers' byte-reproducible writer
    (``narration_worker.wav``); the ``protocol.AudioReply`` members (the loop adds ``id`` and ``ok``)."""
    audio = np.ascontiguousarray(rendered.audio, dtype="<f4")
    write_float32_mono(out, audio, rendered.sample_rate)
    return {
        "sample_rate": rendered.sample_rate,
        "samples": int(audio.size),
        "gen_s": round(rendered.gen_s, 3),
        "hit_token_cap": rendered.hit_token_cap,
        "new_tokens": rendered.new_tokens,
        "max_new_tokens": rendered.max_new_tokens,
    }


def _op_error(exc: EngineError) -> OpError:
    return OpError(exc.code, exc.message, exc.details or None)
