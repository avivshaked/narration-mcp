"""The Qwen3-TTS engine: Base (ICL clone) and VoiceDesign, run with explicit settings (design section 10.1).

A model runner only (plan.md P1): it loads one pinned snapshot, renders, and returns raw audio plus how the
generation stopped. It knows nothing of the protocol. ``worker.Qwen3Handler`` adapts requests to it, and the
Phase 0 spikes drive it directly.

The call pattern is the bakeoff's (``models/qwen3-tts/generate.py``): ``Qwen3TTSModel.from_pretrained`` in
bfloat16 with ``sdpa`` attention, ``create_voice_clone_prompt(ref_audio=(wav, sr), ref_text)`` once per
voice, then ``generate_voice_clone(text, language, voice_clone_prompt)``; and
``generate_voice_design(text, instruct, language)``. What differs is that every audio-changing setting is
passed explicitly (``settings``), and that the engine records how generation stopped.

**Token cap.** Each ``synthesize`` or ``design`` call passes its own ``max_new_tokens`` (design section 10.1,
DC-4): at least 2 (qwen-tts's ``min_new_tokens``) and at most the loaded ceiling,
``settings.generation.max_new_tokens``. qwen-tts returns only waveforms. To tell a render that ended on the
codec's end token from one cut off at the cap, the engine wraps two methods **on the loaded model instance**
(never on the class): the talker's ``generate``, whose ``sequences`` show whether the last sampled token was
the end token and whose ``max_new_tokens`` is the cap actually applied, and the model's ``generate``, whose
codes are what gets decoded. The wrappers pass everything through unchanged.

- ``new_tokens`` is the number of codec frames decoded: the talker's steps less one, because the last
  sampled token (the end token, or the one the cap cut off) is never decoded. So it is at most the cap less
  one. The codec runs at 12.5 frames per second of audio (1920 samples at 24 kHz; KNOW, spike h+i).
- ``hit_token_cap`` is true exactly when the last sampled token is not the end token.

**Prepared voices** are kept per loaded model, at most ``MAX_PREPARED_VOICES``, the least recently used
evicted first, and every ``load`` clears them. A caller whose ``synthesize`` gets ``VOICE_NOT_PREPARED``
prepares the voice again and retries: that is expected, not a fault.

The caller seeds (``narration_worker.determinism.seed_everything``) immediately before ``synthesize`` or
``design``, and applies the determinism switches before ``load``.

**One exception to the switches (KNOW, spike h+i, 2026-09-26).** With ``torch.use_deterministic_algorithms``
on, torch 2.11 sends CUDA ``replicate`` padding through a decomposition (``torch._decomp._replication_pad``)
whose dtype wrapper casts every tensor argument to float. The Mimi encoder inside Qwen's speech tokenizer
(transformers 4.57.3, ``MimiModel.downsample``) passes its right padding as a tensor, so ``prepare_voice``
fails with ``_unsafe_index found unexpected index type Float``. Padding's forward pass is a gather, so the
engine turns deterministic algorithms off for the whole ``create_voice_clone_prompt`` call and restores them
after (``_deterministic_algorithms_suspended``). Nothing else is exempt: generation and decoding run with every
switch the caller set. Spike (d) showed renders are bit-exact with this in place (ADR 0002).
"""

from __future__ import annotations

import dataclasses
import gc
import logging
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import soundfile as sf
from narration_worker.errors import is_transient_load_error
from narration_worker.handler import MIN_MAX_NEW_TOKENS
from narration_worker.protocol import QwenSettings, WorkerErrorCode

from .settings import (
    DTYPES,
    ModelKind,
    SettingsError,
    SnapshotUnreadable,
    ceiling_of,
    generation_kwargs,
    read_model_kind,
)

log = logging.getLogger(__name__)

BASE: Final = "base"
VOICE_DESIGN: Final = "voice_design"
SUPPORTED_MODEL_TYPES: Final = (BASE, VOICE_DESIGN)
MAX_PREPARED_VOICES: Final = 16
"""Voice prompts kept at once (each holds its reference codes and speaker embedding on the device). The least
recently used is evicted first, so ``VOICE_NOT_PREPARED`` means "prepare it again and retry"."""
INSTALL_HINT: Final = "reinstall the models and sync the worker's venv (narration-admin install)"


class EngineError(Exception):
    """A failure the protocol reports with a code (``narration_worker.protocol.WORKER_ERROR_CODES``)."""

    def __init__(self, code: WorkerErrorCode, message: str, details: dict[str, object] | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code: WorkerErrorCode = code
        self.message = message
        self.details = details or {}


@dataclass(frozen=True, slots=True)
class LoadResult:
    """What ``load`` measured: wall time, the allocator's reserved VRAM (``None`` on the CPU), the model type."""

    load_s: float
    vram_mb: int | None
    model_type: str
    reused: bool


@dataclass(frozen=True, slots=True)
class TalkerStop:
    """How the talker's generation ended: ``steps`` tokens sampled, the last one the end token or not."""

    steps: int
    ended_on_eos: bool
    max_new_tokens: int | None


@dataclass(frozen=True, slots=True)
class Rendered:
    """One render: mono float32 audio, and how generation stopped.

    ``new_tokens`` is the number of codec frames decoded: ``talker_steps`` less one, so at most
    ``max_new_tokens`` less one, at 12.5 frames per second of audio. ``hit_token_cap`` is true when the last
    sampled token is not the end token, so generation stopped at ``max_new_tokens`` and the audio may be cut
    off (``TOKEN_CAP_HIT``, section 11.1; the verdict is the server's). ``max_new_tokens`` is the cap the talker
    was given for this call.
    """

    audio: np.ndarray
    sample_rate: int
    new_tokens: int
    hit_token_cap: bool
    talker_steps: int
    gen_s: float
    max_new_tokens: int


@dataclass(frozen=True, slots=True)
class _Loaded:
    wrapper: Any  # qwen_tts.Qwen3TTSModel
    snapshot_dir: Path
    device: str
    dtype: str
    attn_implementation: str
    model_type: str
    settings: QwenSettings
    eos_token_id: int

    def identity(self) -> tuple[Path, str, str, str]:
        """What a model load depends on; ``settings`` apply per render and can change without a reload."""
        return (self.snapshot_dir, self.device, self.dtype, self.attn_implementation)


class _Recorder:
    """Wraps one bound method on a model instance and keeps what the last call returned."""

    def __init__(self, original: Callable[..., Any], observe: Callable[[Any, dict[str, Any]], None]) -> None:
        self.original = original
        self._observe = observe

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self.original(*args, **kwargs)
        self._observe(result, kwargs)
        return result


class QwenEngine:
    """One Qwen3-TTS model at a time (design section 4: one resident group), and its prepared voices."""

    def __init__(self, torch: Any) -> None:
        self._torch = torch
        self._loaded: _Loaded | None = None
        self._voices: OrderedDict[str, list[Any]] = OrderedDict()
        self._last_stop: TalkerStop | None = None
        self._last_codes: int | None = None

    # ------------------------------------------------------------------ state
    @property
    def loaded(self) -> bool:
        return self._loaded is not None

    @property
    def model_type(self) -> str | None:
        return self._loaded.model_type if self._loaded else None

    @property
    def settings(self) -> QwenSettings | None:
        return self._loaded.settings if self._loaded else None

    @property
    def ceiling(self) -> int | None:
        """The loaded ``settings.generation.max_new_tokens``: the most a call's own cap may be."""
        return ceiling_of(self._loaded.settings["generation"]) if self._loaded else None

    def prepared(self, voice_hash: str) -> bool:
        return voice_hash in self._voices

    # ------------------------------------------------------------------ load / unload
    def load(
        self,
        *,
        snapshot_dir: Path,
        device: str,
        dtype: str,
        attn_implementation: str,
        settings: QwenSettings,
    ) -> LoadResult:
        """Load the snapshot's model (offline, from the folder itself), replacing any other model.

        The same snapshot, device, dtype and attention again keeps the resident model and only takes the new
        ``settings``. Every ``load`` clears the prepared voices, reused model or not, so no state carries from
        one ``load`` to the next. The caller has checked that ``snapshot_dir`` exists and is named by its
        revision, and should have called ``check_snapshot`` before applying the determinism switches.
        """
        kind = check_snapshot(snapshot_dir)
        if dtype not in DTYPES:
            raise EngineError("INVALID_REQUEST", f"dtype must be one of {', '.join(DTYPES)}", {"field": "dtype"})
        self._voices.clear()
        current = self._loaded
        if current is not None and current.identity() == (snapshot_dir, device, dtype, attn_implementation):
            self._loaded = dataclasses.replace(current, settings=settings)
            return LoadResult(load_s=0.0, vram_mb=self._reserved_mb(device), model_type=current.model_type, reused=True)
        self.unload()
        try:
            return self._load(kind, snapshot_dir, device, dtype, attn_implementation, settings)
        except BaseException:
            self._release_memory()  # whatever a failed load allocated
            raise

    def _load(
        self,
        kind: ModelKind,
        snapshot_dir: Path,
        device: str,
        dtype: str,
        attn_implementation: str,
        settings: QwenSettings,
    ) -> LoadResult:
        try:  # heavy: imports transformers; the offline variables are already set
            # CI type-checks this worker without the model packages; in the worker's own venv this resolves.
            from qwen_tts import Qwen3TTSModel  # pyright: ignore[reportMissingImports]
        except (ImportError, OSError) as exc:
            raise _load_error(exc, "qwen-tts cannot be imported", {"module": "qwen_tts"}) from exc
        torch = self._torch
        started = time.perf_counter()
        try:
            wrapper = Qwen3TTSModel.from_pretrained(
                str(snapshot_dir),
                device_map=device,
                dtype=getattr(torch, dtype),
                attn_implementation=attn_implementation,
                local_files_only=True,
            )
        except OSError as exc:  # a missing or unreadable weights or tokenizer file in the snapshot
            raise _load_error(
                exc, f"the snapshot {snapshot_dir.name} cannot be loaded", {"field": "model.snapshot_dir"}
            ) from exc
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        load_s = time.perf_counter() - started
        model_type = str(wrapper.model.tts_model_type)
        if model_type != kind.tts_model_type:
            raise EngineError(
                "INTERNAL", f"config.json says {kind.tts_model_type!r} but the model loaded as {model_type!r}"
            )
        eos = int(wrapper.model.config.talker_config.codec_eos_token_id)
        self._install_recorders(wrapper, eos)
        self._loaded = _Loaded(
            wrapper=wrapper,
            snapshot_dir=snapshot_dir,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            model_type=model_type,
            settings=settings,
            eos_token_id=eos,
        )
        log.info("loaded %s (%s) from %s in %.1f s", model_type, dtype, snapshot_dir.name, load_s)
        return LoadResult(load_s=load_s, vram_mb=self._reserved_mb(device), model_type=model_type, reused=False)

    def unload(self) -> None:
        """Drop the model and every prepared voice, and return the cached device memory.

        It always collects garbage and empties torch's CUDA cache (when CUDA has started), also when nothing is
        loaded: a ``load`` that failed part-way may have left memory behind.
        """
        self._loaded = None
        self._voices.clear()
        self._release_memory()

    def _release_memory(self) -> None:
        gc.collect()
        cuda = getattr(self._torch, "cuda", None)
        if cuda is not None and cuda.is_available() and cuda.is_initialized():
            cuda.empty_cache()

    # ------------------------------------------------------------------ Base: voices and synthesis
    def prepare_voice(self, voice_hash: str, ref_wav: Path, ref_text: str, x_vector_only_mode: bool) -> None:
        """Build the voice-clone prompt for a clip once; later ``synthesize`` calls name it by ``voice_hash``.

        ``x_vector_only_mode`` false is ICL (reference audio and its exact transcript), the only mode the
        design uses; it is passed explicitly either way (section 10.1). At most ``MAX_PREPARED_VOICES`` are
        kept; the least recently prepared or used is evicted first.
        """
        loaded = self._require(BASE, "prepare_voice")
        if not x_vector_only_mode and not ref_text.strip():
            raise EngineError(
                "INVALID_REQUEST", "ref_text is required when x_vector_only_mode is false (ICL)", {"field": "ref_text"}
            )
        audio, sample_rate = read_mono_float32(ref_wav)
        with _deterministic_algorithms_suspended(self._torch):
            items = loaded.wrapper.create_voice_clone_prompt(
                ref_audio=(audio, sample_rate), ref_text=ref_text, x_vector_only_mode=x_vector_only_mode
            )
        self._voices[voice_hash] = items
        self._voices.move_to_end(voice_hash)
        while len(self._voices) > MAX_PREPARED_VOICES:
            self._voices.popitem(last=False)

    def synthesize(self, voice_hash: str, text: str, language: str, max_new_tokens: int) -> Rendered:
        """Clone a prepared voice speaking ``text``, generating at most ``max_new_tokens`` (the caller has
        seeded)."""
        loaded = self._require(BASE, "synthesize")
        cap = self._check_cap(loaded, max_new_tokens)
        prompt = self._voices.get(voice_hash)
        if prompt is None:
            raise EngineError(
                "VOICE_NOT_PREPARED",
                "prepare_voice was not called for this voice_hash, or it was evicted: prepare it again",
                {"voice_hash": voice_hash},
            )
        self._voices.move_to_end(voice_hash)
        self._check_language(loaded, language)
        settings = loaded.settings
        return self._render(
            loaded,
            cap,
            lambda: loaded.wrapper.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=prompt,
                non_streaming_mode=settings["non_streaming_mode"],
                **_call_kwargs(settings, cap),
            ),
        )

    # ------------------------------------------------------------------ VoiceDesign
    def design(self, description: str, text: str, language: str, max_new_tokens: int) -> Rendered:
        """Design a voice from a description, speaking ``text``, generating at most ``max_new_tokens`` (the
        caller has seeded)."""
        loaded = self._require(VOICE_DESIGN, "design")
        cap = self._check_cap(loaded, max_new_tokens)
        self._check_language(loaded, language)
        settings = loaded.settings
        return self._render(
            loaded,
            cap,
            lambda: loaded.wrapper.generate_voice_design(
                text=text,
                instruct=description,
                language=language,
                non_streaming_mode=settings["non_streaming_mode"],
                **_call_kwargs(settings, cap),
            ),
        )

    # ------------------------------------------------------------------ internals
    def _require(self, model_type: str, op: str) -> _Loaded:
        loaded = self._loaded
        if loaded is None:
            raise EngineError("NOT_LOADED", f"{op} needs a model: send load first")
        if loaded.model_type != model_type:
            wanted = "Base" if model_type == BASE else "VoiceDesign"
            raise EngineError(
                "INVALID_REQUEST",
                f"{op} needs the {wanted} model; the loaded model is {loaded.model_type!r}",
                {"op": op, "loaded": loaded.model_type},
            )
        return loaded

    @staticmethod
    def _check_cap(loaded: _Loaded, max_new_tokens: int) -> int:
        ceiling = ceiling_of(loaded.settings["generation"])
        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or not MIN_MAX_NEW_TOKENS <= max_new_tokens <= ceiling
        ):
            raise EngineError(
                "INVALID_REQUEST",
                f"max_new_tokens must be an integer from {MIN_MAX_NEW_TOKENS} to the loaded ceiling ({ceiling})",
                {"field": "max_new_tokens", "ceiling": ceiling},
            )
        return max_new_tokens

    @staticmethod
    def _check_language(loaded: _Loaded, language: str) -> None:
        supported = loaded.wrapper.get_supported_languages()
        if supported is not None and language.lower() not in supported:
            raise EngineError(
                "INVALID_REQUEST",
                f"language {language!r} is not supported; supported: {', '.join(supported)}",
                {"field": "language"},
            )

    def _render(self, loaded: _Loaded, cap: int, call: Callable[[], tuple[list[Any], int]]) -> Rendered:
        self._last_stop = None
        self._last_codes = None
        torch = self._torch
        started = time.perf_counter()
        wavs, sample_rate = call()
        if loaded.device.startswith("cuda"):
            torch.cuda.synchronize(loaded.device)
        gen_s = time.perf_counter() - started
        stop, codes = self._last_stop, self._last_codes
        if stop is None or codes is None:
            raise EngineError("INTERNAL", "the generation recorders saw no call: the qwen-tts call path changed")
        if stop.max_new_tokens != cap:
            raise EngineError(
                "INTERNAL",
                f"the talker was given max_new_tokens={stop.max_new_tokens}, not the call's {cap}: "
                "the qwen-tts call path changed",
            )
        audio = to_mono_float32(wavs[0])
        return Rendered(
            audio=audio,
            sample_rate=int(sample_rate),
            new_tokens=codes,
            hit_token_cap=not stop.ended_on_eos,
            talker_steps=stop.steps,
            gen_s=gen_s,
            max_new_tokens=cap,
        )

    def _install_recorders(self, wrapper: Any, eos_token_id: int) -> None:
        model = wrapper.model

        def on_talker(result: Any, kwargs: dict[str, Any]) -> None:
            sequences = result.sequences
            steps = int(sequences.shape[1])
            last = int(sequences[0, steps - 1]) if steps else None
            cap = kwargs.get("max_new_tokens")
            self._last_stop = TalkerStop(
                steps=steps, ended_on_eos=last == eos_token_id, max_new_tokens=cap if isinstance(cap, int) else None
            )

        def on_model(result: Any, kwargs: dict[str, Any]) -> None:
            codes_list = result[0]
            self._last_codes = int(codes_list[0].shape[0])

        model.talker.generate = _Recorder(model.talker.generate, on_talker)
        model.generate = _Recorder(model.generate, on_model)

    def _reserved_mb(self, device: str) -> int | None:
        if not device.startswith("cuda"):
            return None
        return int(self._torch.cuda.memory_reserved(device) // (1024 * 1024))


def check_snapshot(snapshot_dir: Path) -> ModelKind:
    """What a snapshot folder holds, if this worker can run it; before anything else of a ``load`` happens.

    ``BACKEND_NOT_INSTALLED`` when its ``config.json`` is missing or unreadable (a broken install);
    ``INVALID_REQUEST`` (``field`` ``model.snapshot_dir``) for another model's snapshot, or a Qwen3-TTS model
    other than Base and VoiceDesign.
    """
    try:
        kind = read_model_kind(snapshot_dir)
    except SnapshotUnreadable as exc:
        raise EngineError(
            "BACKEND_NOT_INSTALLED",
            f"{exc.message}; {INSTALL_HINT}",
            {"field": exc.field, "snapshot_dir": str(snapshot_dir)},
        ) from exc
    except SettingsError as exc:
        raise EngineError(
            "INVALID_REQUEST",
            f"{exc.message}; this worker runs Qwen3-TTS Base and VoiceDesign",
            {"field": exc.field},
        ) from exc
    if kind.tts_model_type not in SUPPORTED_MODEL_TYPES:
        raise EngineError(
            "INVALID_REQUEST",
            f"the snapshot holds a {kind.tts_model_type!r} model; this worker runs Base and VoiceDesign",
            {"field": "model.snapshot_dir", "tts_model_type": kind.tts_model_type},
        )
    return kind


def _load_error(exc: BaseException, what: str, details: dict[str, object]) -> EngineError:
    """A failure to import qwen-tts or to read the snapshot's files, as ``narration_worker`` classifies one.

    A file another process holds (``errors.is_transient_load_error``: an antivirus scanning a DLL or the
    weights, say) is ``INTERNAL`` with ``details.transient``, since a retry gets past it; anything else is a
    broken install, ``BACKEND_NOT_INSTALLED`` with the hint.
    """
    error = f"{type(exc).__name__}: {exc}"[:2000]
    if is_transient_load_error(exc):
        return EngineError(
            "INTERNAL", f"{what} because another process holds a file ({error}); try again",
            {**details, "error": error, "transient": True},
        )  # fmt: skip
    return EngineError("BACKEND_NOT_INSTALLED", f"{what} ({error}); {INSTALL_HINT}", {**details, "error": error})


def _call_kwargs(settings: QwenSettings, cap: int) -> dict[str, bool | int | float]:
    """The ten sampling values, every one explicit, with this call's own ``max_new_tokens``."""
    return {**generation_kwargs(settings["generation"]), "max_new_tokens": cap}


@contextmanager
def _deterministic_algorithms_suspended(torch: Any) -> Iterator[bool]:
    """Turn ``torch.use_deterministic_algorithms`` off inside the block and restore it (mode and warn-only)
    after; yields whether it was on. See the module docstring for why the voice prompt's encode needs it."""
    enabled = bool(torch.are_deterministic_algorithms_enabled())
    warn_only = bool(torch.is_deterministic_algorithms_warn_only_enabled())
    if enabled:
        torch.use_deterministic_algorithms(False)
    try:
        yield enabled
    finally:
        if enabled:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)


def read_mono_float32(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV as float32 mono (channels averaged); ``UNSUPPORTED_AUDIO`` if it cannot be read."""
    try:
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    except (sf.LibsndfileError, RuntimeError, OSError) as exc:
        raise EngineError("UNSUPPORTED_AUDIO", f"cannot read {path.name}: {exc}", {"path": str(path)}) from exc
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0:
        raise EngineError("UNSUPPORTED_AUDIO", f"{path.name} holds no audio", {"path": str(path)})
    return audio, int(sample_rate)


def to_mono_float32(audio: Any) -> np.ndarray:
    """A 1-D float32 array from what qwen-tts returns (a numpy array; a tensor is moved to the CPU)."""
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    array = np.asarray(audio, dtype=np.float32)
    return array.reshape(-1) if array.ndim != 1 else array
