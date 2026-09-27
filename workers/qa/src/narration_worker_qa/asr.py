"""Whisper-large-v3 transcription with word times (design sections 4 and 11.1 step 2; Appendix A ``transcribe``;
plan.md WP22).

**Decoding, pinned** (every setting that changes the output is passed explicitly, never left to a library default,
as section 10.1 asks of Qwen): English, **five beams**, no sampling, temperature 0 and no temperature fallback,
sequential long-form transcription (Whisper's own algorithm: 30 s windows, each after the last timestamp of the
one before) **not conditioned on the previous window's text**, timestamps on. That is the bake-off's decoding, the
one its evidence (WER 5.5–8.6 %) was made with. The pinned snapshot's ``generation_config.json`` supplies only the
model's own token tables (the language, task, timestamp and suppressed tokens, the alignment heads and the
448-token decoder length), which its revision pins.

**A change to section 11.1 step 2** (DC-14, ADR 0004, accepted), which said greedy and conditioned on the
previous text. KNOW
(``spikes/acceptance-wp22/decoding.py`` and ``decoding.json``, the bake-off's six clone takes, through this
class): greedy and conditioned, with no fallback, loops on two of the six whole takes (about 85 words repeated,
WER 0.40 and 0.40), and OpenAI's temperature fallback does not stop it (0.40 and 0.45); greedy and not
conditioned gives WER 0.051–0.098 (mean 0.071) and needs about 2.8 GB less VRAM; five beams, not conditioned,
gives exactly the bake-off's WER on all six (mean 0.064). Switching is this one constant.

Two traps the explicit settings avoid (KNOW, transformers 5.17.0's source):

- the ``automatic-speech-recognition`` pipeline fills an unset ``num_beams`` with **5** ("follows openai's whisper
  implementation"), so a pipeline called without a generation config decodes with beam search. The bake-off's
  ``eval/evaluate.py`` called it that way, which is why its WERs were made with five beams and without
  conditioning. This module passes its own generation config, so the pipeline's defaults never apply;
- word timestamps come from cross-attention (dynamic time warping over the alignment heads), and asking for them
  switches the model's attention to ``eager`` inside ``generate``. The model is therefore loaded with ``eager``
  attention, so the same audio decodes the same way whether or not word times are asked for.

The model runs in float16 on a CUDA device and in float32 on the CPU (fp16 on the CPU is slow and not what the
evidence used). Audio is mixed to mono and resampled to 16 kHz with ``scipy.signal.resample_poly``, as the
bake-off did (``align.to_mono_16k``).
"""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from .align import SAMPLE_RATE, InvalidRequest, NotLoaded, is_broken_file, load_error
from .gpu import release_cached_memory
from .snapshots import check_architecture

ARCHITECTURE: Final = "WhisperForConditionalGeneration"
MODEL_TYPE: Final = "whisper"
ATTN_IMPLEMENTATION: Final = "eager"
WINDOW_S: Final = 30.0
"""Whisper's input window: a ``long_form: false`` request must fit one."""
LANGUAGES: Final[Mapping[str, str]] = {"english": "en", "en": "en"}
"""The languages ``transcribe`` accepts, by name (the job engine sends ``names.LANGUAGE``, "English", as it does to
Qwen) or by Whisper's code, in any case, and the language token each maps to. v1 transcribes English only (section
11.1): the number reader and the QA thresholds are English."""
DECODING: Final[Mapping[str, Any]] = {
    "task": "transcribe",
    "num_beams": 5,
    "do_sample": False,
    "temperature": 0.0,
    "condition_on_prev_tokens": False,
    "compression_ratio_threshold": None,
    "logprob_threshold": None,
    "no_speech_threshold": None,
}
"""The decoding settings that change the transcript, passed explicitly on every call (see the module docstring for
why five beams and no conditioning). The thresholds are None: no temperature fallback and no skipping of windows
judged silent, so every window is decoded once, the same way every time."""
GENERATION_CONFIG_KEYS: Final = ("num_beams", "do_sample")
"""The members of ``DECODING`` that live on the generation config rather than in ``generate``'s arguments."""


def whisper_language(value: str) -> str:
    """Whisper's language code for a language as the request names it; ``InvalidRequest`` (``language``) for any
    other."""
    code = LANGUAGES.get(value.strip().lower())
    if code is None:
        raise InvalidRequest(
            f"language {value!r} is not supported: the QA worker transcribes English (send 'English')",
            {"field": "language", "supported": sorted(set(LANGUAGES))},
        )
    return code


def decoding_settings(language: str, *, long_form: bool, decoding: Mapping[str, Any] = DECODING) -> dict[str, Any]:
    """The ``generate`` arguments of one call besides the generation config: ``DECODING`` without the
    generation-config members, the language, and a single window when ``long_form`` is false.

    The pipeline adds the timestamps from its own ``return_timestamps`` argument: ``True`` turns on Whisper's
    timestamp tokens (which long-form decoding needs), and ``"word"`` also asks ``generate`` for token timestamps.
    """
    settings = {k: v for k, v in decoding.items() if k not in GENERATION_CONFIG_KEYS}
    settings.update(language=language, force_unique_generate_call=not long_form)
    return settings


def words_of(output: Mapping[str, Any], *, word_timestamps: bool) -> tuple[str, list[dict[str, Any]]]:
    """The transcript and its words from the pipeline's output.

    With word times, each word is a chunk of the pipeline's ``return_timestamps="word"`` output, stripped, with its
    start and end in seconds (null where Whisper gave none). Without, the words are the text's whitespace-separated
    tokens with null times. ``probability`` is always null: the worker does not score words.
    """
    text = str(output.get("text") or "").strip()
    if not word_timestamps:
        return text, [{"text": w, "start_s": None, "end_s": None, "probability": None} for w in text.split()]
    words: list[dict[str, Any]] = []
    for chunk in output.get("chunks") or []:
        word = str(chunk.get("text") or "").strip()
        if not word:
            continue
        start, end = (chunk.get("timestamp") or (None, None))[:2]
        words.append({"text": word, "start_s": _seconds(start), "end_s": _seconds(end), "probability": None})
    return text, words


def _seconds(value: object) -> float | None:
    return None if value is None else round(float(value), 3)  # type: ignore[arg-type]


class WhisperAsr:
    """Whisper-large-v3 on one device, decoding as pinned (module docstring).

    ``torch`` is the imported module with the CPU thread cap applied (``WorkerHandler.torch()``). ``decoding`` is
    ``DECODING`` in the worker; a spike may pass another set of the same keys to compare decodings.
    """

    def __init__(self, torch: Any, decoding: Mapping[str, Any] = DECODING) -> None:
        self._torch = torch
        self._decoding = dict(decoding)
        self._pipeline: Any = None
        self._generation: Any = None
        self.repo = ""
        self.revision = ""
        self.device = ""

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def load(self, repo: str, revision: str, snapshot_dir: Path, device: str) -> float:
        """Load the model, its tokenizer and feature extractor offline from the snapshot; returns the seconds taken.

        ``BackendMissing`` (with the install hint) when the snapshot is not Whisper (``config.json``) or a file is
        missing or damaged; ``INTERNAL`` with ``details.transient`` for a file another process holds. The weights
        load with ``weights_only=True``.
        """
        check_architecture(snapshot_dir, ARCHITECTURE, MODEL_TYPE, "ASR")
        torch = self._torch
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

        start = time.perf_counter()
        try:
            processor = WhisperProcessor.from_pretrained(str(snapshot_dir), local_files_only=True)
            model: Any = WhisperForConditionalGeneration.from_pretrained(
                str(snapshot_dir),
                local_files_only=True,
                weights_only=True,
                dtype=dtype,
                attn_implementation=ATTN_IMPLEMENTATION,
            )
        except Exception as exc:
            if not is_broken_file(exc):
                raise
            raise load_error(
                exc, f"the ASR snapshot {snapshot_dir.name} cannot be loaded", {"path": str(snapshot_dir)}
            ) from exc
        model.to(device)
        model.eval()
        generation = copy.deepcopy(model.generation_config)
        generation.update(**{k: self._decoding[k] for k in GENERATION_CONFIG_KEYS})
        for key in ("compression_ratio_threshold", "logprob_threshold", "no_speech_threshold"):
            setattr(generation, key, self._decoding[key])
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
        )
        self._generation = generation
        self.repo, self.revision, self.device = repo, revision, device
        return time.perf_counter() - start

    def unload(self) -> None:
        self._pipeline = None
        self._generation = None

    def transcribe(
        self, audio_16k: npt.NDArray[np.float32], language: str, *, word_timestamps: bool, long_form: bool
    ) -> dict[str, Any]:
        """The ``TranscribeReply`` members other than ``id`` and ``ok`` for mono 16 kHz audio.

        ``language`` is a request's language (``whisper_language``). ``long_form`` false decodes one 30 s window,
        so longer audio is ``InvalidRequest`` (``long_form``). No audio at all is an empty transcript.
        """
        if self._pipeline is None or self._generation is None:
            raise NotLoaded("transcribe needs the ASR model loaded: send load with models.asr")
        code = whisper_language(language)
        seconds = audio_16k.shape[0] / SAMPLE_RATE
        if not long_form and seconds > WINDOW_S:
            raise InvalidRequest(
                f"the audio is {seconds:.1f} s long, beyond one {WINDOW_S:g} s window: send long_form true",
                {"field": "long_form", "duration_s": round(seconds, 3)},
            )
        reply: dict[str, Any] = {"text": "", "words": [], "model": self.repo, "revision": self.revision}
        if audio_16k.shape[0] == 0:
            return reply
        generate = decoding_settings(code, long_form=long_form, decoding=self._decoding)
        generate["generation_config"] = copy.deepcopy(self._generation)
        try:
            output = self._pipeline(
                {"raw": np.ascontiguousarray(audio_16k, dtype=np.float32), "sampling_rate": SAMPLE_RATE},
                return_timestamps="word" if word_timestamps else True,
                generate_kwargs=generate,
            )
        finally:
            release_cached_memory(self._torch, self.device)
        reply["text"], reply["words"] = words_of(output, word_timestamps=word_timestamps)
        return reply
