"""WavLM-base-plus-sv speaker embeddings (design sections 4, 10.1 and 11.1; Appendix A ``embed``; plan.md WP22).

An embedding is the model's x-vector for mono 16 kHz audio, L2-normalised, computed as the bake-off's
``eval/evaluate.py`` computed it (``Scorer.embed``): the snapshot's feature extractor (no normalisation, an
attention mask), ``WavLMForXVector`` in float32, ``embeddings[0]``, then ``torch.nn.functional.normalize``. The
cosine similarity of two embeddings is then their dot product, which the server computes.

**Devices.** The model loads on the ``load`` request's device. An ``embed`` names its device too: ``cuda`` runs on
the loaded CUDA device (and needs one); ``cpu`` runs on the CPU, from a CPU copy loaded from the same snapshot on
first use when the model was loaded on a GPU. The canary check runs that way (section 10.1): seconds on the CPU,
without a model swap. The copy is dropped with the model.

**The pinned revision** is ``main`` (``feb593a6…``), whose weights are ``pytorch_model.bin``. KNOW (transformers
5.17.0's source): with the library's default ``use_safetensors=None``, a repo without ``model.safetensors`` loads
``pytorch_model.bin``, so the bake-off's evidence was made from these weights; the Hub's safetensors conversion
(``refs/pr/8``) is a by-product of that load (a background download "for next time"), not what it read. KNOW
(2026-09-26): the two files hold the same 266 float32 tensors, bit for bit, and through this class the worker
reproduces every similarity of ``refs/auditions/voicelock.csv`` (52 rows) and the bake-off's ``spk_to_ref`` and
``spk_consist`` to within 0.0001 (``spikes/acceptance-wp22``). ``main`` is pinned because its snapshot is complete
(``config.json``, ``preprocessor_config.json``); ``refs/pr/8`` holds only the weights. The ``.bin`` is a pickle,
so it loads with ``weights_only=True``, which refuses anything but tensors.

**Windows (DC-15, amended to 60 s).** Audio of up to 60 s is embedded in one pass, exactly as above. Longer
audio is cut into ``ceil(length / 60 s)`` consecutive windows of equal length (differing by at most one sample),
each is embedded in one pass, and the result is the mean of the windows' L2-normalised embeddings, normalised
again. Why: WavLM attends over the whole clip, so one pass's memory grows with the square of its length (KNOW,
``spikes/h-i-qa-load``: 0.6 GB above the resident models at 30 s, 2.2 GB at 60 s, 8.4 GB at 119 s); windows keep
it at the 60 s figure for any length, below transcription's peak. The cost is a slightly different vector for
audio over 60 s; the same spike measures its cosine to the one-pass embedding on a real take.

Why 60 s and not 30 s (KNOW, ``spikes/acceptance-wp22/windows.json``): the bake-off's ``voicelock.csv`` compares
clips of up to 37 s, embedded in one pass. With 30 s windows 3 of its 52 rows moved by more than the acceptance's
0.002 (at most 0.0057); with 60 s windows every clip is embedded in one pass and every row is reproduced (largest
difference 0.0001). The take segments of the evidence are all shorter than 30 s.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from .align import SAMPLE_RATE, InvalidRequest, NotLoaded, UnreadableAudio, is_broken_file, load_error
from .snapshots import check_architecture

ARCHITECTURE: Final = "WavLMForXVector"
MODEL_TYPE: Final = "wavlm"
DEVICES: Final = ("cuda", "cpu")
"""The devices an ``embed`` may name."""
WINDOW_S: Final = 60
"""The longest audio embedded in one pass, in seconds (DC-15, amended by the lead from 30 s to 60 s); longer audio
is embedded in equal windows of at most this."""
WINDOW_SAMPLES: Final = WINDOW_S * SAMPLE_RATE


def windows(audio_16k: npt.NDArray[np.float32], limit: int = WINDOW_SAMPLES) -> list[npt.NDArray[np.float32]]:
    """The audio as it is when it has at most ``limit`` samples; otherwise ``ceil(len / limit)`` consecutive windows
    of equal length (``numpy.array_split``: lengths differ by at most one sample), which together are the audio."""
    count = -(-int(audio_16k.shape[0]) // limit)
    if count <= 1:
        return [audio_16k]
    return list(np.array_split(audio_16k, count))


def min_samples(config: Any) -> int:
    """The shortest audio (16 kHz samples) a WavLM x-vector model can embed: enough feature-encoder frames for its
    TDNN layers' receptive field, plus one, since the statistics pooling takes a standard deviation over the TDNN's
    output frames and one frame has none (NaN). For the pinned model: a 400-sample window every 320 samples, and
    15 + 1 frames for the TDNN (kernels 5, 3, 3, 1, 1; dilations 1, 2, 3, 1, 1), so 5200 samples (0.325 s)."""
    window, stride = 1, 1
    for kernel, step in zip(config.conv_kernel, config.conv_stride, strict=True):
        window += (int(kernel) - 1) * stride
        stride *= int(step)
    frames = 2 + sum((int(k) - 1) * int(d) for k, d in zip(config.tdnn_kernel, config.tdnn_dilation, strict=True))
    return window + (frames - 1) * stride


class WavLmSv:
    """WavLM-base-plus-sv: a model on the loaded device, and a CPU copy made on demand (module docstring).

    ``torch`` is the imported module with the CPU thread cap applied (``WorkerHandler.torch()``).
    """

    def __init__(self, torch: Any) -> None:
        self._torch = torch
        self._models: dict[str, Any] = {}
        self._extractor: Any = None
        self._snapshot: Path | None = None
        self.min_samples = 0
        self.repo = ""
        self.revision = ""
        self.device = ""

    @property
    def loaded(self) -> bool:
        return bool(self._models)

    def load(self, repo: str, revision: str, snapshot_dir: Path, device: str) -> float:
        """Load the model offline from the snapshot onto ``device``; returns the seconds taken.

        ``BackendMissing`` (with the install hint) when the snapshot is not WavLM-SV (``config.json``) or a file is
        missing or damaged; ``INTERNAL`` with ``details.transient`` for a file another process holds.
        """
        check_architecture(snapshot_dir, ARCHITECTURE, MODEL_TYPE, "speaker-verification")
        start = time.perf_counter()
        extractor = self._from_snapshot(snapshot_dir, extractor=True)
        model = self._from_snapshot(snapshot_dir, extractor=False)
        self._extractor = extractor
        self._models = {device: model.to(device).eval()}
        self._snapshot = snapshot_dir
        self.min_samples = min_samples(model.config)
        self.repo, self.revision, self.device = repo, revision, device
        return time.perf_counter() - start

    def _from_snapshot(self, snapshot_dir: Path, *, extractor: bool) -> Any:
        from transformers import AutoFeatureExtractor, WavLMForXVector

        try:
            if extractor:
                return AutoFeatureExtractor.from_pretrained(str(snapshot_dir), local_files_only=True)
            return WavLMForXVector.from_pretrained(
                str(snapshot_dir), local_files_only=True, weights_only=True, dtype=self._torch.float32
            )
        except Exception as exc:
            if not is_broken_file(exc):
                raise
            raise load_error(
                exc,
                f"the speaker-verification snapshot {snapshot_dir.name} cannot be loaded",
                {"path": str(snapshot_dir)},
            ) from exc

    def unload(self) -> None:
        self._models = {}
        self._extractor = None
        self._snapshot = None

    def _model_for(self, device: str) -> tuple[Any, str]:
        """The model to embed on for an ``embed``'s ``device``, and the torch device it is on."""
        if device == "cpu":
            if "cpu" not in self._models:
                if self._snapshot is None:  # pragma: no cover - loaded implies a snapshot
                    raise NotLoaded("embed needs the speaker-verification model loaded: send load with models.sv")
                self._models["cpu"] = self._from_snapshot(self._snapshot, extractor=False).to("cpu").eval()
            return self._models["cpu"], "cpu"
        if not self.device.startswith("cuda"):
            raise InvalidRequest(
                f"embed on cuda needs the speaker-verification model loaded on a CUDA device, not {self.device!r}: "
                "send load with device cuda, or embed on cpu",
                {"field": "device", "loaded_on": self.device},
            )
        return self._models[self.device], self.device

    def embed(self, audio_16k: npt.NDArray[np.float32], device: str) -> dict[str, Any]:
        """The ``EmbedReply`` members other than ``id`` and ``ok`` for mono 16 kHz audio: one pass for up to 60 s,
        else the re-normalised mean of equal windows' embeddings (DC-15; module docstring).

        ``InvalidRequest`` (``device``) for a device other than ``DEVICES``, or ``cuda`` with the model on the CPU;
        ``UnreadableAudio`` (``UNSUPPORTED_AUDIO``, reason ``too_short``) for audio shorter than the model can
        embed (``min_samples``).
        """
        if not self._models or self._extractor is None:
            raise NotLoaded("embed needs the speaker-verification model loaded: send load with models.sv")
        if device not in DEVICES:
            raise InvalidRequest(f"device must be one of {', '.join(DEVICES)}", {"field": "device"})
        if audio_16k.shape[0] < self.min_samples:
            raise UnreadableAudio(
                f"the audio is too short to embed: {audio_16k.shape[0]} samples at 16 kHz, fewer than the "
                f"{self.min_samples} the model needs",
                {
                    "field": "wav",
                    "reason": "too_short",
                    "samples_16k": int(audio_16k.shape[0]),
                    "minimum": self.min_samples,
                },
            )
        model, where = self._model_for(device)
        vectors = [self._one_pass(model, where, window) for window in windows(audio_16k)]
        if len(vectors) == 1:
            vector = vectors[0]
        else:
            mean = np.mean(np.stack(vectors), axis=0)
            vector = mean / np.linalg.norm(mean)
        values = [float(v) for v in vector.tolist()]
        return {"embedding": values, "dim": len(values), "model": self.repo, "revision": self.revision}

    def _one_pass(self, model: Any, where: str, audio_16k: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        """One pass of the bake-off's recipe (module docstring): the L2-normalised x-vector, as float64."""
        torch = self._torch
        features = self._extractor(
            np.ascontiguousarray(audio_16k, dtype=np.float32), sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).to(where)
        with torch.inference_mode():
            embedding = model(**features).embeddings[0]
            vector = torch.nn.functional.normalize(embedding.float(), dim=-1).cpu().numpy()
        return np.asarray(vector, dtype=np.float64)
