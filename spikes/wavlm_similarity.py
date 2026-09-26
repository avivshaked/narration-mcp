"""WavLM-SV speaker similarity between WAV files, as the bakeoff's evidence measured it (for WP20's spikes).

The method of the bakeoff's ``eval/evaluate.py`` (``Scorer.embed``): the audio mixed to mono and resampled to
16 kHz with ``scipy.signal.resample_poly``, the ``microsoft/wavlm-base-plus-sv`` feature extractor,
``WavLMForXVector`` embeddings, L2-normalised; the similarity is their dot product (cosine). It runs on the
CPU, offline, from the snapshot the evidence used (revision ``feb593a6…``, the repo's ``main``). The QA
worker's own ``embed`` op is WP22's; this is a spike tool, not the service's.

Run in the QA worker's venv (transformers 5.17.0, the version the evidence used)::

    workers/qa/.venv/Scripts/python.exe spikes/wavlm_similarity.py --pairs pairs.json --out result.json

``pairs.json`` is a list of objects ``{"a": <wav>, "b": <wav>, "a_span": [start, end]?, "b_span": [...]?,
"label": <text>?}``; a span cuts ``[start, end)`` samples out of a file first (a segment of a concatenated
take). The result repeats each pair with its ``similarity``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from math import gcd
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_spike_common as common


def to16k(audio: Any, sample_rate: int) -> Any:
    """The bakeoff's ``to16k``: mono, float32, polyphase resampling to 16 kHz."""
    import numpy as np
    from scipy.signal import resample_poly

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sample_rate == 16000:
        return audio
    g = gcd(16000, sample_rate)
    return resample_poly(audio, 16000 // g, sample_rate // g).astype(np.float32)


class Embedder:
    """WavLM-base-plus-sv x-vectors on the CPU, L2-normalised."""

    def __init__(self) -> None:
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        snapshot, self.revision = common.snapshot(common.MODEL_SV, common.SV_REVISION_EVIDENCE)
        torch.set_num_threads(common.CPU_THREADS)
        self._torch = torch
        self._extractor = AutoFeatureExtractor.from_pretrained(str(snapshot), local_files_only=True)
        self._model = WavLMForXVector.from_pretrained(str(snapshot), local_files_only=True).eval()

    def embed(self, audio16: Any) -> Any:
        torch = self._torch
        with torch.no_grad():
            inputs = self._extractor(audio16, sampling_rate=16000, return_tensors="pt")
            vector = self._model(**inputs).embeddings[0]
            return torch.nn.functional.normalize(vector, dim=-1).numpy()


def read(path: str, span: list[int] | None) -> Any:
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32")
    if span is not None:
        audio = audio[span[0] : span[1]]
    return to16k(audio, rate)


def similarities(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    embedder = Embedder()
    cache: dict[tuple[str, tuple[int, ...] | None], Any] = {}

    def vector(path: str, span: list[int] | None) -> Any:
        key = (path, tuple(span) if span else None)
        if key not in cache:
            cache[key] = embedder.embed(read(path, span))
        return cache[key]

    out = []
    for pair in pairs:
        a = vector(pair["a"], pair.get("a_span"))
        b = vector(pair["b"], pair.get("b_span"))
        out.append({**pair, "similarity": round(float(a @ b), 4)})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--pairs", required=True, help="JSON list of pairs to compare")
    parser.add_argument("--out", required=True, help="where to write the result JSON")
    args = parser.parse_args()
    common.prepare_process_env()
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    pairs = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    result = {
        "model": common.MODEL_SV,
        "revision_prefix": common.SV_REVISION_EVIDENCE,
        "device": "cpu",
        "method": "bakeoff eval/evaluate.py Scorer.embed: to16k (resample_poly), WavLMForXVector, L2-normalised",
        "pairs": similarities(pairs),
    }
    common.write_json(Path(args.out), result)
    for pair in result["pairs"]:
        print(f"{pair['similarity']:.4f}  {pair.get('label', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
