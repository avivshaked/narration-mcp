"""Whisper's long-form decoding compared on the bakeoff's six clone takes (WP22; design section 11.1 step 2).

The acceptance run (``run.py``) found that the decoding section 11.1 pins, greedy and conditioned on the previous
window's text with no temperature fallback, loops on two of the six whole takes (about 100 words repeated, WER
0.40), while the bakeoff's decoding (five beams, not conditioned) did not. This measures the candidates on the
same takes, through the worker's own class (``narration_worker_qa.asr.WhisperAsr``, float16, eager attention,
the worker's determinism switches, word times on, long-form), and scores them as ``eval/evaluate.py`` did:

- ``greedy_conditioned``: section 11.1 as written (``asr.DECODING`` before this measurement);
- ``greedy_unconditioned``: the same, not conditioned on the previous text;
- ``beam5_unconditioned``: the bakeoff's decoding (the ASR pipeline's defaults), passed explicitly;
- ``greedy_conditioned_fallback``: conditioned, with OpenAI's temperature fallback (0.0 to 1.0 by 0.2) and
  thresholds (compression ratio 2.4, log-probability -1.0, no speech 0.6), seeded before each call.

Run from the checkout in the QA worker's venv, with the GPU lock held (about 12 minutes)::

    workers/qa/.venv/Scripts/python.exe spikes/acceptance-wp22/decoding.py

Writes ``spikes/acceptance-wp22/decoding.json``: numbers keyed by the bakeoff's take ids, no text.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_spike_common as common
from run import SCRIPT, SEEDS, TAKE_DIRS, bakeoff_results, normaliser, wer

HERE = Path(__file__).resolve().parent
DEVICE = "cuda:0"
BASE = {
    "task": "transcribe",
    "num_beams": 1,
    "do_sample": False,
    "temperature": 0.0,
    "condition_on_prev_tokens": True,
    "compression_ratio_threshold": None,
    "logprob_threshold": None,
    "no_speech_threshold": None,
}
CANDIDATES: dict[str, dict[str, Any]] = {
    "greedy_conditioned": dict(BASE),
    "greedy_unconditioned": BASE | {"condition_on_prev_tokens": False},
    "beam5_unconditioned": BASE | {"num_beams": 5, "condition_on_prev_tokens": False},
    "greedy_conditioned_fallback": BASE
    | {
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
    },
}


def main() -> int:
    common.prepare_process_env()
    import soundfile as sf
    import torch
    from narration_worker.determinism import apply_determinism, seed_everything
    from narration_worker_qa.align import to_mono_16k
    from narration_worker_qa.asr import WhisperAsr
    from narration_worker_qa.worker import QA_DETERMINISM

    common.cap_torch_threads(torch)
    apply_determinism(QA_DETERMINISM, torch)
    bakeoff = common.bakeoff_root()
    snapshot, revision = common.snapshot(common.MODEL_ASR)
    normalise = normaliser(snapshot)
    expected = bakeoff_results()
    script = json.loads((bakeoff / "scripts" / f"{SCRIPT}.json").read_text(encoding="utf-8"))
    reference = " ".join(s["text"] for s in script["segments"])
    audio: dict[str, Any] = {}
    for voice, folder in TAKE_DIRS.items():
        for seed in SEEDS:
            data, rate = sf.read(str(bakeoff / "outputs" / f"{folder}-seed{seed}" / f"{SCRIPT}.wav"), dtype="float32")
            audio[f"{voice} seed{seed}"] = to_mono_16k(data, rate)
    reference_words = len(normalise(reference).split())

    results: dict[str, Any] = {
        "check": "Whisper long-form decodings on the bakeoff's clone takes (WP22)",
        "ran_at": common.now(),
        "software": common.software_facts(torch),
        "model": [common.MODEL_ASR, revision],
        "reference_words_normalised": reference_words,
        "bakeoff_wer": {take: row["wer"] for take, row in expected.items()},
        "candidates": {},
    }
    for name, decoding in CANDIDATES.items():
        asr = WhisperAsr(torch, decoding)
        asr.load(common.MODEL_ASR, revision, snapshot, DEVICE)
        per_take: dict[str, Any] = {}
        for take, clip in audio.items():
            seed_everything(0, torch=torch)
            torch.cuda.reset_peak_memory_stats(DEVICE)
            started = time.perf_counter()
            reply = asr.transcribe(clip, "English", word_timestamps=True, long_form=True)
            seconds = time.perf_counter() - started
            per_take[take] = {
                "wer": round(wer(normalise, reference, reply["text"]), 4),
                "hypothesis_words_normalised": len(normalise(reply["text"]).split()),
                "seconds": round(seconds, 1),
                "peak_allocated_mb": common.mb(torch.cuda.max_memory_allocated(DEVICE)),
            }
            print(name, take, per_take[take], flush=True)
        wers = [v["wer"] for v in per_take.values()]
        results["candidates"][name] = {
            "decoding": {k: list(v) if isinstance(v, tuple) else v for k, v in decoding.items()},
            "takes": per_take,
            "mean_wer": round(sum(wers) / len(wers), 4),
            "max_wer": max(wers),
        }
        asr.unload()
        torch.cuda.empty_cache()
    common.write_json(HERE / "decoding.json", results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
