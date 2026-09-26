"""Spike (e), step 2 of 2: score the ``non_streaming_mode`` A/B renders (design sections 10.1, 20).

For each render of ``render.py``: the Whisper-large-v3 transcript and its WER against the text (the bakeoff's
method: the transformers ASR pipeline in fp16, and Whisper's English normaliser on both sides), the WavLM-SV
similarity to the voice's own clip (``spk_to_ref``, as in the evidence), and the pace in words per minute.
WER is computed here as a word-level edit distance (jiwer is not in the QA venv; the numbers are the same
definition: substitutions, deletions and insertions over reference words).

Run from the checkout in the QA worker's venv, with the GPU lock held (a few minutes)::

    workers/qa/.venv/Scripts/python.exe spikes/e-non-streaming-ab/score.py

Reads ``.dev/spikes/e/renders.json``; writes ``spikes/e-non-streaming-ab/results.json`` (published).
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_spike_common as common

HERE = Path(__file__).resolve().parent
OUT = common.DEV_SPIKES / "e"


def word_errors(reference: list[str], hypothesis: list[str]) -> int:
    """Levenshtein distance over words (substitutions + deletions + insertions)."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_word != hyp_word)))
        previous = current
    return previous[-1]


def main() -> int:
    common.prepare_process_env()
    import soundfile as sf
    import torch
    from transformers import WhisperProcessor, pipeline
    from wavlm_similarity import Embedder, to16k

    guard = common.NetworkGuard().install()
    data = json.loads((OUT / "renders.json").read_text(encoding="utf-8"))
    snapshot, asr_revision = common.snapshot(common.MODEL_ASR)
    asr = pipeline("automatic-speech-recognition", model=str(snapshot), dtype=torch.float16, device="cuda:0")
    normalise = WhisperProcessor.from_pretrained(str(snapshot)).tokenizer.normalize
    embedder = Embedder()
    clip_vectors: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for render in data["renders"]:
        audio, rate = sf.read(render["wav"], dtype="float32")
        audio16 = to16k(audio, rate)
        out: dict[str, Any] = cast(
            dict[str, Any],
            asr(
                {"raw": audio16, "sampling_rate": 16000},
                return_timestamps=True,
                generate_kwargs={"language": "english", "task": "transcribe"},
            ),
        )
        heard = str(out["text"]).strip()
        reference, hypothesis = normalise(render["text"]).split(), normalise(heard).split()
        if render["clip"] not in clip_vectors:
            clip_audio, clip_rate = sf.read(render["clip"], dtype="float32")
            clip_vectors[render["clip"]] = embedder.embed(to16k(clip_audio, clip_rate))
        similarity = float(embedder.embed(audio16) @ clip_vectors[render["clip"]])
        rows.append(
            {
                "non_streaming_mode": render["non_streaming_mode"],
                "voice": render["voice"],
                "segment": render["segment"],
                "audio_s": render["audio_s"],
                "new_tokens": render["new_tokens"],
                "wpm": round(len(render["text"].split()) / render["audio_s"] * 60, 1),
                "wer": round(word_errors(reference, hypothesis) / len(reference), 4),
                "spk_to_ref": round(similarity, 4),
                "heard": heard,
            }
        )
        print(
            rows[-1]["non_streaming_mode"],
            rows[-1]["voice"],
            rows[-1]["segment"],
            rows[-1]["wer"],
            rows[-1]["spk_to_ref"],
        )

    def summary(mode: bool) -> dict[str, float]:
        mine = [r for r in rows if r["non_streaming_mode"] == mode]
        return {
            "renders": len(mine),
            "wer_mean": round(statistics.fmean(r["wer"] for r in mine), 4),
            "spk_to_ref_mean": round(statistics.fmean(r["spk_to_ref"] for r in mine), 4),
            "spk_to_ref_min": min(r["spk_to_ref"] for r in mine),
            "wpm_mean": round(statistics.fmean(r["wpm"] for r in mine), 1),
            "audio_s_total": round(sum(r["audio_s"] for r in mine), 2),
        }

    results = {
        "spike": "e (non_streaming_mode A/B for Base)",
        "design_sections": ["10.1", "20 (e)"],
        "rendered_at": data["ran_at"],
        "scored_at": common.now(),
        "software": common.software_facts(torch),
        "model": data["model"],
        "asr": {"model": common.MODEL_ASR, "revision": asr_revision},
        "sv": {"model": common.MODEL_SV, "revision_prefix": common.SV_REVISION_EVIDENCE, "device": "cpu"},
        "rows": rows,
        "summary": {"non_streaming_false": summary(False), "non_streaming_true": summary(True)},
        "network_attempts": guard.attempts,
    }
    common.write_json(HERE / "results.json", results)
    print(json.dumps(results["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
