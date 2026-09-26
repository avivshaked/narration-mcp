"""Spike (e), step 1 of 2: render the ``non_streaming_mode`` A/B for Base (design sections 10.1, 20).

Section 10.1 pins ``non_streaming_mode=false`` for Base, "which is what every piece of clone evidence used.
Change it only if a Phase 0 A/B justifies ``true``, and that would mean a new engine profile." In qwen-tts
0.1.1 the flag decides how the text reaches the talker: false feeds the text alongside the codec stream
("simulated streaming"), true gives the whole text up front.

This step renders the service's three calibration paragraphs (``narration-en.v1`` cal-01..03) in the d2 and
d4 voices, once with each mode and the same seed, through the worker's engine with the section 10.1
switches. ``score.py`` (in the QA worker's venv) then transcribes and embeds them.

Run from the checkout in the Qwen worker's venv, with the GPU lock held (about 6 minutes)::

    workers/qwen3tts/.venv/Scripts/python.exe spikes/e-non-streaming-ab/render.py [--material <dir>]

Writes the audio and ``renders.json`` under ``.dev/spikes/e/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_spike_common as common

OUT = common.DEV_SPIKES / "e"
DEVICE = "cuda:0"
SEGMENTS = ("cal-01", "cal-02", "cal-03")
VOICES = ("d2", "d4")
SEED = 31


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--material", help="the service's material folder (default: <checkout>/material)")
    args = parser.parse_args()
    common.prepare_process_env()
    texts = common.calibration_paragraphs(common.material_root(args.material))

    import torch
    from narration_qwen3tts.engine import QwenEngine
    from narration_qwen3tts.settings import effective_generation

    common.cap_torch_threads(torch)
    common.apply_switches(torch, on=True)
    snapshot, revision = common.snapshot(common.MODEL_BASE)
    generation = effective_generation(snapshot)
    engine = QwenEngine(torch)
    renders: list[dict[str, Any]] = []
    for mode in (False, True):
        engine.load(
            snapshot_dir=snapshot,
            device=DEVICE,
            dtype="bfloat16",
            attn_implementation="sdpa",
            settings={"non_streaming_mode": mode, "generation": generation},
        )
        for voice in VOICES:
            clip, transcript, _ = common.voice_clip(voice)
            engine.prepare_voice(voice, clip, transcript, x_vector_only_mode=False)
            for segment in SEGMENTS:
                common.seed_all(torch, SEED)
                rendered = engine.synthesize(voice, texts[segment], common.LANGUAGE, common.CEILING)
                wav = OUT / f"nsm-{str(mode).lower()}-{voice}-{segment}.wav"
                common.write_wav(wav, rendered.audio, rendered.sample_rate)
                renders.append(
                    {
                        "non_streaming_mode": mode,
                        "voice": voice,
                        "segment": segment,
                        "seed": SEED,
                        "text": texts[segment],
                        "wav": str(wav),
                        "clip": str(clip),
                        "audio_s": round(rendered.audio.size / rendered.sample_rate, 3),
                        "gen_s": round(rendered.gen_s, 2),
                        "new_tokens": rendered.new_tokens,
                        "hit_token_cap": rendered.hit_token_cap,
                        "sha256_float32": common.sha256_array(rendered.audio),
                    }
                )
                print(mode, voice, segment, renders[-1]["audio_s"], "s")
    engine.unload()
    common.write_json(
        OUT / "renders.json",
        {"ran_at": common.now(), "model": {"repo": common.MODEL_BASE, "revision": revision}, "renders": renders},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
