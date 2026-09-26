"""Evidence for DC-4 (plan.md section 1.5): does ``max_new_tokens`` change a render that does not reach it?

plan.md section 1.3 item 1: the effective cap is 8192 (the snapshots' ``generation_config.json``), and every
piece of clone evidence ran with it. If the cap only ever truncates, a lower cap, or one derived from the
text's length, leaves every render that stays under it bit-identical, and the evidence stays valid.

What it does, in one process with the section 10.1 switches (the ``bit_exact`` tier, spike d):

- renders two of the service's paragraphs (``narration-en.v1`` ladder-150 and ladder-350) in the d2 voice
  with ``max_new_tokens`` 8192, and records the talker's steps *N* (the frames decoded are *N* − 1);
- renders each again, with the same seed, at 2048 and at *N* (the smallest cap that lets the talker emit
  its end token), which must be bit-identical if the cap only truncates;
- and at *N* − 1, which must stop one step short: ``hit_token_cap`` true;
- records frames per spoken character for every render, the basis of a length-derived cap.

Each cap is the call's own ``max_new_tokens`` (DC-4, as the worker takes it), under the loaded ceiling of 8192.
The run of 2026-09-26 set each cap as the ``load``'s ceiling instead, before the engine took a per-call cap;
both reach qwen-tts's ``generate`` as the same ``max_new_tokens``.

Run from the checkout in the Qwen worker's venv, with the GPU lock held (about 5 minutes)::

    workers/qwen3tts/.venv/Scripts/python.exe spikes/dc4-max-new-tokens/run.py [--material <dir>]

Writes ``spikes/dc4-max-new-tokens/results.json`` and the audio under ``.dev/spikes/dc4/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_spike_common as common

HERE = Path(__file__).resolve().parent
OUT = common.DEV_SPIKES / "dc4"
DEVICE = "cuda:0"
ITEMS = [("ladder-150", 12), ("ladder-350", 21)]


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--material", help="the service's material folder (default: <checkout>/material)")
    args = parser.parse_args()
    common.prepare_process_env()
    texts = common.calibration_paragraphs(common.material_root(args.material))

    import numpy as np
    import torch
    from narration_qwen3tts.engine import QwenEngine
    from narration_qwen3tts.settings import effective_generation

    common.cap_torch_threads(torch)
    switches = common.apply_switches(torch, on=True)
    snapshot, revision = common.snapshot(common.MODEL_BASE)
    pinned = effective_generation(snapshot)
    ceiling = int(pinned.get("max_new_tokens", common.CEILING))
    engine = QwenEngine(torch)

    engine.load(
        snapshot_dir=snapshot,
        device=DEVICE,
        dtype="bfloat16",
        attn_implementation="sdpa",
        settings={"non_streaming_mode": False, "generation": pinned},
    )
    clip, transcript, _ = common.voice_clip("d2")
    engine.prepare_voice("d2", clip, transcript, x_vector_only_mode=False)

    def render(segment: str, seed: int, cap: int) -> tuple[dict[str, Any], Any]:
        common.seed_all(torch, seed)
        rendered = engine.synthesize("d2", texts[segment], common.LANGUAGE, cap)
        common.write_wav(OUT / f"{segment}-cap{cap}.wav", rendered.audio, rendered.sample_rate)
        return (
            {
                "max_new_tokens": cap,
                "talker_steps": rendered.talker_steps,
                "new_tokens": rendered.new_tokens,
                "hit_token_cap": rendered.hit_token_cap,
                "samples": int(rendered.audio.size),
                "sha256_float32": common.sha256_array(rendered.audio),
                "gen_s": round(rendered.gen_s, 2),
            },
            rendered.audio,
        )

    items: list[dict[str, Any]] = []
    for segment, seed in ITEMS:
        full, full_audio = render(segment, seed, ceiling)
        steps = full["talker_steps"]
        runs = [full]
        for cap in (2048, steps, steps - 1):
            facts, audio = render(segment, seed, cap)
            facts["identical_to_8192"] = facts["sha256_float32"] == full["sha256_float32"]
            if facts["hit_token_cap"]:
                n = min(audio.size, full_audio.size)
                same = np.flatnonzero(audio[:n] != full_audio[:n])
                facts["identical_prefix_samples"] = int(same[0]) if same.size else int(n)
            runs.append(facts)
        spoken = len(texts[segment])
        items.append(
            {
                "segment": segment,
                "seed": seed,
                "spoken_chars": spoken,
                "frames_per_spoken_char": round(full["new_tokens"] / spoken, 4),
                "seconds_per_spoken_char": round(full["samples"] / 24000 / spoken, 4),
                "runs": runs,
            }
        )
        print(segment, [(r["max_new_tokens"], r["hit_token_cap"], r.get("identical_to_8192")) for r in runs])
    engine.unload()
    results = {
        "evidence_for": "DC-4 (plan.md 1.5): max_new_tokens",
        "ran_at": common.now(),
        "software": common.software_facts(torch),
        "model": {"repo": common.MODEL_BASE, "revision": revision},
        "switches": switches,
        "items": items,
        "cap_only_truncates": all(
            r["identical_to_8192"] for item in items for r in item["runs"][1:] if not r["hit_token_cap"]
        )
        and all(r["hit_token_cap"] for item in items for r in item["runs"][3:]),
    }
    common.write_json(HERE / "results.json", results)
    return 0 if results["cap_only_truncates"] else 1


if __name__ == "__main__":
    sys.exit(main())
