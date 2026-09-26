"""Spike (d): the repeat test on the Base clone path. It sets the determinism tier (design sections 10.1, 20).

Question: with the determinism switches of section 10.1, does the same request (voice, text, seed) render the
same bytes again, **within one process** and **in a fresh process**? If yes, the tier is ``bit_exact``;
otherwise ``similar``.

Design of the test:

- Items: the allowlisted voices d2 and d4, three of the service's own paragraphs (``narration-en.v1``:
  ladder-080, ladder-150 and cal-02, about 5, 9 and 13 s of speech), fixed seeds.
- Each render goes through the worker's engine (``narration_qwen3tts.engine``), exactly as the worker runs
  it: the voice prompt made once per process, then seed, then ``generate_voice_clone`` with every
  audio-changing setting explicit. The float32 samples are hashed as generated.
- Three variants of the environment, each in its own processes, because ``CUBLAS_WORKSPACE_CONFIG`` can only
  be set before CUDA starts:
  - ``pinned``: the section 10.1 switches (TF32 off, cuDNN deterministic, benchmark off, deterministic
    algorithms warn-only, ``CUBLAS_WORKSPACE_CONFIG=:4096:8``), with the voice-prompt exemption the engine
    documents;
  - ``no_det_algos``: the same, but ``torch.use_deterministic_algorithms`` off;
  - ``defaults``: torch's defaults and no ``CUBLAS_WORKSPACE_CONFIG``, as the bakeoff ran.
- Within a process the items are rendered in one order, then again in reverse; fresh processes render them
  in other orders, and prepare the voices in another order, so that neither the order of work nor the
  process matters if the result is bit-exact.

Run from the checkout in the Qwen worker's venv, with the GPU lock held (about 15 minutes)::

    workers/qwen3tts/.venv/Scripts/python.exe spikes/d-repeat-test/run.py [--material <dir>]

Writes ``spikes/d-repeat-test/results.json`` and ``renders.csv`` (published) and the audio under
``.dev/spikes/d/``. ``--compare-only`` recomputes the summary from what a run saved there, without the GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_spike_common as common

HERE = Path(__file__).resolve().parent
OUT = common.DEV_SPIKES / "d"
DEVICE = "cuda:0"
ITEMS: list[dict[str, Any]] = [
    {"key": "d2/ladder-080/s11", "voice": "d2", "segment": "ladder-080", "seed": 11},
    {"key": "d2/ladder-150/s12", "voice": "d2", "segment": "ladder-150", "seed": 12},
    {"key": "d2/cal-02/s13", "voice": "d2", "segment": "cal-02", "seed": 13},
    {"key": "d4/ladder-080/s11", "voice": "d4", "segment": "ladder-080", "seed": 11},
]
ORDERS: dict[str, list[int]] = {"forward": [0, 1, 2, 3], "reverse": [3, 2, 1, 0], "shuffled": [2, 0, 3, 1]}
PLAN: list[dict[str, Any]] = [
    {"name": "pinned-p1", "variant": "pinned", "voices": ["d2", "d4"], "passes": ["forward", "reverse"]},
    {"name": "pinned-p2", "variant": "pinned", "voices": ["d4", "d2"], "passes": ["shuffled"]},
    {"name": "pinned-p3", "variant": "pinned", "voices": ["d2", "d4"], "passes": ["forward"]},
    {"name": "no_det_algos-p1", "variant": "no_det_algos", "voices": ["d2", "d4"], "passes": ["forward", "reverse"]},
    {"name": "no_det_algos-p2", "variant": "no_det_algos", "voices": ["d4", "d2"], "passes": ["shuffled"]},
    {"name": "defaults-p1", "variant": "defaults", "voices": ["d2", "d4"], "passes": ["forward", "reverse"]},
    {"name": "defaults-p2", "variant": "defaults", "voices": ["d4", "d2"], "passes": ["shuffled"]},
]


# ---------------------------------------------------------------------- one process
def child(name: str, variant: str, voices: list[str], passes: list[str], material: Path) -> int:
    common.prepare_process_env()
    if variant == "defaults":
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    import torch
    from narration_qwen3tts.engine import QwenEngine
    from narration_qwen3tts.settings import effective_generation

    common.cap_torch_threads(torch)
    if variant == "pinned":
        switches = common.apply_switches(torch, on=True)
    elif variant == "no_det_algos":
        switches = common.apply_switches(torch, on=True)
        torch.use_deterministic_algorithms(False)
        switches["deterministic_algorithms"] = "off"
    else:
        switches = {"torch_defaults": True, "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    texts = common.calibration_paragraphs(material)
    snapshot, _ = common.snapshot(common.MODEL_BASE)
    engine = QwenEngine(torch)
    engine.load(
        snapshot_dir=snapshot,
        device=DEVICE,
        dtype="bfloat16",
        attn_implementation="sdpa",
        settings={"non_streaming_mode": False, "generation": effective_generation(snapshot)},
    )
    for voice in voices:
        clip, transcript, _ = common.voice_clip(voice)
        engine.prepare_voice(voice, clip, transcript, x_vector_only_mode=False)
    renders: list[dict[str, Any]] = []
    for pass_name in passes:
        for index in ORDERS[pass_name]:
            item = ITEMS[index]
            common.seed_all(torch, item["seed"])
            with common.captured_warnings() as warned:
                rendered = engine.synthesize(item["voice"], texts[item["segment"]], common.LANGUAGE, common.CEILING)
            wav = OUT / name / f"{pass_name}-{item['key'].replace('/', '_')}.wav"
            common.write_wav(wav, rendered.audio, rendered.sample_rate)
            renders.append(
                {
                    "process": name,
                    "variant": variant,
                    "pass": pass_name,
                    "item": item["key"],
                    "sha256_float32": common.sha256_array(rendered.audio),
                    "samples": int(rendered.audio.size),
                    "new_tokens": rendered.new_tokens,
                    "hit_token_cap": rendered.hit_token_cap,
                    "gen_s": round(rendered.gen_s, 2),
                    "warnings": warned,
                }
            )
            print(f"{name} {pass_name} {item['key']}: {renders[-1]['sha256_float32'][:16]} {rendered.gen_s:.1f} s")
    engine.unload()
    common.write_json(OUT / f"{name}.json", {"process": name, "switches": switches, "renders": renders})
    return 0


# ---------------------------------------------------------------------- the parent: run, then compare
def compare(renders: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    def audio(render: dict[str, Any]) -> Any:
        path = OUT / render["process"] / f"{render['pass']}-{render['item'].replace('/', '_')}.wav"
        return sf.read(str(path), dtype="float32")[0]

    def diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        x, y = audio(a), audio(b)
        facts: dict[str, Any] = {"samples": [int(x.size), int(y.size)]}
        if x.size == y.size:
            delta = np.abs(x.astype(np.float64) - y.astype(np.float64))
            facts["max_abs_diff"] = float(delta.max())
            facts["first_differing_sample"] = int(np.argmax(delta > 0)) if delta.max() > 0 else None
        return facts

    summary: dict[str, Any] = {"by_variant": {}, "across_variants": {}}
    for variant in ("pinned", "no_det_algos", "defaults"):
        mine = [r for r in renders if r["variant"] == variant]
        by_item: dict[str, Any] = {}
        for item in (i["key"] for i in ITEMS):
            rows = [r for r in mine if r["item"] == item]
            processes = sorted({r["process"] for r in rows})
            in_process = all(len({r["sha256_float32"] for r in rows if r["process"] == p}) == 1 for p in processes)
            first_pass = [next(r for r in rows if r["process"] == p) for p in processes]
            across = len({r["sha256_float32"] for r in first_pass}) == 1
            entry: dict[str, Any] = {
                "renders": len(rows),
                "processes": len(processes),
                "identical_within_process": in_process,
                "identical_across_processes": across,
                "distinct_hashes": len({r["sha256_float32"] for r in rows}),
            }
            if not across:
                entry["first_vs_second_process"] = diff(first_pass[0], first_pass[1])
            if not in_process:
                p = processes[0]
                a, b = [r for r in rows if r["process"] == p][:2]
                entry["within_first_process"] = diff(a, b)
            by_item[item] = entry
        summary["by_variant"][variant] = {
            "bit_exact_within_process": all(e["identical_within_process"] for e in by_item.values()),
            "bit_exact_across_processes": all(e["identical_across_processes"] for e in by_item.values()),
            "items": by_item,
        }
    for item in (i["key"] for i in ITEMS):
        hashes = {
            variant: sorted({r["sha256_float32"] for r in renders if r["variant"] == variant and r["item"] == item})
            for variant in ("pinned", "no_det_algos", "defaults")
        }
        summary["across_variants"][item] = {
            "pinned_equals_no_det_algos": hashes["pinned"] == hashes["no_det_algos"],
            "pinned_equals_defaults": hashes["pinned"] == hashes["defaults"],
        }
    summary["warnings"] = warnings_summary(renders)
    pinned = summary["by_variant"]["pinned"]
    summary["tier"] = (
        "bit_exact" if pinned["bit_exact_within_process"] and pinned["bit_exact_across_processes"] else "similar"
    )
    return summary


def warnings_summary(renders: list[dict[str, Any]]) -> dict[str, Any]:
    """What Python warnings the renders raised, above all ``use_deterministic_algorithms(warn_only=True)``'s
    reports of a non-deterministic op (only the ``pinned`` variant turns them on)."""
    distinct = sorted({w for r in renders for w in r["warnings"]})
    return {
        "renders_checked": len(renders),
        "renders_checked_with_deterministic_algorithms_warn_only": sum(r["variant"] == "pinned" for r in renders),
        "renders_with_warnings": sum(bool(r["warnings"]) for r in renders),
        "distinct": distinct,
    }


def saved_renders() -> list[dict[str, Any]]:
    """Every render the child processes recorded, from their files under ``.dev/spikes/d/``."""
    renders: list[dict[str, Any]] = []
    for spec in PLAN:
        renders.extend(json.loads((OUT / f"{spec['name']}.json").read_text(encoding="utf-8"))["renders"])
    return renders


def compare_only() -> int:
    """Recompute ``results.json``'s summary from the saved renders, without the GPU; the rest is kept."""
    results = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    results["summary"] = compare(saved_renders())
    results["summary_recomputed_at"] = common.now()
    common.write_json(HERE / "results.json", results)
    print(json.dumps(results["summary"]["warnings"], indent=2))
    return 0


def parent(material_arg: str | None) -> int:
    material = common.material_root(material_arg)
    started = common.now()
    runs: list[dict[str, Any]] = []
    for spec in PLAN:
        t0 = time.perf_counter()
        argv = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            spec["name"],
            "--variant",
            spec["variant"],
            "--voices",
            ",".join(spec["voices"]),
            "--passes",
            ",".join(spec["passes"]),
            "--material",
            str(material),
        ]
        env = {k: v for k, v in os.environ.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
        code = subprocess.run(argv, env=env, check=False).returncode
        runs.append({**spec, "exit_code": code, "wall_s": round(time.perf_counter() - t0, 1)})
        if code != 0:
            print(f"{spec['name']} failed with exit code {code}", file=sys.stderr)
            return code
    renders = saved_renders()
    switches: dict[str, Any] = {}
    for spec in PLAN:
        switches[spec["variant"]] = json.loads((OUT / f"{spec['name']}.json").read_text(encoding="utf-8"))["switches"]
    import torch

    results = {
        "spike": "d (repeat test, Base clone path)",
        "design_sections": ["10.1", "10.3", "20 (d)"],
        "ran_at": started,
        "software": common.software_facts(torch),
        "items": ITEMS,
        "orders": ORDERS,
        "plan": runs,
        "switches": switches,
        "voice_prompt_exemption": "deterministic algorithms are off during create_voice_clone_prompt only "
        "(narration_qwen3tts.engine; spike h+i found the Mimi encoder fails under them)",
        "summary": compare(renders),
    }
    common.write_json(HERE / "results.json", results)
    with (HERE / "renders.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["process", "variant", "pass", "item", "sha256_float32", "samples", "new_tokens"]
        fields += ["hit_token_cap", "gen_s"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(renders)
    print(json.dumps(results["summary"], indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--material", help="the service's material folder (default: <checkout>/material)")
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="recompute results.json's summary from the renders saved under .dev/spikes/d/ (no GPU)",
    )
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--variant", help=argparse.SUPPRESS)
    parser.add_argument("--voices", help=argparse.SUPPRESS)
    parser.add_argument("--passes", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.compare_only:
        return compare_only()
    if args.child:
        material = common.material_root(args.material)
        return child(args.child, args.variant, args.voices.split(","), args.passes.split(","), material)
    return parent(args.material)


if __name__ == "__main__":
    sys.exit(main())
