"""Are WavLM-base-plus-sv's two snapshots the same weights? (WP22: which revision to pin.)

The models root holds ``main`` (``feb593a6…``: ``config.json``, ``preprocessor_config.json`` and
``pytorch_model.bin``) and ``refs/pr/8`` (``1bfd64ec…``: ``model.safetensors`` only, the Hub's conversion). This
loads both on the CPU, the ``.bin`` with ``weights_only=True``, and compares every tensor bit for bit.

Run from the checkout in the QA worker's venv (no GPU)::

    workers/qa/.venv/Scripts/python.exe spikes/h-i-qa-load/wavlm_weights.py

Writes ``spikes/h-i-qa-load/wavlm_weights.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_spike_common as common

HERE = Path(__file__).resolve().parent
MAIN = "feb593a6"
PR8 = "1bfd64ec"


def main() -> int:
    common.prepare_process_env()
    import torch
    from safetensors.torch import load_file

    main_dir, main_revision = common.snapshot(common.MODEL_SV, MAIN)
    pr8_dir, pr8_revision = common.snapshot(common.MODEL_SV, PR8)
    pickled = torch.load(str(main_dir / "pytorch_model.bin"), map_location="cpu", weights_only=True)
    converted = load_file(str(pr8_dir / "model.safetensors"))
    common_keys = sorted(set(pickled) & set(converted))
    differing = [
        k
        for k in common_keys
        if pickled[k].dtype != converted[k].dtype
        or pickled[k].shape != converted[k].shape
        or not torch.equal(pickled[k], converted[k])
    ]
    results = {
        "check": "WavLM-base-plus-sv: main's pytorch_model.bin against refs/pr/8's model.safetensors",
        "ran_at": common.now(),
        "main": {
            "revision": main_revision,
            "files": sorted(p.name for p in main_dir.iterdir()),
            "tensors": len(pickled),
        },
        "refs_pr_8": {
            "revision": pr8_revision,
            "files": sorted(p.name for p in pr8_dir.iterdir()),
            "tensors": len(converted),
        },
        "only_in_main": sorted(set(pickled) - set(converted)),
        "only_in_refs_pr_8": sorted(set(converted) - set(pickled)),
        "differing_tensors": differing,
        "dtypes": sorted({str(v.dtype) for v in pickled.values()}),
        "identical": not differing and set(pickled) == set(converted),
    }
    common.write_json(HERE / "wavlm_weights.json", results)
    print(f"identical: {results['identical']}")
    return 0 if results["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
