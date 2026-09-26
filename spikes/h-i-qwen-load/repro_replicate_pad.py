"""Minimal reproduction: deterministic mode breaks CUDA replicate padding given a tensor-valued pad.

Found by spike (h)+(i) on 2026-09-26: with ``torch.use_deterministic_algorithms(True, warn_only=True)`` (design
section 10.1), ``create_voice_clone_prompt`` failed inside the Mimi encoder of Qwen's speech tokenizer with
``RuntimeError: _unsafe_index found unexpected index type Float``. This script shows the cause without a
model: in deterministic mode torch routes CUDA ``replicate`` padding through
``torch._decomp.decompositions._replication_pad``, whose ``pw_cast_for_opmath`` wrapper casts every tensor
argument to the computation dtype, including a padding amount passed as a 0-dim integer tensor, as Mimi's
``MimiConv1d`` does (``extra_padding``).

Run in the Qwen worker's venv (it needs CUDA, and a few hundred MB of VRAM for the context)::

    workers/qwen3tts/.venv/Scripts/python.exe spikes/h-i-qwen-load/repro_replicate_pad.py

Writes ``spikes/h-i-qwen-load/repro_replicate_pad.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_spike_common as common

HERE = Path(__file__).resolve().parent


def attempt(torch: Any, *, deterministic: bool, dtype: str, tensor_pad: bool) -> dict[str, object]:
    import torch.nn.functional as functional

    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    x = torch.randn(1, 4, 10, device="cuda", dtype=getattr(torch, dtype))
    right: Any = torch.tensor(1) if tensor_pad else 1
    case = {"deterministic_algorithms": deterministic, "dtype": dtype, "right_pad_is_tensor": tensor_pad}
    try:
        out = functional.pad(x, (3, right), mode="replicate")
    except RuntimeError as exc:
        return {**case, "ok": False, "error": str(exc).splitlines()[0]}
    reference = functional.pad(x.float().cpu(), (3, 1), mode="replicate")
    return {**case, "ok": True, "matches_cpu": bool(torch.equal(out.float().cpu(), reference))}


def main() -> int:
    common.prepare_process_env()
    import torch

    torch.manual_seed(0)
    cases = [
        attempt(torch, deterministic=deterministic, dtype=dtype, tensor_pad=tensor_pad)
        for deterministic in (True, False)
        for dtype in ("float32", "bfloat16")
        for tensor_pad in (True, False)
    ]
    torch.use_deterministic_algorithms(False)
    result = {
        "script": "spikes/h-i-qwen-load/repro_replicate_pad.py",
        "ran_at": common.now(),
        "software": common.software_facts(torch),
        "cases": cases,
        "conclusion": "fails exactly when deterministic algorithms are on and the pad amount is a tensor"
        if all(c["ok"] != (c["deterministic_algorithms"] and c["right_pad_is_tensor"]) for c in cases)
        else "see cases",
    }
    common.write_json(HERE / "repro_replicate_pad.json", result)
    for case in cases:
        print(case)
    return 0


if __name__ == "__main__":
    sys.exit(main())
