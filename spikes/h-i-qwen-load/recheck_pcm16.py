"""Re-check spike (h)+(i)'s d2 re-design against the bakeoff's clip, with the comparison fixed.

``run.py`` first compared the two by reading its own FLOAT WAV with ``dtype="int16"``. libsndfile does not
scale float to integer on read, so that returned only -1, 0 and 1, and ``results.json`` reports a false
mismatch (``pcm16_max_abs_diff`` 23935, which is the clip's peak). ``run.py`` is fixed. This script redoes
the comparison from the renders that run saved, without the GPU. It converts each float render to 16 bits
the way the bakeoff wrote its files (``gpu_spike_common.as_bakeoff_pcm16``) and compares every sample.

Run from the checkout in the Qwen worker's venv (no GPU, no lock)::

    workers/qwen3tts/.venv/Scripts/python.exe spikes/h-i-qwen-load/recheck_pcm16.py

Reads ``.dev/spikes/h-i/design-d2-redesign-*.wav`` and the bakeoff's d2 clip. Writes
``spikes/h-i-qwen-load/recheck_pcm16.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_spike_common as common

HERE = Path(__file__).resolve().parent
OUT = common.DEV_SPIKES / "h-i"


def main() -> int:
    import numpy as np
    import soundfile as sf

    clip, _, sidecar = common.voice_clip("d2")
    reference = sf.read(str(clip), dtype="int16")[0]
    results: dict[str, Any] = {
        "recheck_of": "results.json models.voice_design.d2_redesign_vs_bakeoff (pcm16_* fields)",
        "ran_at": common.now(),
        "seed": sidecar["seed"],
        "reference_samples": int(reference.size),
        "method": "float render -> soundfile.write default WAV subtype (PCM_16) in memory -> int16, "
        "compared sample for sample with the bakeoff clip",
    }
    identical = True
    for label in ("switches_off", "switches_on"):
        wav = OUT / f"design-d2-redesign-{label}.wav"
        audio, rate = sf.read(str(wav), dtype="float32")
        pcm = common.as_bakeoff_pcm16(audio, rate)
        same_length = pcm.size == reference.size
        facts: dict[str, Any] = {
            "sha256_float32": common.sha256_array(audio),
            "pcm16_samples": int(pcm.size),
            "pcm16_bit_identical_to_bakeoff_clip": bool(same_length and np.array_equal(pcm, reference)),
        }
        if same_length:
            facts["pcm16_max_abs_diff"] = int(np.max(np.abs(pcm.astype(np.int32) - reference.astype(np.int32))))
        identical = identical and facts["pcm16_bit_identical_to_bakeoff_clip"]
        results[label] = facts
    common.write_json(HERE / "recheck_pcm16.json", results)
    print("bit-identical to the bakeoff clip:", identical)
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
