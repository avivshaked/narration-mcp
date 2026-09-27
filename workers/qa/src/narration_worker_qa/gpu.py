"""The QA worker's manners on a shared GPU (design section 4, the GPU scheduler; plan.md WP22).

PyTorch's caching allocator keeps the blocks an op freed, to reuse them. Between Whisper's and WavLM's ops of
different shapes it cannot always reuse them, so without a release the process's footprint on the device grows
past what any one op needs. KNOW (``spikes/h-i-qa-load``, the run of 2026-09-27 13:29 UTC, before the release;
its ``results.json`` is in commit ``05114c0``): a whole take transcribed and then embedded left 14.2 GB reserved
for 5.6 GB allocated. The scheduler admits the QA group on its declared need (``vram_need_mb``), and the GPU is
shared, so after each GPU op the worker returns the cached blocks to the device: the footprint between ops is then
the resident models', and during an op that op's own peak.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def release_cached_memory(torch: Any, device: str) -> None:
    """After an op on ``device``: return the CUDA allocator's cached, unused blocks to the device. A no-op for the
    CPU. It changes no result: the kernels are deterministic whatever addresses they are given.

    It never raises. It runs in the ``finally`` of every GPU op, so an exception here would replace the op's own
    (a ``GPU_OOM`` would be reported as ``INTERNAL``, and the daemon would not unload and retry); a failure to
    release is logged to stderr instead, and the next op's release tries again.
    """
    if not device.startswith("cuda"):
        return
    try:
        torch.cuda.empty_cache()
    except Exception:  # never mask the op's own outcome
        log.warning("could not return the CUDA allocator's cached blocks after an op on %s", device, exc_info=True)
