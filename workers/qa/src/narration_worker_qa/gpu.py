"""The QA worker's manners on a shared GPU (design section 4, the GPU scheduler; plan.md WP22).

PyTorch's caching allocator keeps the blocks an op freed, to reuse them. Between Whisper's and WavLM's ops of
different shapes it cannot always reuse them, so without a release the process's footprint on the device grows
past what any one op needs. KNOW (``spikes/h-i-qa-load``, 2026-09-27): a whole take transcribed and then embedded
left 14.2 GB reserved for 5.6 GB allocated. The scheduler admits the QA group on its declared need
(``vram_need_mb``), and the GPU is shared, so after each GPU op the worker returns the cached blocks to the
device: the footprint between ops is then the resident models', and during an op that op's own peak.
"""

from __future__ import annotations

from typing import Any


def release_cached_memory(torch: Any, device: str) -> None:
    """After an op on ``device``: return the CUDA allocator's cached, unused blocks to the device. A no-op for the
    CPU. It changes no result: the kernels are deterministic whatever addresses they are given."""
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
