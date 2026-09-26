"""Delivery post-processing (design section 13; WP13): raw take → trimmed, 48 kHz, -16 LUFS, PCM_24 file.

``DeliveryPipeline`` is the ``DeliveryProcessor`` of ``narration.contracts.interfaces``. ``deliver`` runs the
same pipeline in memory. The steps, the pinned tools and the rules for degenerate takes are described in
``narration.post.pipeline``.
"""

from __future__ import annotations

from .pipeline import (
    GAIN_HIGH_DB,
    Delivered,
    DeliveryPipeline,
    deliver,
    delivery_tools,
    fade_edges,
    measure_signal,
)

__all__ = [
    "GAIN_HIGH_DB",
    "Delivered",
    "DeliveryPipeline",
    "deliver",
    "delivery_tools",
    "fade_edges",
    "measure_signal",
]
