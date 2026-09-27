"""A measurement's renders keep their keys and seeds when the pace rule changes (design sections 10.2, 10.3;
plan.md WP47).

WP47 moved the pace model from words per minute to characters per second of speaking time. That changes the
measurement key, the QA profile and so every ladder take's analysis key, which is intended: a voice measured
before is measured again. It must not change what is rendered: the calibration set's and the ladder's engine
texts, seeds and render keys. Then measuring again finds every render and take in the cache and costs QA time
only.

The values pinned here were computed by the handler's own planning **before** WP47's change (``main`` at
2a069e0), for the service's frozen corpus, a fixed clip and the job-engine tests' engine profile. No text is
written here: the engine texts come from ``material/``, and the pins are hashes.
"""

from __future__ import annotations

import hashlib
import io
import time
import wave
from pathlib import Path
from typing import Final

from narration.contracts.models import ProvenanceEntry
from narration.measure.corpus import default_material_root
from narration.store.store import utc_iso
from tests.jobs.support import DESIGN_ID

from .support import corpus_data, make_world, measure_request, submit_measure

FULL_LADDER: Final = (80, 150, 250, 300, 350, 400, 450, 500, 560)
SEEDS: Final = 3

PINNED_LADDER_080_ATTEMPT_0: Final = (
    1231823624,
    "sha256:45cd3911a57871e4b54a5bda9e02e2c410abe3b4eccc3ada42debf3267be596a",
)
"""(seed, render_key) of ``ladder-080``'s attempt 0."""
PINNED_PLAN_SHA256: Final = "c8145f1abe8aebcff6a08b52aca14f06c863f975f06e14a10c5e94e32c2d0e9c"
"""sha256 of every planned attempt's ``segment_id attempt seed render_key``, one line each, in plan order."""


def fixed_clip(path: Path) -> str:
    """A clip whose bytes are the same on every platform (the standard library's WAV writer, silence), so the
    voice hash, and with it every seed and render key, is fixed. Returns its sha256."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(24_000)
        out.writeframes(b"\x00\x00" * 2_400)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def planned_attempts(root: Path) -> list[tuple[str, int, int, str]]:
    """(segment_id, attempt, seed, render_key) of every attempt a full measurement plans, in plan order."""
    world = make_world(
        root, ladder=FULL_LADDER, seeds=SEEDS, corpus=corpus_data(), material=default_material_root()
    )
    try:
        clip = root / "pinned" / "clip.wav"
        sha = fixed_clip(clip)
        world.store.add_provenance(ProvenanceEntry(clip_sha256=sha, design_id=DESIGN_ID, date=utc_iso(time.time())))
        job = submit_measure(world.store, measure_request(clip, sha))
        run = world.handler.open(world.host, job)
        return [
            (seg.segment_id, attempt.attempt, attempt.seed, attempt.render_key)
            for seg in run.segments
            for slot in seg.slots
            for attempt in slot
        ]
    finally:
        world.close()


def test_measurement_render_keys_and_seeds_are_unchanged_by_the_pace_rule_s10_2_s10_3(tmp_path: Path) -> None:
    planned = planned_attempts(tmp_path)
    lines = "".join(f"{segment_id} {attempt} {seed} {key}\n" for segment_id, attempt, seed, key in planned)
    first = next((seed, key) for segment_id, attempt, seed, key in planned if segment_id == "ladder-080" and not attempt)
    assert first == PINNED_LADDER_080_ATTEMPT_0
    assert hashlib.sha256(lines.encode("utf-8")).hexdigest() == PINNED_PLAN_SHA256
    assert len(planned) == 13 * SEEDS  # the design text, three calibration paragraphs and nine rungs
