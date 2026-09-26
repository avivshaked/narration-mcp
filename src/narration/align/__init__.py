"""Cue alignment, the pure part (design section 11.2; plan.md WP15).

``CtcAligner`` implements ``narration.contracts.interfaces.AlignerCore``: the aligner's transcript and its
token → (cue, word) map, the ``T ≥ L + R`` guard, snapping boundaries into pauses, confidence, unplaceable
cues (null, never interpolated), and the Whisper cross-check. The model runs in the QA worker
(``narration_worker_qa.align``, plan.md P1); this package never loads one.
"""

from __future__ import annotations

from .alphabet import ALPHABET, WILDCARD, WORD_SEPARATOR, fold_letter, spell
from .core import AlignerParams, CtcAligner
from .crosscheck import Boundary, BoundaryCheck, cross_check, interval_distance, plain
from .snap import Pause, PauseParams, find_pauses, frame_levels_db
from .transcript import build_transcript, ctc_frames, guard, has_letters, repeats, wildcard_runs

__all__ = [
    "ALPHABET",
    "WILDCARD",
    "WORD_SEPARATOR",
    "AlignerParams",
    "Boundary",
    "BoundaryCheck",
    "CtcAligner",
    "Pause",
    "PauseParams",
    "build_transcript",
    "cross_check",
    "ctc_frames",
    "find_pauses",
    "fold_letter",
    "frame_levels_db",
    "guard",
    "has_letters",
    "interval_distance",
    "plain",
    "repeats",
    "spell",
    "wildcard_runs",
]
