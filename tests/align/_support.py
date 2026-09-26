"""Builders for the alignment tests: segments through the real text pipeline, synthetic audio, and replies
shaped like the worker's (``narration_worker_qa.align``), all deterministic.

The audio is a 200 Hz tone where "speech" is and exact zeros where a pause is, at 48 kHz, the delivery
rate. A reply places each word's letters evenly over the frames the test gives it, and each ``|`` on the
frame after the word before it, as ``merge_tokens`` would report spans.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from narration.contracts.interfaces import AlignTranscript
from narration.contracts.models import CueIn, Hint, SegmentIn, SegmentText
from narration.contracts.names import MODEL_ALIGNER
from narration.contracts.worker import AlignReply, AsrWord, TokenSpan
from narration.text import TextPipeline

RATE = 48_000
FRAME_S = 0.02
REVISION = "54074b1c16f4de6a5ad59affb4caa8f2ea03a119"
"""The pinned revision of the aligner (spike (b)); the tests only need it to match the reply's."""


def segment(*cues: str, hints: Sequence[Hint] = ()) -> SegmentText:
    """A segment of ``cues`` through the service's text pipeline (section 9.1)."""
    return TextPipeline().plan_segment(SegmentIn(segment_id="s1", cues=tuple(CueIn(text=c) for c in cues)), hints)


def audio(duration_s: float, speech: Sequence[tuple[float, float]]) -> npt.NDArray[np.float32]:
    """``duration_s`` of silence (exact zeros) with a 200 Hz tone, amplitude 0.3, in each speech interval."""
    n = round(duration_s * RATE)
    out = np.zeros(n, dtype=np.float32)
    t = np.arange(n, dtype=np.float64) / RATE
    for start, end in speech:
        a, b = round(start * RATE), round(end * RATE)
        out[a:b] = (0.3 * np.sin(2 * np.pi * 200.0 * t[a:b])).astype(np.float32)
    return out


def num_frames(samples: int) -> int:
    n16 = -(-samples // 3)
    return 0 if n16 < 400 else (n16 - 400) // 320 + 1


def reply(
    transcript: AlignTranscript,
    word_times: Mapping[tuple[int, int], tuple[float, float]],
    *,
    samples: int,
    scores: Mapping[int, float] | None = None,
    default_score: float = 0.95,
    model: str = MODEL_ALIGNER,
    revision: str = REVISION,
) -> AlignReply:
    """A reply whose words sit at ``word_times`` (seconds, per (cue, word)); ``scores`` per cue index."""
    scores = scores or {}
    owners = transcript.token_words
    groups: list[tuple[tuple[int, int] | None, list[int]]] = []
    for i, owner in enumerate(owners):
        key = owner
        if key is None:  # a separator inside a word (a hyphen) belongs to that word's group
            before = owners[i - 1] if i else None
            key = before if before is not None and before == _next_owner(owners, i) else None
        if key is not None and groups and groups[-1][0] == key:
            groups[-1][1].append(i)
        else:
            groups.append((key, [i]))
    spans: list[TokenSpan] = []
    cursor = 0
    for key, indices in groups:
        if key is None:  # the separator between two words: the frame after the word before
            spans.append(_span(indices[0], cursor, cursor + 1, default_score))
            cursor += 1
            continue
        start_s, end_s = word_times[key]
        first, last = round(start_s / FRAME_S), round(end_s / FRAME_S)
        count = len(indices)
        assert last - first >= count, f"{key}: {count} tokens need {count} frames, have {last - first}"
        assert first >= cursor, f"{key} starts before the previous token ends"
        score = scores.get(key[0], default_score)
        for k, index in enumerate(indices):
            a = first + (last - first) * k // count
            b = first + (last - first) * (k + 1) // count
            spans.append(_span(index, a, b, score))
        cursor = last
    return {
        "id": 1,
        "ok": True,
        "frame_s": FRAME_S,
        "num_frames": num_frames(samples),
        "spans": spans,
        "model": model,
        "revision": revision,
        "device": "cpu",
    }


def _next_owner(owners: Sequence[tuple[int, int] | None], i: int) -> tuple[int, int] | None:
    for owner in owners[i + 1 :]:
        if owner is not None:
            return owner
    return None


def _span(index: int, start: int, end: int, score: float) -> TokenSpan:
    return {"token_index": index, "start_frame": start, "end_frame": end, "score": score}


def asr(*words: tuple[str, float | None, float | None]) -> list[AsrWord]:
    """Whisper words as the QA worker reports them."""
    return [{"text": text, "start_s": start, "end_s": end} for text, start, end in words]


def cue_times(alignment: Any) -> list[tuple[float | None, float | None]]:
    return [(c.start_s, c.end_s) for c in alignment.cues]


def codes_of(alignment: Any) -> list[tuple[str, int | None]]:
    return [(f.code, f.cue) for f in alignment.flags]
