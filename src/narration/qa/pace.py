"""Pace (design sections 3.2 and 11.1 step 9; plan.md WP47, the owner's decision D2): one rule for the voice's
measurement and for QA, ``names.PACE_METHOD``.

**Pace is spoken characters per second of speaking time** (an articulation rate):

- **spoken characters**: the segment's ``spoken_chars`` (section 7.2), the characters of the text as it is
  spoken, which is what the ladder's rungs are named by;
- **the voiced span**: the delivery file from its first speech frame to its last (``SignalStats``, the trim's
  speech rule, DC-10);
- **speaking time**: the voiced span less every **pause** inside it, a run of non-speech frames of at least
  ``MIN_PAUSE_S`` (``SignalStats.internal_silences_s``). Shorter silences stay in: they are part of speaking,
  such as the closure before a stop consonant.

Why not words per minute: the words of the service's ladder paragraphs differ in length from rung to rung, so
the voice's words per minute followed the corpus's word lengths while its characters per second stayed flat
(KNOW, the first real voice: 150-208 wpm, 15.4-17.2 characters per second over the voiced span). Why not
characters per second over the whole voiced span: the span holds the pauses between sentences, so a
one-sentence segment, which has none, read as fast against a curve measured on paragraphs of several sentences
(KNOW, the owner's first job: single sentences at 16.6-19.9 characters per second against rungs at
16.2-16.8; the owner listened and they were not fast).

The ladder's takes are scored by the same ``Scorer`` as every other take, so the measurement's curve and a
take's pace are the same quantity by construction (``narration.measure.ladder`` reads
``QaMetrics.articulation_cps``).

**When the silences were not measured** (``internal_silences_s`` None: a ``SignalStats`` built without them),
the rate is over the whole voiced span, and ``pause_s`` is None to say so. The service's own signal stage
always measures them. With no voiced span there is no pace.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Final

from narration.contracts.interfaces import SignalStats
from narration.contracts.models import Pace

__all__ = [
    "MIN_PAUSE_S",
    "TakeRate",
    "estimated_seconds",
    "expected_cps",
    "expected_wpm_at",
    "pause_seconds",
    "take_rate",
]

MIN_PAUSE_S: Final = 0.25
"""The shortest silence inside the voiced span that counts as a pause (seconds). At the signal stage's 20 ms
frames, the shortest pause is 13 frames (0.26 s); 12 frames (0.24 s) stay in the speaking time.

Where the value comes from. BELIEVE: 250 ms is the usual lower bound for a silent pause in speech-rate
research, after Goldman-Eisler; shorter silences inside running speech are mostly articulation, such as a stop
consonant's closure. KNOW (``spikes/b-forced-align-cpu/calibration.json``, ``pauses.between_sentences``): on
the bake-off's six clone takes, 65 of the 66 gaps between two sentences held silence under a rule like the
signal stage's (20 ms frames 40 dB under the take's 95th percentile), so the pauses this removes are there to
find; how long they are was not measured. ASSUME until the rate is measured on the service's own takes (the
first voice measured again after WP47): that 0.25 s separates the pauses from the silences of speaking. A
change to it is a change to ``names.PACE_METHOD``."""


def pause_seconds(signal: SignalStats) -> float | None:
    """The time taken out of the voiced span: the sum of the silences inside it of at least ``MIN_PAUSE_S``.
    None when the silences were not measured."""
    if signal.internal_silences_s is None:
        return None
    return float(sum(s for s in signal.internal_silences_s if s >= MIN_PAUSE_S))


@dataclass(frozen=True, slots=True, kw_only=True)
class TakeRate:
    """A take's pace, and what it is made of. Every value is None when the take has no voiced span.

    ``articulation_cps`` is the pace (spoken characters per second of speaking time); ``spoken_cps`` and
    ``spoken_wpm`` are over the whole voiced span, pauses included, for information. ``pause_s`` is None when
    the silences were not measured (the rate is then over the whole voiced span).
    """

    voiced_s: float | None
    pause_s: float | None
    speaking_s: float | None
    articulation_cps: float | None
    spoken_cps: float | None
    spoken_wpm: float | None


def take_rate(spoken_chars: int, spoken_words: int, signal: SignalStats) -> TakeRate:
    """The take's pace from its segment's spoken characters and words and the delivery file's signal facts."""
    start, end = signal.voiced_start_s, signal.voiced_end_s
    voiced = end - start if start is not None and end is not None else None
    if voiced is None or voiced <= 0:
        return TakeRate(
            voiced_s=None, pause_s=None, speaking_s=None, articulation_cps=None, spoken_cps=None, spoken_wpm=None
        )
    pause = pause_seconds(signal)
    speaking = voiced - (pause or 0.0)
    return TakeRate(
        voiced_s=voiced,
        pause_s=pause,
        speaking_s=speaking,
        articulation_cps=spoken_chars / speaking if speaking > 0 else None,
        spoken_cps=spoken_chars / voiced,
        spoken_wpm=spoken_words / voiced * 60.0,
    )


def expected_cps(pace: Pace, spoken_chars: int) -> float | None:
    """The voice's pace curve at this spoken length, in spoken characters per second of speaking time.

    Inside the curve's range: straight lines between its points (points at one length are averaged). Outside
    it: the nearest end point's value, held flat (DC-20: a voice's pace does not follow length, so no slope is
    extended). With no curve (no rung passed): the voice's level, ``Pace.level_cps``. The trend is never read.
    None when the result is not a positive pace.
    """
    by_chars: dict[int, list[float]] = {}
    for point in pace.curve:
        by_chars.setdefault(point.chars, []).append(point.cps)
    points = sorted((c, sum(v) / len(v)) for c, v in by_chars.items())
    if not points:
        value = pace.level_cps
    elif spoken_chars <= points[0][0]:
        value = points[0][1]
    elif spoken_chars >= points[-1][0]:
        value = points[-1][1]
    else:
        value = points[0][1]
        for (c0, v0), (c1, v1) in pairwise(points):
            if c0 <= spoken_chars <= c1:
                value = v0 + (v1 - v0) * (spoken_chars - c0) / (c1 - c0)
                break
    return value if value > 0 else None


def expected_wpm_at(rate: TakeRate, spoken_words: int, expected: float | None, spoken_chars: int) -> float | None:
    """The take's own spoken words per minute at the expected pace: its speaking time at ``expected`` characters
    per second, its pauses as they are. Information only (``QaMetrics.expected_spoken_wpm``)."""
    if expected is None or expected <= 0 or rate.voiced_s is None:
        return None
    span = spoken_chars / expected + (rate.pause_s or 0.0)
    return spoken_words / span * 60.0 if span > 0 else None


def estimated_seconds(pace: Pace, spoken_chars: int) -> float | None:
    """How long a take of this spoken length should last, pauses included: its characters at the curve's pace,
    over the voice's ``speaking_share``. For progress and estimates only (sections 7.3 and 12), never a verdict.
    None when the curve gives no positive pace."""
    expected = expected_cps(pace, spoken_chars)
    if expected is None:
        return None
    share = pace.speaking_share if 0.0 < pace.speaking_share <= 1.0 else 1.0
    return spoken_chars / expected / share
