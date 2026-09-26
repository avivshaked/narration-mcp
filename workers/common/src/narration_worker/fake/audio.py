"""The fake worker's synthetic speech, and how its "ears" recognise it again.

**Speech.** Each word of the text is a burst of tone, separated from the next by silence: 0.1 s after a
word, 0.25 s after ``,`` ``;`` ``:`` or a dash, 0.4 s after ``.`` ``!`` or ``?``. A burst is made of 50 ms
segments, two more than the word has letters and never fewer than four, so a word of five letters lasts
0.35 s: about the pace of narration. The file starts with 0.12 s of silence and ends with 0.2 s.

**The id in the pitch.** Every segment is a tone at one of 256 pitches, 120 + 1.5·k Hz. Segment ``j`` of the
utterance (counted across bursts) carries byte ``j mod 4`` of a 32-bit utterance id. Resampling, a static
gain, trimming silence, fades at the ends and 24-bit quantisation (the delivery post-processing of section
13) leave pitch and the burst pattern alone, so ``decode`` recovers the id from a delivery file, and the
fake's QA ops look the utterance up in its registry (``registry.py``) by that id.

**Bit-identical everywhere.** The waveform is integer arithmetic only: a 32-bit phase accumulator, a
parabolic approximation of a sine from a table of integers, integer ramps. Every sample is an integer
divided by 32768, which float32 holds exactly, so the same request gives the same bytes in every process
and on every platform.
"""

from __future__ import annotations

from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from operator import mul
from typing import Final

from .wav import Audio

SAMPLE_RATE: Final = 24_000
"""Qwen3-TTS's output rate."""
SEGMENT_S: Final = 0.05
SEGMENT: Final = 1200
ID_BYTES: Final = 4
MIN_SEGMENTS: Final = ID_BYTES
BASE_HZ: Final = 120.0
STEP_HZ: Final = 1.5
SYMBOLS: Final = 256
RAMP: Final = 120
"""A 5 ms linear attack and release on every burst."""
PEAK: Final = 16384
"""The bursts' peak, of 32768 (-6 dBFS)."""
HEAD: Final = 2880
TAIL: Final = 4800
GAP: Final = 2400
CLAUSE_GAP: Final = 6000
SENTENCE_GAP: Final = 9600
TOKENS_PER_SECOND: Final = 12
"""Qwen3-TTS-12Hz: codec frames per second, used for ``new_tokens`` and the token cap."""

_MASK: Final = 0xFFFFFFFF
_SCALE: Final = 1.0 / 32768
_HZ_TOLERANCE: Final = 0.45
_SENTENCE_END = frozenset(".!?…")
_CLAUSE_END = frozenset(",;:—–-")
_CLOSERS = "\"'”’)]»"


@dataclass(frozen=True, slots=True)
class Burst:
    """One word's tone: its first sample and its number of 50 ms segments."""

    start: int
    segments: int

    @property
    def end(self) -> int:
        return self.start + self.segments * SEGMENT


def spoken_tokens(text: str) -> list[str]:
    """The words the fake speaks: whitespace-separated tokens with at least one letter or digit."""
    return [token for token in text.split() if any(ch.isalnum() for ch in token)]


def word_segments(token: str) -> int:
    """A word's length in segments: two more than its letters and digits, at least ``MIN_SEGMENTS``."""
    return max(MIN_SEGMENTS, 2 + sum(ch.isalnum() for ch in token))


def gap_after(token: str) -> int:
    """The silence after a word, in samples, from its closing punctuation."""
    tail = token.rstrip(_CLOSERS)
    if tail and tail[-1] in _SENTENCE_END:
        return SENTENCE_GAP
    if tail and tail[-1] in _CLAUSE_END:
        return CLAUSE_GAP
    return GAP


def layout(tokens: Sequence[str]) -> tuple[list[Burst], int]:
    """The bursts for ``tokens`` and the file's length in samples."""
    bursts: list[Burst] = []
    pos = HEAD
    for index, token in enumerate(tokens):
        burst = Burst(pos, word_segments(token))
        bursts.append(burst)
        pos = burst.end + (gap_after(token) if index < len(tokens) - 1 else TAIL)
    return bursts, (pos if tokens else HEAD + TAIL)


def truncate(bursts: Sequence[Burst], cut: int) -> tuple[list[Burst], int]:
    """The layout cut at sample ``cut`` (a token cap): whole segments only, and the file ends there."""
    kept: list[Burst] = []
    end = cut
    for burst in bursts:
        if burst.start >= cut:
            break
        if burst.end <= cut:
            kept.append(burst)
            continue
        whole = (cut - burst.start) // SEGMENT
        if whole >= 1:
            kept.append(Burst(burst.start, whole))
            end = burst.start + whole * SEGMENT
        else:
            end = burst.start
        break
    return kept, end


def symbol_of(utterance_id: int, segment_index: int) -> int:
    """The pitch symbol of segment ``segment_index``: byte ``segment_index mod 4`` of the id."""
    return (utterance_id >> (8 * (segment_index % ID_BYTES))) & 0xFF


def symbol_hz(symbol: int) -> float:
    """The pitch of a symbol, in Hz."""
    return BASE_HZ + STEP_HZ * symbol


@cache
def _table() -> list[int]:
    """One period of the tone, indexed by the phase's top 16 bits: 4u(1-|u|), scaled to ``PEAK``."""
    table: list[int] = []
    for index in range(65536):
        u = index - 65536 if index >= 32768 else index
        table.append((u * (32768 - abs(u)) >> 13) * PEAK // 32768)
    return table


@cache
def _increment(symbol: int) -> int:
    return round(symbol_hz(symbol) * 2**32 / SAMPLE_RATE)


def render(bursts: Sequence[Burst], total: int, utterance_id: int) -> array[float]:
    """The waveform: float32 samples, each an integer over 32768."""
    table = _table()
    ints = [0] * total
    index = 0
    for burst in bursts:
        phase = 0
        pos = burst.start
        for _ in range(burst.segments):
            inc = _increment(symbol_of(utterance_id, index))
            ints[pos : pos + SEGMENT] = [table[((phase + inc * i) & _MASK) >> 16] for i in range(1, SEGMENT + 1)]
            phase = (phase + inc * SEGMENT) & _MASK
            pos += SEGMENT
            index += 1
        ramp = min(RAMP, (burst.end - burst.start) // 2)
        for i in range(ramp):
            ints[burst.start + i] = ints[burst.start + i] * i // ramp
            ints[burst.end - 1 - i] = ints[burst.end - 1 - i] * i // ramp
    return array("f", [value * _SCALE for value in ints])


def new_tokens(samples: int) -> int:
    """The codec frames a render of ``samples`` would take."""
    return -(-samples * TOKENS_PER_SECOND // SAMPLE_RATE)


# ---------------------------------------------------------------------- hearing


@dataclass(frozen=True, slots=True)
class Heard:
    """What ``decode`` found in a file: its bursts as (first sample, segments), and the id if readable."""

    sample_rate: int
    bursts: tuple[tuple[int, int], ...]
    utterance_id: int | None


def find_bursts(samples: Sequence[float], sample_rate: int) -> list[tuple[int, int]]:
    """The spans of tone in a file, as (first sample, end sample), from 10 ms frame energy."""
    frame = max(1, sample_rate // 100)
    powers = [
        sum(map(mul, chunk, chunk)) / len(chunk)
        for chunk in (samples[i : i + frame] for i in range(0, len(samples), frame))
    ]
    if not powers or max(powers) <= 0.0:
        return []
    threshold = max(powers) * 0.01
    runs: list[list[int]] = []
    for index, power in enumerate(powers):
        if power <= threshold:
            continue
        if runs and index - runs[-1][1] <= 3:
            runs[-1][1] = index
        else:
            runs.append([index, index])
    peak = max(abs(v) for v in samples)
    level = peak * 0.02
    spans: list[tuple[int, int]] = []
    for first, last in runs:
        lo = max(0, (first - 1) * frame)
        hi = min(len(samples), (last + 2) * frame)
        start = next((i for i in range(lo, hi) if abs(samples[i]) > level), lo)
        end = next((i + 1 for i in range(hi - 1, lo - 1, -1) if abs(samples[i]) > level), hi)
        spans.append((start, end))
    return spans


def estimate_hz(samples: Sequence[float], start: int, end: int, sample_rate: int) -> float | None:
    """The pitch of a steady tone over [start, end), from the interpolated times of its rising zero crossings."""
    first: float | None = None
    last = 0.0
    count = 0
    prev = samples[start] if start < end else 0.0
    for i in range(start + 1, end):
        cur = samples[i]
        if prev < 0.0 <= cur:
            t = i - 1 + (-prev) / (cur - prev)
            if first is None:
                first = t
            last = t
            count += 1
        prev = cur
    if first is None or count < 2 or last <= first:
        return None
    return sample_rate * (count - 1) / (last - first)


def decode(audio: Audio) -> Heard:
    """Find the bursts and read the utterance id from their pitch (a majority vote per id byte)."""
    samples = audio.samples
    rate = audio.sample_rate
    seg = rate * SEGMENT_S
    votes: list[dict[int, int]] = [{} for _ in range(ID_BYTES)]
    found: list[tuple[int, int]] = []
    index = 0
    for start, end in find_bursts(samples, rate):
        count = round((end - start) / seg)
        if count < 1:
            continue
        found.append((start, count))
        for k in range(count):
            lo = start + round((k + 0.25) * seg)
            hi = min(end, start + round((k + 0.75) * seg))
            hz = estimate_hz(samples, lo, hi, rate)
            if hz is not None:
                symbol = round((hz - BASE_HZ) / STEP_HZ)
                if 0 <= symbol < SYMBOLS and abs(hz - symbol_hz(symbol)) <= _HZ_TOLERANCE:
                    tally = votes[index % ID_BYTES]
                    tally[symbol] = tally.get(symbol, 0) + 1
            index += 1
    utterance_id: int | None = None
    if all(votes):
        utterance_id = 0
        for position, tally in enumerate(votes):
            best = max(tally.items(), key=lambda item: (item[1], -item[0]))[0]
            utterance_id |= best << (8 * position)
    return Heard(sample_rate=rate, bursts=tuple(found), utterance_id=utterance_id)
