"""What a generation job asks for, read from its request (design sections 7.3, 8, 10.3).

A job keeps its request by value, as the tool's arguments after they passed the input schema
(``JobRecord.request``). The engine reads it again here: the voice, the hints, the segments, and the options
with the service's defaults (``[defaults]``). Everything the engine plans follows from the request alone:

- **Attempts** (section 10.3): a segment's ``attempts``, or ``0..takes-1``; one take slot per attempt, in
  that order. A retake uses the next attempt number above every attempt of the segment so far, in slot
  order (``next_attempt``), so the same request always asks for the same attempts, and so the same seeds.
- **The hints used** in a segment: the request's hints whose term the text pipeline applied in one of its
  cues (``hints_applied``), with the term as it matched there. They are what QA and the analysis key see
  (``QaInputs.hints``); a hint the segment does not use never changes its cached verdict.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from narration.config import DefaultsConfig
from narration.contracts.models import Hint, MeasurementRecord, SegmentIn, SegmentText
from narration.contracts.names import Priority
from narration.contracts.serial import ContractError, from_json
from narration.qa.checks import expected_wpm
from narration.text import canonical_form, words

from .admission import CHARS_PER_AUDIO_S


class RequestError(ValueError):
    """A job's request cannot be read (it should have passed the input schema at submit)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class VoiceSpec:
    """The request's voice (section 7.2 Voice): a clip the caller keeps, by location. Never stored."""

    path: str
    sha256: str
    transcript: str


@dataclass(frozen=True, slots=True, kw_only=True)
class GenerateRequest:
    """A ``submit_job`` request as the engine reads it (kinds ``generate`` and ``analyse``)."""

    voice: VoiceSpec
    hints: tuple[Hint, ...]
    segments: tuple[SegmentIn, ...]
    takes: int
    max_retakes: int
    strict_text: bool
    priority: Priority
    expect_engine_profile: str | None

    @classmethod
    def parse(cls, request: Mapping[str, Any], defaults: DefaultsConfig) -> GenerateRequest:
        """Read a stored request; raises ``RequestError`` for one that does not have the input schema's shape."""
        try:
            voice = request["voice"]
            spec = VoiceSpec(
                path=_str(voice, "path"), sha256=_str(voice, "sha256"), transcript=_str(voice, "transcript")
            )
            hints = tuple(from_json(Hint, h, path=f"$.hints[{i}]") for i, h in enumerate(request.get("hints") or ()))
            segments = tuple(
                from_json(SegmentIn, s, path=f"$.segments[{i}]") for i, s in enumerate(request["segments"])
            )
            options = request.get("options") or {}
            takes = _int(options, "takes", defaults.takes)
            max_retakes = _int(options, "max_retakes", defaults.max_retakes)
            strict = options.get("strict_text", False)
            priority = options.get("priority", "batch")
            expect = request.get("expect_engine_profile")
        except (KeyError, TypeError, ContractError) as exc:
            raise RequestError(f"the job's request cannot be read: {exc}") from exc
        if not segments:
            raise RequestError("the job's request has no segments")
        if not isinstance(strict, bool) or priority not in ("batch", "interactive"):
            raise RequestError("the job's options are malformed")
        if expect is not None and not isinstance(expect, str):
            raise RequestError("expect_engine_profile must be a string")
        return cls(
            voice=spec,
            hints=hints,
            segments=segments,
            takes=takes,
            max_retakes=max_retakes,
            strict_text=strict,
            priority=priority,
            expect_engine_profile=expect,
        )


def _str(mapping: Any, name: str) -> str:
    value = mapping[name]
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _int(mapping: Mapping[str, Any], name: str, default: int) -> int:
    value = mapping.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"options.{name} must be a non-negative integer")
    return value


def requested_attempts(segment: SegmentIn, takes: int) -> tuple[int, ...]:
    """The segment's attempts (section 10.3): the ones it names, or ``0..takes-1``. One take slot each."""
    if segment.attempts:
        return tuple(segment.attempts)
    return tuple(range(takes))


def next_attempt(used: Iterable[int]) -> int:
    """The next attempt number above every attempt of the segment so far (section 8): from the request alone."""
    return max(used, default=-1) + 1


def hints_used(segment: SegmentText, hints: Sequence[Hint]) -> tuple[Hint, ...]:
    """The request's hints the segment uses: those whose term the pipeline applied in one of its cues.

    Each is returned with its term and respelling in canonical form (as they matched the canonical text), no
    ``note`` (it enters no key), and its aliases in the order sent. Order: as the request lists them.
    """
    applied = {a.term for cue in segment.cues for a in cue.hints_applied}
    used: list[Hint] = []
    for hint in hints:
        term = canonical_form(hint.term)
        if term in applied:
            respell = canonical_form(hint.respell) if hint.respell is not None else None
            used.append(Hint(term=term, respell=respell, align_as=hint.align_as, asr_aliases=hint.asr_aliases))
    return tuple(used)


def estimated_audio_s(segment: SegmentText, measurement: MeasurementRecord | None) -> float:
    """The audio seconds one take of the segment should last: its spoken words at the voice's pace curve for
    this length, or ``CHARS_PER_AUDIO_S`` without one. Only for progress and estimates, never for a verdict."""
    if measurement is not None:
        wpm = expected_wpm(measurement.pace, segment.spoken_chars)
        count = len(words(segment.spoken_text))
        if wpm is not None and count:
            return max(1.0, count / wpm * 60.0)
    return max(1.0, segment.spoken_chars / CHARS_PER_AUDIO_S)


__all__ = [
    "GenerateRequest",
    "RequestError",
    "VoiceSpec",
    "estimated_audio_s",
    "hints_used",
    "next_attempt",
    "requested_attempts",
]
