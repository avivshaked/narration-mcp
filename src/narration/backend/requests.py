"""What a request asks for, read from a tool's validated arguments (design sections 7.2, 7.3, 14, 16).

The front-end has already checked the arguments against the tool's input schema (``narration.mcp``), so what
is read here has the schema's shape. What the schema cannot say is checked here:

- **Controls** (section 3.3): every property of a segment's ``controls`` is refused by every current engine
  (``CONTROL_UNSUPPORTED``, naming the first one).
- **Limits** (``[limits]``, section 16): segments per job, cues per segment, spoken characters per segment
  and per job, hints per job. The operator may set them below the schema's bounds; a request over one is
  ``LIMIT_EXCEEDED`` (not retryable: split the request).
- **The job's request** is kept by value (``JobRecord.request``): the arguments as sent, with ``options``
  given the service's defaults (``[defaults]``) and without ``dry_run``, so the daemon runs exactly what the
  front-end planned. It still passes the tool's input schema.
- **Identity** (section 7.3, DC-6): ``request_sha256`` hashes the kind and that request (RFC 8785), without
  ``label`` (opaque, never used) and ``idempotency_key`` (a retry's name for the same request). The same
  request is the same job while one is active.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from narration import keys
from narration.config import DefaultsConfig, LimitsConfig
from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.models import Hint, SegmentIn, SubmitOptions
from narration.contracts.names import JobKind, Priority
from narration.contracts.serial import ContractError, from_json
from narration.jobs.plan import VoiceSpec
from narration.text import canonical_form, join

REQUEST_SCHEMA: Final = "narration.request/v1"
"""The schema id of the object ``request_sha256`` hashes."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SubmitRequest:
    """A ``submit_job`` request as the front-end reads it (section 7.3). ``stored`` is what the job keeps."""

    voice: VoiceSpec
    hints: tuple[Hint, ...]
    segments: tuple[SegmentIn, ...]
    options: SubmitOptions
    expect_engine_profile: str | None
    label: str | None
    stored: dict[str, Any]


def _invalid(message: str, field: str) -> NarrationError:
    return NarrationError(codes.INVALID_ARGUMENT, message, field=field)


def voice_of(args: Mapping[str, Any], field: str = "voice") -> VoiceSpec:
    """The request's Voice fragment (section 7.2)."""
    voice = args[field]
    return VoiceSpec(path=str(voice["path"]), sha256=str(voice["sha256"]), transcript=str(voice["transcript"]))


def hints_of(args: Mapping[str, Any]) -> tuple[Hint, ...]:
    """The request's hints, as records."""
    try:
        return tuple(from_json(Hint, h, path=f"hints[{i}]") for i, h in enumerate(args.get("hints") or ()))
    except ContractError as exc:
        raise _invalid(f"a hint cannot be read: {exc}", "hints") from exc


def segments_of(args: Mapping[str, Any]) -> tuple[SegmentIn, ...]:
    """The request's segments, as records."""
    out: list[SegmentIn] = []
    for i, segment in enumerate(args.get("segments") or ()):
        try:
            out.append(from_json(SegmentIn, segment, path=f"segments[{i}]"))
        except ContractError as exc:
            raise _invalid(f"segments[{i}] cannot be read: {exc}", f"segments[{i}]") from exc
    return tuple(out)


def stored_request(args: Mapping[str, Any], defaults: DefaultsConfig) -> dict[str, Any]:
    """The request a ``generate`` job keeps: the arguments, ``options`` with the service's defaults, no
    ``dry_run``."""
    stored = copy.deepcopy(dict(args))
    given = dict(stored.get("options") or {})
    given.pop("dry_run", None)
    stored["options"] = {
        "takes": defaults.takes,
        "max_retakes": defaults.max_retakes,
        "strict_text": False,
        "priority": "batch",
        **given,
    }
    return stored


def parse_submit(args: Mapping[str, Any], defaults: DefaultsConfig) -> SubmitRequest:
    """``submit_job``'s arguments as a ``SubmitRequest``."""
    stored = stored_request(args, defaults)
    given = dict(args.get("options") or {})
    options = stored["options"]
    priority: Priority = options["priority"]
    return SubmitRequest(
        voice=voice_of(args),
        hints=hints_of(args),
        segments=segments_of(args),
        options=SubmitOptions(
            dry_run=bool(given.get("dry_run", False)),
            takes=int(options["takes"]),
            max_retakes=int(options["max_retakes"]),
            strict_text=bool(options["strict_text"]),
            priority=priority,
            idempotency_key=options.get("idempotency_key"),
        ),
        expect_engine_profile=args.get("expect_engine_profile"),
        label=args.get("label"),
        stored=stored,
    )


def request_sha256(kind: JobKind, stored: Mapping[str, Any]) -> str:
    """The request's identity (section 7.3): sha256 (64 hex) of the RFC 8785 JSON of the kind and the stored
    request, without ``label`` and ``options.idempotency_key``."""
    body = {k: copy.deepcopy(v) for k, v in stored.items() if k != "label"}
    options = body.get("options")
    if isinstance(options, dict):
        options.pop("idempotency_key", None)
    return hashlib.sha256(keys.canonical_json({"schema": REQUEST_SCHEMA, "kind": kind, "request": body})).hexdigest()


# ---------------------------------------------------------------- controls (section 3.3)


def check_controls(segments: Sequence[SegmentIn]) -> None:
    """``CONTROL_UNSUPPORTED`` for the first control any segment sets: no current engine takes one."""
    for i, segment in enumerate(segments):
        if segment.controls:
            names = sorted(segment.controls)
            raise NarrationError(
                codes.CONTROL_UNSUPPORTED,
                f"segments[{i}].controls sets {', '.join(names)}, which no current engine supports",
                field=f"segments[{i}].controls.{names[0]}",
                details={"segment_id": segment.segment_id, "controls": names},
            )


# ---------------------------------------------------------------- limits (section 16)


def spoken_chars_of(segment: SegmentIn) -> int:
    """The segment's spoken length before any other check (section 7.2): the join of its cues' canonical
    forms, or its text's canonical form."""
    if segment.cues:
        return len(join(canonical_form(c.text) for c in segment.cues))
    return len(canonical_form(segment.text or ""))


def _limit(message: str, field: str, hint: str, details: dict[str, Any]) -> NarrationError:
    return NarrationError(codes.LIMIT_EXCEEDED, message, field=field, hint=hint, details=details)


def check_limits(segments: Sequence[SegmentIn], hints: Sequence[Hint], limits: LimitsConfig) -> None:
    """``LIMIT_EXCEEDED`` for the first ``[limits]`` bound the request is over (never retryable)."""
    if len(segments) > limits.max_segments_per_job:
        raise _limit(
            f"the request has {len(segments)} segments; this service takes at most {limits.max_segments_per_job}",
            "segments",
            "Split the request, and send the rest in another request.",
            {"segments": len(segments), "max_segments_per_job": limits.max_segments_per_job},
        )
    if len(hints) > limits.max_hints_per_job:
        raise _limit(
            f"the request has {len(hints)} hints; this service takes at most {limits.max_hints_per_job}",
            "hints",
            "Send only the hints the text uses, or split the request.",
            {"hints": len(hints), "max_hints_per_job": limits.max_hints_per_job},
        )
    total = 0
    for i, segment in enumerate(segments):
        if len(segment.cues) > limits.max_cues_per_segment:
            raise _limit(
                f"segments[{i}] has {len(segment.cues)} cues; this service takes at most {limits.max_cues_per_segment}",
                f"segments[{i}].cues",
                "Split the segment: put the rest of its cues in another segment.",
                {"cues": len(segment.cues), "max_cues_per_segment": limits.max_cues_per_segment},
            )
        chars = spoken_chars_of(segment)
        if chars > limits.max_chars_per_segment:
            raise _limit(
                f"segments[{i}] is {chars} spoken characters; this service takes at most "
                f"{limits.max_chars_per_segment} per segment",
                f"segments[{i}]",
                "Split the segment into shorter ones.",
                {"spoken_chars": chars, "max_chars_per_segment": limits.max_chars_per_segment},
            )
        total += chars
    if total > limits.max_chars_per_job:
        raise _limit(
            f"the request is {total} spoken characters; this service takes at most {limits.max_chars_per_job}",
            "segments",
            "Split the request, and send the rest in another request.",
            {"spoken_chars": total, "max_chars_per_job": limits.max_chars_per_job},
        )


__all__ = [
    "REQUEST_SCHEMA",
    "SubmitRequest",
    "check_controls",
    "check_limits",
    "hints_of",
    "parse_submit",
    "request_sha256",
    "segments_of",
    "spoken_chars_of",
    "stored_request",
    "voice_of",
]
