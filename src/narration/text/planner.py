"""The text pipeline of design section 9.1 (``TextPlanner``, WP10).

Per request: duplicate segment ids and ambiguous hints are refused; every cue, segment text and hint is
sanitised, and all offenders of the request are reported together (``TEXT_REFUSED``). Per segment: the
cues are put in canonical form and joined; a ``text`` sent with cues must equal the join; exact spans
become word ranges. Per cue: hints make the engine text, and the four checks report what looks unspoken.
Across cues: a term that would match only across a boundary is not applied (``TERM_SPLIT_ACROSS_CUES``).
With ``strict_text``, any warning left refuses the request instead (``TEXT_REFUSED``).

The text is spoken as sent: only whitespace and Unicode form change; warnings never change the text.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import rfc8785

from narration.config import TextConfig
from narration.contracts import codes
from narration.contracts.errors import ConfigError, NarrationError
from narration.contracts.models import CueText, ExactWords, Flag, Hint, SegmentIn, SegmentText, TextChecksInfo
from narration.contracts.names import TEXT_CHECKS_VERSION

from .canonical import canonical_form, join
from .checks import Finding, check
from .hints import PreparedHint, SplitTerm, apply_hints, matching_order, prepare_hints, split_terms
from .rules import rules_document
from .sanitise import find_offenders
from .spans import resolve_exact

Offender = dict[str, Any]

_REFUSED_HINT = (
    "Remove every offender listed in details.offenders: markup the engine could take as instructions, or a "
    "control character other than tab, line feed and carriage return. Then send the request again."
)
_STRICT_HINT = (
    "Reword each token listed in details.offenders as spoken words, and keep each hinted term inside one cue; "
    "or send strict_text false to render with the warnings. A lone letter is info and never refuses."
)


def rules_sha256(refuse: tuple[str, ...]) -> str:
    """sha256 (64 hex) of the RFC 8785 canonical JSON of ``rules.rules_document(refuse)``."""
    return hashlib.sha256(rfc8785.dumps(rules_document(refuse))).hexdigest()


def _location(finding: Finding) -> str:
    return "in spoken text" if finding.respell_of is None else f"in the respelling of '{finding.respell_of}'"


def _message(finding: Finding) -> str:
    token, where = finding.token, _location(finding)
    match finding.kind:
        case "digit":
            return (
                f"'{token}' is a digit {where}; the engine will read it as it reads digits. "
                "Write the number in words to choose how it is read."
            )
        case "symbol":
            points = " ".join(f"U+{ord(ch):04X}" for ch in token)
            return (
                f"'{token}' ({points}) is a symbol {where}; the engine may read it aloud, skip it, or stumble. "
                "Write what should be said in words."
            )
        case "unit_like":
            return (
                f"'{token}' {where} looks like a unit symbol; the engine may read it letter by letter. "
                "Write the unit in words."
            )
        case "letter":
            return (
                f"'{token}' is a lone letter {where}; it may be an initial or a unit symbol, "
                "so listen to how it is read."
            )


def _warning(finding: Finding, segment_id: str, cue: int) -> Flag:
    details: dict[str, Any] = {"token": finding.token, "kind": finding.kind, "offset": finding.offset}
    if finding.respell_of is not None:
        details["respell_of"] = finding.respell_of
    return Flag(
        code=codes.WRITTEN_FORM_TOKEN,
        severity="info" if finding.kind == "letter" else "warn",
        message=_message(finding),
        segment_id=segment_id,
        cue=cue,
        details=details,
    )


def _split_flag(split: SplitTerm, segment_id: str) -> Flag:
    cues = (
        f"cues {split.first_cue} and {split.last_cue}"
        if split.last_cue == split.first_cue + 1
        else f"cues {split.first_cue} to {split.last_cue}"
    )
    return Flag(
        code=codes.TERM_SPLIT_ACROSS_CUES,
        severity="warn",
        message=(
            f"'{split.term}' runs across {cues}, so its hint was not applied there. "
            "Keep the whole term inside one cue for the hint to apply."
        ),
        segment_id=segment_id,
        cue=split.first_cue,
        details={"term": split.term, "cues": [split.first_cue, split.last_cue], "offset": split.offset},
    )


def _first_difference(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return min(len(a), len(b))


class TextPipeline:
    """The ``TextPlanner`` of section 9.1, configured by ``[text]`` (section 16).

    Raises ``ConfigError`` when ``text.checks`` names a rule set other than the one this code implements
    (``names.TEXT_CHECKS_VERSION``), or ``text.refuse`` holds an empty string.
    """

    def __init__(self, config: TextConfig | None = None) -> None:
        config = config if config is not None else TextConfig()
        if config.checks != TEXT_CHECKS_VERSION:
            raise ConfigError(
                f"text.checks = {config.checks!r}, but this service implements {TEXT_CHECKS_VERSION!r}; "
                "remove the key or set it to that value"
            )
        if any(not markup for markup in config.refuse):
            raise ConfigError("text.refuse must not contain an empty string")
        self._refuse = tuple(config.refuse)
        self._info = TextChecksInfo(version=TEXT_CHECKS_VERSION, rules_sha256=rules_sha256(self._refuse))

    @property
    def checks_info(self) -> TextChecksInfo:
        """``text_checks_version`` and the sha256 of the rules in force (section 9)."""
        return self._info

    # ------------------------------------------------------------------ the TextPlanner protocol

    def plan_segment(self, segment: SegmentIn, hints: Sequence[Hint], *, field_prefix: str = "") -> SegmentText:
        """One segment through section 9.1 steps 1–5.

        ``field_prefix`` (e.g. ``"segments[3]."``) is put in front of every ``field`` an error names.
        Raises ``NarrationError``: ``TEXT_REFUSED`` for markup or control characters (every offender in
        ``details.offenders``); ``INVALID_ARGUMENT`` (with ``field``) for an empty cue, a ``text`` that is
        not the join of the cues, an exact span that is out of range, overlapping or cuts a word, or a
        hint that is empty, has an empty respelling, or repeats a term.
        """
        prepared = prepare_hints(hints)
        offenders = self._hint_offenders(hints) + self._segment_offenders(segment, field_prefix)
        if offenders:
            raise self._refused(offenders)
        return self._plan(segment, prepared, field_prefix)

    def plan_request(
        self, segments: Sequence[SegmentIn], hints: Sequence[Hint], *, strict_text: bool
    ) -> tuple[SegmentText, ...]:
        """Every segment of a request, in order (``segments[i]`` in every ``field``).

        Duplicate segment ids are ``INVALID_ARGUMENT``. Markup and control characters anywhere in the
        request are one ``TEXT_REFUSED`` listing every offender. With ``strict_text``, any text warning left
        (severity warn: ``WRITTEN_FORM_TOKEN`` other than a lone letter, and ``TERM_SPLIT_ACROSS_CUES``)
        makes the request ``TEXT_REFUSED``, listing every one.
        """
        first_seen: dict[str, int] = {}
        for i, segment in enumerate(segments):
            if segment.segment_id in first_seen:
                earlier = first_seen[segment.segment_id]
                raise NarrationError(
                    codes.INVALID_ARGUMENT,
                    f"segments[{i}].segment_id '{segment.segment_id}' is already segments[{earlier}].segment_id",
                    field=f"segments[{i}].segment_id",
                    hint="Give every segment of a request its own segment_id.",
                )
            first_seen[segment.segment_id] = i
        prepared = prepare_hints(hints)
        offenders = self._hint_offenders(hints)
        for i, segment in enumerate(segments):
            offenders += self._segment_offenders(segment, f"segments[{i}].")
        if offenders:
            raise self._refused(offenders)
        planned = tuple(self._plan(segment, prepared, f"segments[{i}].") for i, segment in enumerate(segments))
        if strict_text:
            left = self._warnings_left(segments, planned)
            if left:
                raise NarrationError(
                    codes.TEXT_REFUSED,
                    f"strict_text is set and {len(left)} text warning(s) remain, so nothing was rendered",
                    field=left[0]["field"],
                    hint=_STRICT_HINT,
                    details={"offenders": left},
                )
        return planned

    # ------------------------------------------------------------------ sanitising (step 1)

    def _offenders(self, text: str, field: str, **where: Any) -> list[Offender]:
        return [
            {
                "field": field,
                **where,
                "text": o.text,
                "offset": o.offset,
                "reason": o.reason,
                "codepoints": o.codepoints,
            }
            for o in find_offenders(text, self._refuse)
        ]

    def _hint_offenders(self, hints: Sequence[Hint]) -> list[Offender]:
        found: list[Offender] = []
        for i, hint in enumerate(hints):
            found += self._offenders(hint.term, f"hints[{i}].term")
            if hint.respell is not None:
                found += self._offenders(hint.respell, f"hints[{i}].respell")
        return found

    def _segment_offenders(self, segment: SegmentIn, prefix: str) -> list[Offender]:
        sid = segment.segment_id
        found: list[Offender] = []
        for i, cue in enumerate(segment.cues):
            found += self._offenders(cue.text, f"{prefix}cues[{i}].text", segment_id=sid, cue=i)
        if segment.text is not None:
            where: dict[str, Any] = {"segment_id": sid} if segment.cues else {"segment_id": sid, "cue": 0}
            found += self._offenders(segment.text, f"{prefix}text", **where)
        return found

    def _refused(self, offenders: list[Offender]) -> NarrationError:
        return NarrationError(
            codes.TEXT_REFUSED,
            f"the text holds {len(offenders)} refused character(s) or string(s): markup the engine could take "
            f"as instructions ({' '.join(self._refuse)}) or control characters. Nothing was rendered.",
            field=offenders[0]["field"],
            hint=_REFUSED_HINT,
            details={"offenders": offenders},
        )

    # ------------------------------------------------------------------ steps 2–5

    def _plan(self, segment: SegmentIn, prepared: Sequence[PreparedHint], prefix: str) -> SegmentText:
        sid = segment.segment_id
        if segment.cues:
            received = [cue.text for cue in segment.cues]
            fields = [f"{prefix}cues[{i}].text" for i in range(len(received))]
        elif segment.text is not None:
            received = [segment.text]
            fields = [f"{prefix}text"]
        else:
            raise NarrationError(
                codes.INVALID_ARGUMENT,
                f"segment '{sid}' has neither cues nor text",
                field=f"{prefix}cues",
                hint="Send the segment's cues, or its text as one cue.",
            )

        spoken = [canonical_form(text) for text in received]
        for i, text in enumerate(spoken):
            if not text:
                raise NarrationError(
                    codes.INVALID_ARGUMENT,
                    f"{fields[i]} has no characters other than whitespace",
                    field=fields[i],
                    hint="Every cue needs words to speak; remove the empty cue.",
                )
        spoken_text = join(spoken)
        if segment.cues and segment.text is not None:
            text = canonical_form(segment.text)
            if text != spoken_text:
                at = _first_difference(text, spoken_text)
                raise NarrationError(
                    codes.INVALID_ARGUMENT,
                    f"{prefix}text is not the join of the cues (canonical forms differ at code point {at})",
                    field=f"{prefix}text",
                    hint="Send text equal to the cues joined by single spaces, or leave text out.",
                    details={"first_difference": at, "text": text, "join": spoken_text},
                )

        exact: list[tuple[ExactWords, ...]] = [
            resolve_exact(cue.text, cue.exact, f"{prefix}cues[{i}]") for i, cue in enumerate(segment.cues)
        ] or [()]

        ordered = matching_order([h for h in prepared if h.term in spoken_text])
        cues: list[CueText] = []
        engines: list[str] = []
        spoken_pos = engine_pos = 0
        for i, text in enumerate(spoken):
            applied = apply_hints(text, ordered)
            findings = check(applied.engine, applied.origin, applied.respelled_terms)
            warnings = tuple(_warning(f, sid, i) for f in findings)
            cues.append(
                CueText(
                    index=i,
                    received=received[i],
                    spoken=text,
                    engine=applied.engine,
                    spoken_span=(spoken_pos, spoken_pos + len(text)),
                    engine_span=(engine_pos, engine_pos + len(applied.engine)),
                    hints_applied=applied.applied,
                    warnings=warnings,
                    exact=exact[i],
                )
            )
            engines.append(applied.engine)
            spoken_pos += len(text) + 1
            engine_pos += len(applied.engine) + 1

        return SegmentText(
            segment_id=sid,
            cues=tuple(cues),
            spoken_text=spoken_text,
            engine_text=join(engines),
            spoken_chars=len(spoken_text),
            warnings=tuple(_split_flag(s, sid) for s in split_terms(spoken, ordered)),
            text_checks=self._info,
        )

    # ------------------------------------------------------------------ strict_text

    @staticmethod
    def _warnings_left(segments: Sequence[SegmentIn], planned: Sequence[SegmentText]) -> list[Offender]:
        left: list[tuple[tuple[int, int, int], Offender]] = []
        for i, (segment, text) in enumerate(zip(segments, planned, strict=True)):
            flags = [w for cue in text.cues for w in cue.warnings] + list(text.warnings)
            for flag in flags:
                if flag.severity != "warn":
                    continue
                cue = flag.cue if flag.cue is not None else 0
                details = flag.details or {}
                field = f"segments[{i}].cues[{cue}].text" if segment.cues else f"segments[{i}].text"
                entry: Offender = {
                    "field": field,
                    "segment_id": text.segment_id,
                    "cue": cue,
                    "code": flag.code,
                    "severity": flag.severity,
                    **details,
                }
                left.append(((i, cue, int(details.get("offset", 0))), entry))
        left.sort(key=lambda item: item[0])
        return [entry for _, entry in left]
