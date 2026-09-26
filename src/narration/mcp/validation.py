"""In-handler argument validation (design section 14).

The front-end validates a tool's arguments against the tool's own published input schema (JSON Schema
2020-12), inside the tool handler, so that a bad argument comes back as a tool error the model can act on
(``isError: true``, ``INVALID_ARGUMENT`` with ``field`` and ``hint``), never as a JSON-RPC error.

Every schema failure is ``INVALID_ARGUMENT`` (section 14: "schema or semantic failure"), a size bound such as
``maxItems`` included; ``LIMIT_EXCEEDED`` is the backend's, for the configured ``[limits]``. Messages never
repeat the caller's values: they name the field and the rule, and the hint says what to change.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.schemas import TOOLS, ToolSchema

MAX_REPORTED_ERRORS: Final = 20
"""How many schema failures ``details.errors`` lists; the first is the one ``field`` names."""

INSTRUCTION_LIKE_FIELDS: Final = frozenset(
    {
        "instruct",
        "instruction",
        "instructions",
        "style",
        "styles",
        "emotion",
        "emotions",
        "mood",
        "tone",
        "prompt",
        "speaker_prompt",
        "voice_prompt",
        "delivery",
    }
)
"""Unknown fields that ask for free-text delivery control, which v1 does not have (design section 3.3)."""

SECTION_3_3: Final = (
    "Inputs are typed and there is no free-text field; with a cloned clip, delivery comes from the clip "
    "(design section 3.3)."
)
TEXT_MODE_HINT: Final = (
    'Send text_mode "spoken" or leave it out, and write numbers, units and symbols as the words to be spoken: '
    'v1 speaks text as sent and has no normaliser for "written" text (design section 9.2).'
)

# Keywords whose failure is reported before others at the same depth: an unknown field first (the section
# 3.3 refusal), then a missing one.
_RANK: Final = {"additionalProperties": 0, "required": 1}


def field_path(parts: Iterable[str | int]) -> str:
    """``["voice", "path"]`` gives ``voice.path``; ``["segments", 0, "text"]`` gives ``segments[0].text``."""
    out = ""
    for part in parts:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out


def _json_list(values: Iterable[Any]) -> str:
    return ", ".join(json.dumps(v, ensure_ascii=False) for v in values)


def _required_branches(error: ValidationError) -> list[str] | None:
    """For an ``anyOf`` whose branches are only ``required`` lists (a segment's cues or text), their names."""
    branches = error.validator_value
    if error.validator != "anyOf" or not isinstance(branches, list):
        return None
    names: list[str] = []
    for branch in branches:
        if not isinstance(branch, dict) or set(branch) != {"required"}:
            return None
        names.extend(str(n) for n in branch["required"])
    return names


class _Failure:
    """One schema failure, described without the caller's values."""

    __slots__ = ("field", "hint", "message", "rule")

    def __init__(self, field: str, rule: str, message: str, hint: str) -> None:
        self.field = field
        self.rule = rule
        self.message = message
        self.hint = hint

    def as_json(self) -> dict[str, str]:
        return {"field": self.field, "rule": self.rule, "message": self.message}


def _describe(error: ValidationError) -> list[_Failure]:
    """Turn one jsonschema error into failures (an object with two unknown fields gives two)."""
    parts: list[str | int] = list(error.absolute_path)
    here = field_path(parts)
    where = here or "the arguments"
    rule = str(error.validator)
    value = error.validator_value
    schema: Mapping[str, Any] = error.schema if isinstance(error.schema, Mapping) else {}

    if rule == "additionalProperties" and isinstance(error.instance, dict):
        allowed = sorted(str(k) for k in schema.get("properties", {}))
        extra = sorted(str(k) for k in error.instance if k not in allowed)
        extra.sort(key=lambda name: name not in INSTRUCTION_LIKE_FIELDS)
        failures: list[_Failure] = []
        for name in extra:
            field = field_path([*parts, name])
            if name in INSTRUCTION_LIKE_FIELDS:
                hint = f"Remove {field}: it is not an input of this service. {SECTION_3_3}"
            else:
                accepted = ", ".join(allowed) if allowed else "none"
                hint = f"Remove {field}; the fields accepted here are: {accepted}. {SECTION_3_3}"
            failures.append(_Failure(field, rule, f"{field} is not an accepted field", hint))
        return failures

    if rule == "required" and isinstance(error.instance, dict) and isinstance(value, list):
        missing = [str(k) for k in value if k not in error.instance]
        return [
            _Failure(field_path([*parts, name]), rule, f"{field_path([*parts, name])} is required", "")
            for name in missing
        ] or [_Failure(here, rule, f"{where} is missing a required field", "")]

    if (names := _required_branches(error)) is not None:
        message = f"{where} needs one of: {', '.join(names)}"
        return [_Failure(here, rule, message, f"Add {' or '.join(names)} to {where}.")]

    if rule == "enum" and isinstance(value, list):
        hint = TEXT_MODE_HINT if parts and parts[-1] == "text_mode" else ""
        return [_Failure(here, rule, f"{where} must be one of: {_json_list(value)}", hint)]

    messages = {
        "type": lambda: f"{where} must be of type {' or '.join(value) if isinstance(value, list) else value}",
        "const": lambda: f"{where} must be {json.dumps(value, ensure_ascii=False)}",
        "minLength": lambda: f"{where} must be at least {value} characters long",
        "maxLength": lambda: f"{where} must be at most {value} characters long",
        "minItems": lambda: f"{where} must have at least {value} items",
        "maxItems": lambda: f"{where} must have at most {value} items",
        "uniqueItems": lambda: f"{where} must not repeat an item",
        "minimum": lambda: f"{where} must be at least {value}",
        "maximum": lambda: f"{where} must be at most {value}",
        "exclusiveMinimum": lambda: f"{where} must be greater than {value}",
        "exclusiveMaximum": lambda: f"{where} must be less than {value}",
        "pattern": lambda: f"{where} must match the pattern {value}",
        "minProperties": lambda: f"{where} must have at least {value} fields",
        "maxProperties": lambda: f"{where} must have at most {value} fields",
    }
    describe = messages.get(rule)
    message = describe() if describe is not None else f"{where} does not satisfy the schema rule {rule!r}"
    hints = {
        "maxItems": f"Send at most {value} items in {where}; split a larger request into several.",
        "maxLength": f"Shorten {where} to at most {value} characters.",
    }
    return [_Failure(here, rule, message, hints.get(rule, ""))]


def _hint(first: _Failure) -> str:
    if first.hint:
        return first.hint
    if first.rule == "required":
        return f"Add {first.field}; the tool's inputSchema lists its fields."
    return f"Change {first.field or 'the arguments'} to satisfy the tool's inputSchema."


def _sort_key(error: ValidationError) -> tuple[int, int, str]:
    return (len(error.absolute_path), _RANK.get(str(error.validator), 2), field_path(error.absolute_path))


class ArgumentValidator:
    """Validates one tool's arguments against its published input schema."""

    def __init__(self, tool: str, input_schema: Mapping[str, Any]) -> None:
        Draft202012Validator.check_schema(input_schema)
        self.tool = tool
        self._validator = Draft202012Validator(input_schema)

    def _failures(self, arguments: Mapping[str, Any]) -> list[_Failure]:
        errors = sorted(self._validator.iter_errors(dict(arguments)), key=_sort_key)
        out: list[_Failure] = []
        for error in errors:
            out.extend(_describe(error))
        return out

    def validate(self, arguments: Mapping[str, Any]) -> None:
        """Raise ``NarrationError(INVALID_ARGUMENT)`` naming the first failing field, with a hint."""
        failures = self._failures(arguments)
        if not failures:
            return
        first = failures[0]
        details: dict[str, Any] = {"errors": [f.as_json() for f in failures[:MAX_REPORTED_ERRORS]]}
        if len(failures) > MAX_REPORTED_ERRORS:
            details["more_errors"] = len(failures) - MAX_REPORTED_ERRORS
        raise NarrationError(
            codes.INVALID_ARGUMENT,
            first.message,
            field=first.field or None,
            hint=_hint(first),
            details=details,
            retryable=False,
        )


def build_validators(tools: Sequence[ToolSchema] = TOOLS) -> dict[str, ArgumentValidator]:
    """One validator per published tool, keyed by tool name."""
    return {t.name: ArgumentValidator(t.name, t.input_schema) for t in tools}
