"""JSON conversion for the contract records (``narration.contracts.models``).

``to_json`` turns a record into plain JSON values: a dataclass becomes an object keyed by its field names
(which are the design's names), and tuples become arrays. **An optional field that is unset is left out**:
a field whose value is None *and whose default is None* is omitted, so a published object carries only
what applies (``Error.field``, ``Flag.cue``, ``TakeResult.fit`` without ``scene_seconds``, …). A field
that is nullable but has no default (``CueTiming.start_s`` of an unplaced cue, design section 11.2) is
always present, as null.

``from_json`` builds a record back from such a value, checking types against the record's annotations. It
refuses unknown keys, missing required keys, a Literal of the wrong type (``1.0`` for ``1``) and a
versioned id (``schema``, ``seed_scheme``) other than the record's own, so a sidecar that does not match
its contract fails loudly. A missing key with a default takes the default, so omitted optionals read back.

Hashed objects (keys, design section 10.2) are built explicitly by ``narration.keys``, never by hashing a
whole record, so these presentation rules cannot move a key.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from functools import cache
from typing import Any, Literal, Union, get_args, get_origin

JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


class ContractError(ValueError):
    """A value does not match its contract record."""


def to_json(value: Any) -> Any:
    """Convert a record (or a structure of records) to plain JSON values."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out: dict[str, Any] = {}
        for f in dataclasses.fields(value):
            v = getattr(value, f.name)
            if v is None and f.default is None:
                continue
            out[f.name] = to_json(v)
        return out
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_json(v) for k, v in value.items()}
    return value


_VERSIONED_IDS = frozenset({"schema", "seed_scheme"})
"""Fields that name a record's version: a value other than the record's default is refused."""


@cache
def _hints(cls: type) -> dict[str, Any]:
    return typing.get_type_hints(cls)


def from_json[T](cls: type[T], data: Any, *, path: str = "$") -> T:
    """Build a record of type ``cls`` from plain JSON values; raises ContractError on any mismatch."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    if not isinstance(data, dict):
        raise ContractError(f"{path}: expected an object for {cls.__name__}, got {type(data).__name__}")
    hints = _hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ContractError(f"{path}: unknown key(s) for {cls.__name__}: {', '.join(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, field in fields.items():
        if name in _VERSIONED_IDS and name in data and isinstance(field.default, str) and data[name] != field.default:
            raise ContractError(f"{path}.{name}: expected {field.default!r}, got {data[name]!r}")
        if name in data:
            kwargs[name] = _convert(hints[name], data[name], f"{path}.{name}")
        elif field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            raise ContractError(f"{path}: missing required key '{name}' for {cls.__name__}")
    return cls(**kwargs)


def _convert(tp: Any, value: Any, path: str) -> Any:
    if tp is Any:
        return value
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        args = get_args(tp)
        if value is None:
            if type(None) in args:
                return None
            raise ContractError(f"{path}: null is not allowed")
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _convert(arg, value, path)
            except ContractError as exc:
                errors.append(str(exc))
        raise ContractError(f"{path}: matches none of {tp}: {'; '.join(errors)}")
    if origin is Literal:
        if not any(a == value and type(a) is type(value) for a in get_args(tp)):
            raise ContractError(f"{path}: {value!r} is not one of {get_args(tp)}")
        return value
    if dataclasses.is_dataclass(tp):
        return from_json(tp, value, path=path)  # type: ignore[arg-type]
    if origin is tuple:
        if not isinstance(value, list):
            raise ContractError(f"{path}: expected an array")
        args = get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], v, f"{path}[{i}]") for i, v in enumerate(value))
        if len(args) != len(value):
            raise ContractError(f"{path}: expected {len(args)} items, got {len(value)}")
        return tuple(_convert(a, v, f"{path}[{i}]") for i, (a, v) in enumerate(zip(args, value, strict=True)))
    if origin is list:
        if not isinstance(value, list):
            raise ContractError(f"{path}: expected an array")
        (arg,) = get_args(tp)
        return [_convert(arg, v, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ContractError(f"{path}: expected an object")
        _, varg = get_args(tp)
        return {str(k): _convert(varg, v, f"{path}.{k}") for k, v in value.items()}
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractError(f"{path}: expected an integer, got {value!r}")
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise ContractError(f"{path}: expected a boolean, got {value!r}")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise ContractError(f"{path}: expected a string, got {value!r}")
        return value
    raise TypeError(f"{path}: unsupported annotation {tp!r}")
