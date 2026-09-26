"""Records serialised by ``serial.to_json`` validate against the published schemas (sections 7.2, 7.5).

Every record is built twice from its annotations: once with every field filled in, once with every
nullable or optional field set to None. Both must validate, against the record's generated schema and,
for the records a tool returns, against that tool's hand-written output schema.
"""

from __future__ import annotations

import dataclasses
import inspect
import types
import typing
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

import pytest
from jsonschema import Draft202012Validator

from narration.config import LimitsConfig
from narration.contracts import models
from narration.contracts.errors import NarrationError
from narration.contracts.schemas import TOOLS_BY_NAME, fragment, record_schema
from narration.contracts.serial import ContractError, from_json, to_json


def _sample(tp: Any, *, nones: bool) -> Any:
    if tp is Any:
        return 1
    scalars: dict[Any, Any] = {str: "x", int: 1, float: 0.5, bool: True, Path: Path("x")}
    if tp in scalars:
        return scalars[tp]
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        args = get_args(tp)
        if nones and type(None) in args:
            return None
        return _sample(next(a for a in args if a is not type(None)), nones=nones)
    if origin is Literal:
        return get_args(tp)[0]
    if dataclasses.is_dataclass(tp):
        return _record(tp, nones=nones)  # pyright: ignore[reportArgumentType]
    if origin is tuple:
        args = get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return (_sample(args[0], nones=nones),)
        return tuple(_sample(a, nones=nones) for a in args)
    if origin is dict:
        return {"k": _sample(get_args(tp)[1], nones=nones)}
    raise TypeError(tp)


def _record(cls: type, *, nones: bool) -> Any:
    hints = typing.get_type_hints(cls)
    return cls(**{f.name: _sample(hints[f.name], nones=nones) for f in dataclasses.fields(cls)})


_RECORDS = [
    obj
    for _, obj in inspect.getmembers(models, inspect.isclass)
    if dataclasses.is_dataclass(obj) and obj.__module__ == models.__name__
]


def _valid(schema: dict[str, Any], value: Any) -> list[str]:
    return [
        f"{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in Draft202012Validator(schema).iter_errors(value)
    ]


@pytest.mark.parametrize("nones", [False, True], ids=["filled", "nones"])
@pytest.mark.parametrize("cls", [*_RECORDS, LimitsConfig], ids=lambda c: c.__name__)
def test_schema_every_record_validates_against_its_record_schema(cls: type, nones: bool) -> None:
    value = to_json(_record(cls, nones=nones))
    assert _valid(record_schema(cls), value) == []


@pytest.mark.parametrize("nones", [False, True], ids=["filled", "nones"])
def test_schema_get_results_segments_validate_s7_5(nones: bool) -> None:
    segment = to_json(_record(models.SegmentResult, nones=nones))
    error = NarrationError("ENGINE_DRIFT", "the canary drifted").error
    result = {
        "job": {
            "job_id": "job_x",
            "kind": "generate",
            "status": "failed",
            "outcome": None,
            "label": None,
            "error": to_json(error),
        },
        "segments": [segment],
        "consistency": to_json(_record(models.Consistency, nones=nones)),
        "listen_first": [to_json(_record(models.ListenFirstItem, nones=nones))],
    }
    assert _valid(TOOLS_BY_NAME["get_results"].output_schema, result) == []


def test_schema_unfinished_take_has_null_qa_and_alignment_s7_5() -> None:
    take = to_json(_record(models.TakeResult, nones=True))
    assert take["qa"] is None and take["alignment"] is None
    assert "fit" not in take, "fit is left out unless the segment gave scene_seconds (section 12)"


@pytest.mark.parametrize("nones", [False, True], ids=["filled", "nones"])
def test_schema_non_generate_results_validate_s7_6(nones: bool) -> None:
    result = {
        "design": {"design_id": "d", "candidates": [to_json(_record(models.Candidate, nones=nones))]},
        "measurement_result": {
            "path": "m.json",
            "measurement": to_json(_record(models.MeasurementRecord, nones=nones)),
        },
        "profile": to_json(_record(models.ProfileRecord, nones=nones)),
        "audition": to_json(_record(models.AuditionResult, nones=nones)),
    }
    assert _valid(TOOLS_BY_NAME["get_results"].output_schema, result) == []


def test_schema_error_and_flag_json_validate_against_fragments_s7_2() -> None:
    error = to_json(NarrationError("INVALID_ARGUMENT", "unknown field", field="instruct").error)
    assert set(error) == {"code", "message", "retryable", "hint", "field"}, "unset optionals are left out"
    assert _valid(fragment("Error"), error) == []
    flag = to_json(models.Flag(code="SEGMENT_TOO_LONG", severity="warn", message="m", segment_id="p1"))
    assert _valid(fragment("Flag"), flag) == []
    for result in ({"error": error}, {"job_id": None, "status": "planned", "warnings": [flag]}):
        assert _valid(TOOLS_BY_NAME["submit_job"].output_schema, result) == []


def test_schema_record_schema_absent_versus_null() -> None:
    schema = record_schema(models.TakeResult)
    assert "fit" not in schema["required"], "optional: left out when unset"
    assert schema["properties"]["fit"]["type"] == "object"
    assert "alignment" in schema["required"], "nullable: always present"
    assert schema["properties"]["alignment"]["type"] == ["object", "null"]
    assert "$ref" not in repr(schema)


def test_schema_nullable_field_without_default_stays_null_s11_2() -> None:
    cue = to_json(models.CueTiming(index=0, start_s=None, end_s=None, confidence=None))
    assert cue["start_s"] is None and cue["end_s"] is None, "an unplaced cue's times are null, never left out"


def test_schema_foreign_schema_id_and_wrong_literal_type_are_refused() -> None:
    take = to_json(_record(models.TakeRecord, nones=False))
    take["schema"] = "narration.take/v99"
    with pytest.raises(ContractError, match="schema"):
        from_json(models.TakeRecord, take)
    with pytest.raises(ContractError):
        from_json(models.Suggestion, {"tier": 1.0, "reason": "r"})
    assert from_json(models.Suggestion, {"tier": 1, "reason": "r"}).tier == 1
