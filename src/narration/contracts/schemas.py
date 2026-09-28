"""The published JSON Schemas: the fragments of design section 7.2 and every tool's input and output.

Rules (sections 5 and 7.2):

* **No ``$ref``.** Fragments are inlined wherever they are used; each use gets its own deep copy.
* **Every ``outputSchema`` has ``"type": "object"`` at its root**, whose properties are the success
  fields plus an optional ``error`` (Error). A **tool error** is ``isError: true`` with only ``error``.
  ``get_job`` and ``get_results`` on a ``failed`` job are successful calls: their result carries the
  job's status and, as ``error`` (``get_job``) or ``job.error`` (``get_results``), the job's terminal
  error (``JobRecord.error``), with ``isError`` false.
* **Absent versus null** follows ``models``: an optional field that is unset is left out; a nullable
  field is always present and may be null (a take with no analysis has ``qa: null``). The schemas of
  results built from records (the design, measurement, profile and audition results; ``limits``) are
  generated from the records by ``record_schema``, so the two cannot drift; the tests validate
  ``serial.to_json`` of every result record against its schema.
* **Inputs are strict** (``additionalProperties: false``), so an unknown field such as ``instruct`` is an
  argument error. **Outputs are open**, so adding a field later is not a breaking change.

DC-2 (plan.md section 1.5) is included: ``retry_after_s`` on Error, ``poll_after_s`` on ``submit_job``
and ``get_job``, and ``admission`` on ``get_server_status``.

The dialect is JSON Schema 2020-12. The front-end validates arguments inside its handlers (section 14).
"""

from __future__ import annotations

import copy
import dataclasses
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Union, get_args, get_origin

from .names import (
    JOB_PHASES,
    JOB_STATUSES,
    SEGMENT_STATES,
    SEVERITIES,
    TOOL_NAMES,
    id_schema_pattern,
)

Schema = dict[str, Any]
DIALECT: Final = "https://json-schema.org/draft/2020-12/schema"


def _c(schema: Schema) -> Schema:
    return copy.deepcopy(schema)


# ======================================================================== fragments (section 7.2)

_ID: Final[Schema] = {"type": "string", "pattern": "^[a-z0-9][a-z0-9._-]{0,63}$"}
_SHA256: Final[Schema] = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
_NUM: Final[Schema] = {"type": "number"}
_NUM_OR_NULL: Final[Schema] = {"type": ["number", "null"]}
_INT: Final[Schema] = {"type": "integer"}
_STR: Final[Schema] = {"type": "string"}
_STR_OR_NULL: Final[Schema] = {"type": ["string", "null"]}
_BOOL: Final[Schema] = {"type": "boolean"}

_FLAG: Final[Schema] = {
    "type": "object",
    "required": ["code", "severity", "message"],
    "properties": {
        "code": {"type": "string"},
        "severity": {"enum": list(SEVERITIES)},
        "message": {"type": "string"},
        "segment_id": {"type": "string"},
        "cue": {"type": "integer"},
        "retake_trigger": {"type": "boolean"},
        "details": {"type": "object"},
    },
}

_ERROR: Final[Schema] = {
    "type": "object",
    "description": "retryable: the same call may succeed later unchanged; false means the arguments must change",
    "required": ["code", "message", "retryable"],
    "properties": {
        "code": {"type": "string"},
        "message": {"type": "string"},
        "retryable": {"type": "boolean"},
        "hint": {"type": "string"},
        "field": {"type": "string"},
        "details": {"type": "object"},
        "retry_after_s": {
            "type": "number",
            "minimum": 0,
            "description": "set on every retryable error: wait at least this long, add your own jitter, then "
            "send the identical request again (it is deduplicated)",
        },
    },
}

_FILE_SHA256: Final[Schema] = {
    **_SHA256,
    "description": "the file's sha256 in 64 lower-case hex digits (lower-case a hash a tool prints in upper case)",
}

_VOICE: Final[Schema] = {
    "type": "object",
    "additionalProperties": False,
    "description": "the voice: a clip you keep, sent exactly as it was kept with the clip. The transcript is part "
    "of the voice, so one that differs by a single character is another voice, with no measurement yet",
    "required": ["path", "sha256", "transcript"],
    "properties": {
        "path": {"type": "string", "description": "absolute path of a WAV on a local drive"},
        "sha256": _FILE_SHA256,
        "transcript": {
            "type": "string",
            "minLength": 1,
            "maxLength": 600,
            "description": "the exact words spoken in the clip, copied character for character from the record "
            "kept with it (where design_voice is available, the candidate's transcript); never retyped",
        },
    },
}

_AUDIO: Final[Schema] = {
    "type": "object",
    "additionalProperties": False,
    "description": "a WAV file, by path and sha256",
    "required": ["path", "sha256"],
    "properties": {
        "path": {"type": "string", "description": "absolute path of a WAV on a local drive"},
        "sha256": _FILE_SHA256,
    },
}

_HINT: Final[Schema] = {
    "type": "object",
    "additionalProperties": False,
    "description": "a hint for one term. Send every invented or unusual name as one, the term alone if it needs "
    "no respelling: QA then scores the name as one word, where a name without a hint counts as misheard words "
    "and can fail the take (WER_HIGH)",
    "required": ["term"],
    "properties": {
        "term": {
            "type": "string",
            "minLength": 1,
            "maxLength": 80,
            "description": "the word or words as they appear in the spoken text, matched case-sensitively as whole "
            "words (the term followed by 's matches too); each term once per request",
        },
        "respell": {
            "type": "string",
            "maxLength": 120,
            "description": "optional: what the engine is given instead; a hint to the engine, never a guarantee. "
            "Leave it out when the engine says the term well as written",
        },
        "align_as": {
            "type": "string",
            "maxLength": 120,
            "description": "optional: letters for the aligner, spelling the term as it sounds (default: the term's "
            "own letters)",
        },
        "asr_aliases": {
            "type": "array",
            "maxItems": 10,
            "items": {"type": "string", "maxLength": 120},
            "description": "optional: spellings the speech recogniser writes for the term (see qa.terms[].heard); "
            "QA counts them as the term",
        },
        "note": {"type": "string", "maxLength": 500, "description": "optional: for the caller; enters no key"},
    },
}

_CONTROLS: Final[Schema] = {
    "type": "object",
    "additionalProperties": False,
    "description": "every property is refused by every current engine (CONTROL_UNSUPPORTED; design section 3.3)",
    "properties": {
        "pace": {
            "type": "object",
            "additionalProperties": False,
            "description": "refused (CONTROL_UNSUPPORTED)",
            "properties": {
                "factor": {"type": "number", "minimum": 0.94, "maximum": 1.06, "description": "part of pace"},
                "mode": {"enum": ["native", "time_stretch"], "description": "part of pace"},
            },
        },
        "context_before": {"type": "string", "maxLength": 1200, "description": "refused (CONTROL_UNSUPPORTED)"},
        "context_after": {"type": "string", "maxLength": 1200, "description": "refused (CONTROL_UNSUPPORTED)"},
    },
}

_EXACT_SPAN: Final[Schema] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["start", "end"],
    "properties": {
        "start": {
            "type": "integer",
            "minimum": 0,
            "description": "Unicode code points into this cue's text as sent: where the span's first word starts",
        },
        "end": {
            "type": "integer",
            "minimum": 1,
            "description": "Unicode code points into this cue's text as sent, end exclusive; whole words only",
        },
    },
}

_CUE: Final[Schema] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text"],
    "properties": {
        "text": {
            "type": "string",
            "minLength": 1,
            "maxLength": 600,
            "description": "the words to be spoken, as the caller wants them heard",
        },
        "at_s": {"type": "number", "minimum": 0, "description": "optional, informational"},
        "exact": {
            "type": "array",
            "maxItems": 20,
            "description": "R14: spans that must be heard exactly",
            "items": _EXACT_SPAN,
        },
    },
}

_SEGMENT: Final[Schema] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["segment_id"],
    "anyOf": [{"required": ["cues"]}, {"required": ["text"]}],
    "properties": {
        "segment_id": {**_ID, "description": "the caller's own name, echoed back"},
        "cues": {
            "type": "array",
            "minItems": 1,
            "maxItems": 40,
            "items": _CUE,
            "description": "the segment's cues (a caption or sentence each), in order. Each take gives each cue's "
            "start and end time, or null times with CUE_UNALIGNED for a cue it could not place (never "
            "interpolated)",
        },
        "text": {
            "type": "string",
            "minLength": 1,
            "maxLength": 1200,
            "description": "optional with cues; if both, must equal the join of the cues",
        },
        "attempts": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "uniqueItems": True,
            "items": {"type": "integer", "minimum": 0, "maximum": 99},
            "description": "optional: exactly which attempts to render; default 0..takes-1",
        },
        "scene_seconds": {
            "type": "number",
            "exclusiveMinimum": 0,
            "maximum": 900,
            "description": "optional budget; absent = no fit reporting of any kind (R4)",
        },
        "fit": {
            "type": "object",
            "additionalProperties": False,
            "description": "optional, used only with scene_seconds: the scene's seconds kept clear of narration",
            "properties": {
                "lead_in_s": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 5,
                    "default": 0,
                    "description": "seconds before the narration starts, taken off the budget",
                },
                "tail_s": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 5,
                    "default": 0,
                    "description": "seconds after the narration ends, taken off the budget",
                },
            },
        },
        "controls": _CONTROLS,
    },
}

_JOB_ID: Final[Schema] = {"type": "string", "pattern": id_schema_pattern("job_id"), "description": "a job_id"}
_ENGINE_REF: Final[Schema] = {"type": "object", "properties": {"id": _STR, "hash": _STR}}


def _array(items: Schema, **extra: Any) -> Schema:
    return {"type": "array", "items": _c(items), **extra}


def _obj(properties: dict[str, Schema], *, required: list[str] | None = None, **extra: Any) -> Schema:
    schema: Schema = {"type": "object", "properties": {k: _c(v) for k, v in properties.items()}}
    if required:
        schema["required"] = required
    schema.update(extra)
    return schema


def _input(properties: dict[str, Schema], *, required: list[str] | None = None, **extra: Any) -> Schema:
    """A tool's input schema: strict, so unknown fields are refused."""
    schema = _obj(properties, required=required, **extra)
    schema["additionalProperties"] = False
    return schema


def _output(properties: dict[str, Schema]) -> Schema:
    """A tool's output schema: an object with the success fields plus an optional error."""
    return _obj({**properties, "error": _ERROR})


def _nullable(schema: Schema) -> Schema:
    """A copy of ``schema`` that also accepts null."""
    out = _c(schema)
    if "enum" in out:
        if None not in out["enum"]:
            out["enum"] = [*out["enum"], None]
    elif isinstance(out.get("type"), str):
        out["type"] = [out["type"], "null"]
    elif isinstance(out.get("type"), list):
        if "null" not in out["type"]:
            out["type"] = [*out["type"], "null"]
    else:
        out = {"anyOf": [out, {"type": "null"}]}
    return out


def record_schema(cls: type) -> Schema:
    """The open output schema of a record's JSON form (``serial.to_json``), built from its annotations.

    Every field is a property. Fields that ``to_json`` always writes are required; an optional field
    (``X | None = None``) is not required and, when present, is never null; a nullable field (``X | None``
    with no default) is required and may be null. No ``$ref``: nested records are inlined.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    hints = typing.get_type_hints(cls)
    properties: dict[str, Schema] = {}
    required: list[str] = []
    for f in dataclasses.fields(cls):
        tp = hints[f.name]
        if f.default is None:
            properties[f.name] = _type_schema(_strip_none(tp))
        else:
            properties[f.name] = _type_schema(tp)
            required.append(f.name)
    schema: Schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _strip_none(tp: Any) -> Any:
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        args = tuple(a for a in get_args(tp) if a is not type(None))
        return args[0] if len(args) == 1 else Union[args]  # noqa: UP007
    return tp


_SCALARS: Final[dict[Any, Schema]] = {str: _STR, int: _INT, float: _NUM, bool: _BOOL, Path: _STR}


def _type_schema(tp: Any) -> Schema:
    if tp is Any:
        return {}
    if tp in _SCALARS:
        return _c(_SCALARS[tp])
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        args = get_args(tp)
        inner = [a for a in args if a is not type(None)]
        schema = _type_schema(inner[0]) if len(inner) == 1 else {"anyOf": [_type_schema(a) for a in inner]}
        return _nullable(schema) if type(None) in args else schema
    if origin is Literal:
        return {"enum": list(get_args(tp))}
    if dataclasses.is_dataclass(tp):
        return record_schema(tp)  # pyright: ignore[reportArgumentType]
    if origin is tuple:
        args = get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": _type_schema(args[0])}
        return {
            "type": "array",
            "prefixItems": [_type_schema(a) for a in args],
            "minItems": len(args),
            "maxItems": len(args),
        }
    if origin is list:
        return {"type": "array", "items": _type_schema(get_args(tp)[0])}
    if origin is dict:
        value = get_args(tp)[1]
        if value is Any:
            return {"type": "object"}
        return {"type": "object", "additionalProperties": _type_schema(value)}
    raise TypeError(f"no schema for {tp!r}")


def _result_records() -> dict[str, Schema]:
    """The schemas of results that are records (imported here: ``models`` must not depend on schemas)."""
    from narration.config import LimitsConfig

    from . import models

    return {
        "candidate": record_schema(models.Candidate),
        "measurement": record_schema(models.MeasurementRecord),
        "profile": record_schema(models.ProfileRecord),
        "audition": record_schema(models.AuditionResult),
        "limits": record_schema(LimitsConfig),
    }


# ======================================================================== shared output parts

_HINT_APPLIED = _obj({"term": _STR, "respell": _STR_OR_NULL, "offset": _INT})
_CUE_TEXT = _obj(
    {
        "index": _INT,
        "received": _STR,
        "spoken": _STR,
        "engine": _STR,
        "hints_applied": _array(_HINT_APPLIED),
        "warnings": _array(_FLAG),
        "exact": _array(
            _obj({"start": _INT, "end": _INT, "words": {"type": "array", "items": _INT, "minItems": 2, "maxItems": 2}})
        ),
    }
)
_WORD_TIMING = _obj({"text": _STR, "start_s": _NUM_OR_NULL, "end_s": _NUM_OR_NULL})
_CUE_TIMING = _obj(
    {
        "index": _INT,
        "start_s": _NUM_OR_NULL,
        "end_s": _NUM_OR_NULL,
        "confidence": _NUM_OR_NULL,
        "words": _array(_WORD_TIMING),
    }
)
_MEASURED_ERROR = _obj(
    {"p50_s": _NUM_OR_NULL, "p95_s": _NUM_OR_NULL, "n": {"type": ["integer", "null"]}, "benchmark": _STR_OR_NULL}
)
_ALIGNMENT = _obj(
    {
        "method": _STR,
        "model": _STR,
        "revision": _STR,
        "cross_check": {"type": "string", "description": "the cross-check in words (the ASR's word timestamps)"},
        "max_disagreement_s": _NUM_OR_NULL,
        "measured_error": {**_nullable(_MEASURED_ERROR), "description": "null until an alignment benchmark exists"},
        "flags": _array(_FLAG),
    }
)
_DELIVERY = _obj({"path": _STR, "sha256": _STR, "samples": _INT, "sample_rate": _INT, "duration_s": _NUM})
_TRIM = _obj({"head_s": _NUM, "tail_s": _NUM, "pad_s": _NUM})
_LOUDNESS = _obj(
    {
        "measured_lufs": {**_NUM_OR_NULL, "description": "null when the take has no measurable loudness"},
        "gain_db": _NUM,
        "true_peak_dbtp": {**_NUM_OR_NULL, "description": "null when the take has no measurable loudness"},
        "ceiling_applied": _BOOL,
    }
)
_EXACT_RESULT = _obj(
    {
        "cue": _INT,
        "start": _INT,
        "end": _INT,
        "expected": _STR,
        "heard": _STR_OR_NULL,
        "match": {"enum": ["same", "different", "missing"]},
    }
)
_TERM_RESULT = _obj({"term": _STR, "cue": _INT, "heard": _STR_OR_NULL, "ok": _BOOL})
_PACE = _obj({"spoken_wpm": _NUM_OR_NULL, "articulation_cps": _NUM_OR_NULL})
_QA = _obj(
    {
        "verdict": {"enum": ["pass", "warn", "fail"]},
        "flags": _array(_FLAG),
        "wer_raw": _NUM_OR_NULL,
        "wer_adj": _NUM_OR_NULL,
        "exact_ok": _BOOL,
        "exact": _array(_EXACT_RESULT),
        "terms": _array(_TERM_RESULT),
        "spk_sim_anchor": _NUM_OR_NULL,
        "pace": _PACE,
        "pace_expected": _PACE,
        "transcript": _STR_OR_NULL,
    }
)
_FIT = _obj(
    {
        "scene_seconds": _NUM,
        "budget_s": _NUM,
        "duration_s": _NUM,
        "slack_s": _NUM,
        "overrun_s": _NUM,
        "flags": _array(_FLAG),
    }
)
_TAKE = _obj(
    {
        "take_id": _STR,
        "render_id": _STR,
        "attempt": _INT,
        "seed": _INT,
        "fresh": {"type": "boolean", "description": "true if this job rendered it, false if it came from the cache"},
        "delivery": _DELIVERY,
        "trim": _TRIM,
        "loudness": _LOUDNESS,
        "analysis_id": _STR_OR_NULL,
        "cues": _array(_CUE_TIMING),
        "alignment": {**_nullable(_ALIGNMENT), "description": "null for a take with no analysis"},
        "qa": {**_nullable(_QA), "description": "null for a take with no analysis"},
        "fit": {**_c(_FIT), "description": "present only when the segment gave scene_seconds"},
        "flags": _array(
            _FLAG,
            description="this take's flags outside its cached verdict, never part of qa.verdict: delivery flags "
            "(LOUDNESS_UNDER_TARGET, GAIN_HIGH) and per-job flags (SPK_OUTLIER, CANARY_MISMATCH, RETAKEN)",
        ),
    }
)
_SEGMENT_TEXT = _obj(
    {
        "segment_id": _STR,
        "spoken_chars": _INT,
        "cues": _array(_CUE_TEXT),
        "max_segment_chars": {"type": ["integer", "null"]},
        "over_by_chars": {"type": ["integer", "null"]},
        "est_duration_s": _NUM_OR_NULL,
        "warnings": _array(_FLAG),
    }
)
_PROGRESS = _obj({"done_s": _NUM, "total_s": _NUM, "fraction": _NUM, "segments_done": _INT, "segments_total": _INT})
_MEASUREMENT_SUMMARY = _obj(
    {"max_segment_chars": {"type": ["integer", "null"]}, "max_segment_seconds": _NUM_OR_NULL, "measured_at": _STR}
)
_LINT = _obj(
    {
        "policy": {"enum": ["warn"]},
        "findings": _array(_obj({"phrase": _STR, "offset": _INT, "suggestion": _STR})),
        "note": _STR,
    }
)

# ======================================================================== tools

_HINTS_NOTE: Final = (
    "pronunciation hints for the terms the segments speak: every invented or unusual name (the term alone is "
    "enough), and every term to respell"
)

POLL_AFTER_S: Final[Schema] = {
    "type": "number",
    "minimum": 0,
    "description": "the earliest poll worth making, in seconds (longer while waiting_for_gpu)",
}


def _submit_job_input() -> Schema:
    return _input(
        {
            "voice": _VOICE,
            "expect_engine_profile": {
                "type": "string",
                "description": "optional engine profile hash; refuse (ENGINE_CHANGED) if the service's differs",
            },
            "text_mode": {
                "enum": ["spoken"],
                "default": "spoken",
                "description": "v1 speaks the text as sent; 'written' is a later phase (design section 9.2)",
            },
            "hints": _array(_HINT, maxItems=500, description=_HINTS_NOTE),
            "segments": _array(
                _SEGMENT,
                minItems=1,
                maxItems=200,
                description="the paragraphs to narrate, a segment each; keep a job to a scene, about 8 to 10 "
                "segments, so a client can read its get_results whole",
            ),
            "label": {"type": "string", "maxLength": 100, "description": "opaque; shown in status, never used"},
            "options": _input(
                {
                    "dry_run": {
                        "type": "boolean",
                        "default": False,
                        "description": "true: plan only, and the text echo; nothing is queued or rendered. The plan "
                        "covers the requested attempts (retakes are not planned): segments_cached (segments whose "
                        "every attempt is already scored), renders_needed, deliveries_needed, analyses_needed, "
                        "est_audio_s and est_wall_s. Scored attempts are counted when the service knows its QA "
                        "pins; otherwise segments_cached is 0 and every attempt counts in analyses_needed",
                    },
                    "takes": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 3,
                        "default": 1,
                        "description": "distinct deliveries per segment (attempts 0..takes-1) unless a segment "
                        "names its attempts (R9)",
                    },
                    "max_retakes": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 3,
                        "default": 2,
                        "description": "automatic retakes per failing take, on the next attempt numbers; a fail "
                        "flag (such as WER_HIGH), CUE_UNALIGNED or HEAD_INSERTION triggers one",
                    },
                    "strict_text": {
                        "type": "boolean",
                        "default": False,
                        "description": "refuse (TEXT_REFUSED) while any text warning remains (a digit, symbol or "
                        "unit-like token, or a hinted term split across cues); default: warn and render, the "
                        "text spoken as the engine reads it",
                    },
                    "priority": {
                        "enum": ["batch", "interactive"],
                        "default": "batch",
                        "description": "'interactive' runs before queued batch jobs, and a running batch job gives "
                        "way to it between pieces of its work: use it for a short redo while a long batch runs",
                    },
                    "idempotency_key": {
                        "type": "string",
                        "maxLength": 64,
                        "description": "optional: your key for this request, so a retry returns its job; reused for "
                        "a different request while that job is active, it is refused (INVALID_ARGUMENT)",
                    },
                },
                description="how the job runs; every field is optional",
            ),
        },
        required=["voice", "segments"],
    )


def _submit_job_output() -> Schema:
    return _output(
        {
            "job_id": {"type": ["string", "null"], "description": "null only for dry_run"},
            "status": {"enum": ["queued", "running", "completed", "planned"]},
            "voice_hash": _STR,
            "engine_profile": _ENGINE_REF,
            "plan": _obj(
                {
                    "segments_total": _INT,
                    "segments_cached": _INT,
                    "renders_needed": _INT,
                    "deliveries_needed": _INT,
                    "analyses_needed": _INT,
                    "est_audio_s": _NUM,
                    "est_wall_s": _NUM,
                    "queue_position": _INT,
                }
            ),
            "poll_after_s": POLL_AFTER_S,
            "text": _array(
                _SEGMENT_TEXT,
                description="dry_run only: per segment, the same text echo and length check as check_text (R7, R8)",
            ),
            "warnings": _array(_FLAG, description="text warnings and SEGMENT_TOO_LONG"),
        }
    )


def _get_job_output() -> Schema:
    return _output(
        {
            "job_id": _STR,
            "kind": _STR,
            "label": _STR_OR_NULL,
            "status": {"enum": list(JOB_STATUSES)},
            "phase": {"enum": [*JOB_PHASES, None]},
            "round": _INT,
            "outcome": {"enum": ["all_passed", "needs_attention", None]},
            "progress": _PROGRESS,
            "eta_s": _NUM_OR_NULL,
            "queue_position": {"type": ["integer", "null"]},
            "poll_after_s": POLL_AFTER_S,
            "message": _STR_OR_NULL,
            "segments": _array(
                _obj(
                    {
                        "segment_id": _STR,
                        "state": {"enum": list(SEGMENT_STATES)},
                        "takes_ok": _INT,
                        "retakes_used": _INT,
                    }
                )
            ),
            "updated_at": _STR,
        }
    )


def _get_results_output() -> Schema:
    records = _result_records()
    return _output(
        {
            "job": _obj(
                {
                    "job_id": _STR,
                    "kind": _STR,
                    "status": {"enum": list(JOB_STATUSES)},
                    "outcome": {"enum": ["all_passed", "needs_attention", None]},
                    "label": _STR_OR_NULL,
                    "error": {**_nullable(_ERROR), "description": "the job's terminal error when it failed"},
                }
            ),
            "voice": _obj({"voice_hash": _STR, "clip_sha256": _STR}),
            "engine_profile": _ENGINE_REF,
            "measurement": _MEASUREMENT_SUMMARY,
            "segments": _array(
                _obj(
                    {
                        "segment_id": _STR,
                        "status": {"enum": list(SEGMENT_STATES)},
                        "suggested_take_id": {
                            **_STR_OR_NULL,
                            "description": "the take QA suggests for this segment (advice; you choose); use it "
                            "rather than takes[0]. With suggestion.tier 4 every take failed QA (the job's outcome "
                            "is then needs_attention): resolve its flags or redo the segment before keeping it",
                        },
                        "suggestion": {
                            **_nullable(_obj({"tier": {"enum": [1, 2, 3, 4]}, "reason": _STR})),
                            "description": "null for a segment with no take",
                        },
                        "text": _obj({"spoken_chars": _INT, "cues": _array(_CUE_TEXT)}),
                        "takes": _array(
                            _TAKE,
                            description="every take rendered or found for this request, failed and replaced "
                            "attempts included",
                        ),
                        "flags": _array(_FLAG),
                    }
                )
            ),
            "consistency": _obj(
                {
                    "min": _NUM_OR_NULL,
                    "median": _NUM_OR_NULL,
                    "outliers": _array(_STR),
                },
                description="a report on the suggested takes of this request, never part of a verdict",
            ),
            "listen_first": _array(
                _obj(
                    {
                        "segment_id": _STR,
                        "take_id": _STR_OR_NULL,
                        "cue": {"type": ["integer", "null"]},
                        "reason": _STR,
                        "from_s": _NUM_OR_NULL,
                        "to_s": _NUM_OR_NULL,
                    }
                )
            ),
            "report_md": _STR,
            "licence": {"type": "object"},
            "design": _obj(
                {"design_id": _STR, "candidates": _array(records["candidate"])},
                description="design_voice jobs: each candidate's clip, exact transcript, seed, lint and profile",
            ),
            "measurement_result": _obj(
                {"path": _STR, "measurement": records["measurement"]},
                description="measure_voice jobs: the full measurement, and the path of its JSON file to keep",
            ),
            "profile": {**records["profile"], "description": "profile_voice jobs: measurements and pictures"},
            "audition": {
                **records["audition"],
                "description": "audition_pronunciation jobs: per variant, its takes and what the ASR heard; "
                "whoever owns the text decides by ear",
            },
        }
    )


def _server_status_output() -> Schema:
    return _output(
        {
            "version": _STR,
            "spec_revision": {
                "type": "string",
                "description": "the MCP specification revision the server is built on (a date), not a revision "
                "of the service's own design",
            },
            "store_root": _STR,
            "daemon": _obj(
                {
                    "state": {"enum": ["stopped", "idle", "busy", "stopping"]},
                    "pid": {"type": ["integer", "null"]},
                    "current_job": {
                        "type": ["object", "null"],
                        "properties": {
                            "job_id": _STR,
                            "kind": _STR,
                            "label": _STR_OR_NULL,
                            "phase": _STR_OR_NULL,
                            "started_at": _STR,
                        },
                    },
                    "workers": _array(_obj({"role": _STR, "pid": _INT})),
                }
            ),
            "gpu": _obj(
                {
                    "name": _STR_OR_NULL,
                    "total_mb": {"type": ["integer", "null"]},
                    "free_mb": {"type": ["integer", "null"]},
                    "in_use": _BOOL,
                    "holder": {"enum": ["qwen", "qa", None]},
                    "unload_in_s": _NUM_OR_NULL,
                }
            ),
            "cpu_threads": _INT,
            "queue": _obj({"length": _INT, "jobs": _array({"type": "object"})}),
            "admission": _obj(
                {
                    "accepting": _BOOL,
                    "queue": _obj({"length": _INT, "max": _INT, "est_drain_s": _NUM_OR_NULL}),
                    "rate": _obj({"remaining": _INT, "resets_in_s": _NUM}),
                    "gpu": _obj(
                        {
                            "in_use": _BOOL,
                            "holder": {"enum": ["qwen", "qa", None]},
                            "free_mb": {"type": ["integer", "null"]},
                            "need_mb": {"type": "object", "description": "VRAM needed by model group"},
                            "waiting_since": _STR_OR_NULL,
                        }
                    ),
                },
                description="DC-2: whether and when to submit; the service suggests, the consumer decides",
            ),
            "engine_profiles": _array(
                _obj(
                    {
                        "id": _STR,
                        "hash": _STR,
                        "installed": _BOOL,
                        "env_ok": _BOOL,
                        "determinism_tier": {"enum": ["bit_exact", "similar", None]},
                    }
                )
            ),
            "capabilities": _obj(
                {
                    "controls": _obj({"pace": _BOOL, "context": _BOOL, "instruct": _BOOL}),
                    "text_modes": _array(_STR),
                }
            ),
            "limits": {**_result_records()["limits"], "description": "the [limits] in force (section 16)"},
            "text_checks_version": _STR,
            "alignment": _obj(
                {
                    "method_id": _STR_OR_NULL,
                    "model": _STR,
                    "revision": _STR_OR_NULL,
                    "measured_error": {
                        **_c(_MEASURED_ERROR),
                        "properties": {**_c(_MEASURED_ERROR)["properties"], "by_kind": {"type": ["object", "null"]}},
                    },
                    "benchmark": {
                        "type": ["object", "null"],
                        "properties": {"id": _STR, "sha256": _STR, "description": _STR},
                    },
                    "measured_at": _STR_OR_NULL,
                }
            ),
        }
    )


def _job_handle_output(extra: dict[str, Schema] | None = None) -> Schema:
    return _output({"job_id": _STR, "status": _STR, "poll_after_s": POLL_AFTER_S, **(extra or {})})


def _check_text_output() -> Schema:
    return _output(
        {
            "voice_hash": _STR_OR_NULL,
            "measurement": {**_c(_MEASUREMENT_SUMMARY), "type": ["object", "null"]},
            "segments": _array(_SEGMENT_TEXT),
            "text_checks_version": _STR,
        }
    )


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """One tool's published schemas and behaviour hints (section 7.1)."""

    name: str
    input_schema: Schema
    output_schema: Schema
    read_only: bool
    idempotent: bool
    returns_job: bool


def _build() -> tuple[ToolSchema, ...]:
    empty = _input({})
    tools = {
        "get_server_status": ToolSchema("get_server_status", empty, _server_status_output(), True, True, False),
        "release_gpu": ToolSchema(
            "release_gpu",
            _c(empty),
            _output({"released": _BOOL, "holder_before": {"enum": ["qwen", "qa", None]}, "busy_job": _STR_OR_NULL}),
            False,
            True,
            False,
        ),
        "get_job": ToolSchema(
            "get_job",
            _input(
                {
                    "job_id": _JOB_ID,
                    "wait_s": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 55,
                        "default": 0,
                        "description": "seconds to hold the answer until the job's status changes; 0 answers at once",
                    },
                    "include_segments": {
                        "type": "boolean",
                        "default": False,
                        "description": "true: also each segment's state, takes_ok and retakes_used",
                    },
                },
                required=["job_id"],
            ),
            _get_job_output(),
            True,
            True,
            False,
        ),
        "get_results": ToolSchema(
            "get_results",
            _input(
                {
                    "job_id": _JOB_ID,
                    "include_words": {
                        "type": "boolean",
                        "default": True,
                        "description": "each cue's word times (takes[].cues[].words). They make the result much "
                        "larger: send false unless you need them; cue start_s and end_s come either way",
                    },
                    "include_transcripts": {
                        "type": "boolean",
                        "default": False,
                        "description": "true: each take's whole speech-recogniser transcript (qa.transcript)",
                    },
                },
                required=["job_id"],
            ),
            _get_results_output(),
            True,
            True,
            False,
        ),
        "cancel_job": ToolSchema(
            "cancel_job",
            _input(
                {
                    "job_id": _JOB_ID,
                    "reason": {
                        "type": "string",
                        "maxLength": 200,
                        "description": "optional: why; shown in the job's message",
                    },
                },
                required=["job_id"],
            ),
            _output({"status": {"enum": list(JOB_STATUSES)}, "completed": _BOOL}),
            False,
            True,
            False,
        ),
        "design_voice": ToolSchema(
            "design_voice",
            _input(
                {
                    "name": {"type": "string", "minLength": 1, "maxLength": 100, "description": "opaque label"},
                    "description": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 600,
                        "description": "positive-only: name the qualities wanted, not those unwanted",
                    },
                    "takes": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 4,
                        "default": 3,
                        "description": "how many candidates to design, each from its own seed",
                    },
                    "design_text": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 400,
                        "description": "optional: the words the candidates speak (default: the service's design "
                        "text); they become each clip's transcript. A long text can make a clip longer than "
                        "measure_voice takes (flag CLIP_TOO_LONG)",
                    },
                },
                required=["name", "description"],
            ),
            _job_handle_output({"design_id": _STR, "lint": _LINT}),
            False,
            False,
            True,
        ),
        "profile_voice": ToolSchema(
            "profile_voice",
            _input({"audio": _AUDIO}, required=["audio"]),
            _job_handle_output(),
            False,
            True,
            True,
        ),
        "measure_voice": ToolSchema(
            "measure_voice",
            _input({"voice": _VOICE}, required=["voice"]),
            _job_handle_output(
                {
                    "voice_hash": _STR,
                    "measurement": {
                        **_result_records()["measurement"],
                        "description": "present at once when this voice is already measured under this engine "
                        "profile; job_id is then the id of the already completed job",
                    },
                }
            ),
            False,
            True,
            True,
        ),
        "check_text": ToolSchema(
            "check_text",
            _input(
                {
                    "voice": {
                        **_VOICE,
                        "description": "optional: a measured voice, to check each segment's spoken length against "
                        "it and estimate its duration; sent exactly as it was kept with the clip",
                    },
                    "hints": _array(
                        _HINT,
                        maxItems=500,
                        description=_HINTS_NOTE + "; send the hints you will send to submit_job",
                    ),
                    "segments": _array(_SEGMENT, minItems=1, maxItems=200, description="the segments to check"),
                },
                required=["segments"],
            ),
            _check_text_output(),
            True,
            True,
            False,
        ),
        "audition_pronunciation": ToolSchema(
            "audition_pronunciation",
            _input(
                {
                    "voice": _VOICE,
                    "term": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 80,
                        "description": "the word or words as they appear in the text",
                    },
                    "variants": _array(
                        _input(
                            {
                                "label": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 40,
                                    "description": "your name for the variant, echoed back",
                                },
                                "respell": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 120,
                                    "description": "what the engine is given in place of the term",
                                },
                            },
                            required=["label", "respell"],
                        ),
                        minItems=1,
                        maxItems=4,
                        description="the respellings to hear, up to four",
                    ),
                    "carrier": {
                        "type": "string",
                        "maxLength": 400,
                        "description": "optional sentence containing the term, spoken for each variant",
                    },
                },
                required=["voice", "term", "variants"],
            ),
            _job_handle_output(),
            False,
            True,
            True,
        ),
        "submit_job": ToolSchema("submit_job", _submit_job_input(), _submit_job_output(), False, True, True),
    }
    assert tuple(tools) == TOOL_NAMES, "tools must be built in the section 7.1 order"
    for tool in tools.values():
        tool.input_schema["$schema"] = DIALECT
        tool.output_schema["$schema"] = DIALECT
    return tuple(tools.values())


TOOLS: Final[tuple[ToolSchema, ...]] = _build()
TOOLS_BY_NAME: Final[dict[str, ToolSchema]] = {t.name: t for t in TOOLS}


def tool_schema(name: str) -> ToolSchema:
    """A tool's schemas, as a fresh copy the caller may modify."""
    t = TOOLS_BY_NAME[name]
    return ToolSchema(t.name, _c(t.input_schema), _c(t.output_schema), t.read_only, t.idempotent, t.returns_job)


# The fragments, for other modules and tests (always copies).
def fragment(name: str) -> Schema:
    """A copy of a section 7.2 fragment: Id, Flag, Error, Voice, Hint, Controls, Segment, or Audio."""
    fragments = {
        "Id": _ID,
        "Flag": _FLAG,
        "Error": _ERROR,
        "Voice": _VOICE,
        "Hint": _HINT,
        "Controls": _CONTROLS,
        "Segment": _SEGMENT,
        "Audio": _AUDIO,
    }
    return _c(fragments[name])
