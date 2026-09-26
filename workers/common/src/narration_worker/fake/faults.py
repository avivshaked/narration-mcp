"""Planted faults for the fake worker, so later work can test how the service handles them (WP31, WP40).

**Control.** The fake reads a fault spec from the JSON file named by the environment variable
``NARRATION_FAKE_SPEC``, and reads it again whenever the file changes. Only the environment the fake is
started with can name it: nothing in a request, the MCP surface or the store does. Without the variable
the fake plants nothing.

**The spec**::

    {
      "faults": [
        {"kind": "crash", "op": "synthesize", "when": {"text_contains": "harbour"}, "times": 1},
        {"kind": "wrong_word", "word": 2, "replacement": "barber"},
        {"kind": "hang", "op": "transcribe"}
      ],
      "transcripts": {"<sha256 of a file>": "the words in it"}
    }

Each fault has a ``kind``; optionally ``op`` (one op, a list, or ``"*"`` for every op), ``when`` (all of
``text_contains``, ``seed``, ``voice_hash`` given must hold), ``times`` (fire at most this many times across
every fake process that shares the store; unlimited when absent), and the kind's own parameters:

========================  ================  ==============================================================
kind                      default op        effect (parameters)
========================  ================  ==============================================================
``crash``                 ``synthesize``    the process exits mid-request without replying (``exit_code``, 3)
``hang``                  ``synthesize``    the request never gets a reply (for timeout tests)
``delay``                 ``synthesize``    the reply comes ``seconds`` late
``gpu_oom``               ``synthesize``    ``ok: false`` with ``GPU_OOM``
``error``                 ``synthesize``    ``ok: false`` with ``code`` (a worker error code) and ``message``
``alignment_error``       ``align``         ``ok: false`` with ``ALIGNMENT_ERROR``
``protocol_break``        ``synthesize``    a line that is not a protocol message on stdout
``stdout_noise``          ``synthesize``    prints to ``sys.stdout``, file descriptor 1 and a child
                                            process's stdout; none of it may reach the protocol stream
``say``                   ``synthesize``    the take says ``text`` instead, and ``transcribe`` hears ``heard``
                                            (``text`` by default) spread over it; this drives the QA fixture
                                            specs (``material/fixtures/qa-faults-v1``: render and asr text)
``wrong_word``            ``synthesize``    the take says ``replacement`` ("wrong") for word ``word``
                                            (index into the text's words; the middle one by default)
``head_insertion``        ``synthesize``    the take starts with extra ``words`` (["so"])
``end_insertion``         ``synthesize``    the take ends with extra ``words`` (["okay"])
``token_cap``             ``synthesize``    generation stops at ``max_new_tokens`` (60 % of the take's
                                            tokens by default): ``hit_token_cap`` and a cut take
========================  ================  ==============================================================

The last five change what the take says, so they apply to ``synthesize`` and ``design`` only; the fake's
QA ops then hear the planted words. ``when`` for a QA op matches the utterance the fake recognises in the
audio. ``transcripts`` gives the transcript of a file the fake did not render (by its sha256).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from narration_worker.protocol import FAKE_OPS, WORKER_ERROR_CODES

from .registry import fake_dir

SPEC_ENV: Final = "NARRATION_FAKE_SPEC"

REQUEST_KINDS: Final = (
    "crash",
    "hang",
    "delay",
    "gpu_oom",
    "error",
    "alignment_error",
    "protocol_break",
    "stdout_noise",
)
CONTENT_KINDS: Final = ("say", "wrong_word", "head_insertion", "end_insertion", "token_cap")
CONTENT_OPS: Final = ("synthesize", "design")

_PARAMS: Final[dict[str, dict[str, str]]] = {
    "crash": {"exit_code": "int"},
    "hang": {},
    "delay": {"seconds": "number"},
    "gpu_oom": {},
    "error": {"code": "code", "message": "str"},
    "alignment_error": {},
    "protocol_break": {},
    "stdout_noise": {},
    "say": {"text": "str", "heard": "str"},
    "wrong_word": {"word": "int", "replacement": "str"},
    "head_insertion": {"words": "words"},
    "end_insertion": {"words": "words"},
    "token_cap": {"max_new_tokens": "int"},
}
_REQUIRED: Final[dict[str, tuple[str, ...]]] = {"delay": ("seconds",), "error": ("code",), "say": ("text",)}
_WHEN: Final[dict[str, str]] = {"text_contains": "str", "seed": "int", "voice_hash": "str"}
_COMMON: Final = frozenset({"kind", "op", "when", "times"})


class SpecError(ValueError):
    """The fault spec is not valid; the message names the entry and the key."""


@dataclass(frozen=True, slots=True)
class Facts:
    """What a fault's ``when`` is matched against: the text, seed and voice of the request's utterance."""

    text: str | None = None
    seed: int | None = None
    voice_hash: str | None = None


@dataclass(frozen=True, slots=True)
class Fault:
    """One entry of the spec. ``key`` names it for its ``times`` counter."""

    kind: str
    ops: tuple[str, ...]
    when: tuple[tuple[str, object], ...]
    times: int | None
    params: Mapping[str, Any]
    key: str

    def applies_to(self, op: str) -> bool:
        return "*" in self.ops or op in self.ops

    def matches(self, facts: Facts) -> bool:
        for name, expected in self.when:
            if name == "text_contains":
                if facts.text is None or not isinstance(expected, str) or expected not in facts.text:
                    return False
            elif getattr(facts, name) != expected:
                return False
        return True


@dataclass(frozen=True, slots=True)
class FaultSpec:
    """The parsed spec."""

    faults: tuple[Fault, ...] = ()
    transcripts: Mapping[str, str] = field(default_factory=dict)

    def for_op(self, op: str, kinds: tuple[str, ...]) -> list[Fault]:
        return [f for f in self.faults if f.kind in kinds and f.applies_to(op)]

    def needs_facts(self, op: str) -> bool:
        return any(f.when for f in self.faults if f.applies_to(op))


def parse_spec(data: object) -> FaultSpec:
    """Validate a decoded spec; raises ``SpecError``."""
    if not isinstance(data, dict):
        raise SpecError("the fake spec must be a JSON object")
    unknown = sorted(set(data) - {"faults", "transcripts"})
    if unknown:
        raise SpecError(f"the fake spec has unknown keys: {', '.join(unknown)}")
    raw_faults = data.get("faults", [])
    if not isinstance(raw_faults, list):
        raise SpecError("faults must be a list")
    faults = tuple(_parse_fault(i, entry) for i, entry in enumerate(raw_faults))
    transcripts = data.get("transcripts", {})
    if not isinstance(transcripts, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in transcripts.items()
    ):
        raise SpecError("transcripts must map sha256 strings to text")
    return FaultSpec(faults=faults, transcripts={k.lower(): v for k, v in transcripts.items()})


def _parse_fault(index: int, entry: object) -> Fault:
    where = f"faults[{index}]"
    if not isinstance(entry, dict):
        raise SpecError(f"{where} must be an object")
    kind = entry.get("kind")
    if kind not in _PARAMS:
        raise SpecError(f"{where}.kind must be one of {', '.join(_PARAMS)}")
    unknown = sorted(set(entry) - _COMMON - set(_PARAMS[kind]))
    if unknown:
        raise SpecError(f"{where} ({kind}) has unknown keys: {', '.join(unknown)}")
    for name in _REQUIRED.get(kind, ()):
        if name not in entry:
            raise SpecError(f"{where} ({kind}) needs {name}")
    ops = _parse_ops(where, kind, entry.get("op"))
    when = entry.get("when", {})
    if not isinstance(when, dict) or not set(when) <= set(_WHEN):
        raise SpecError(f"{where}.when may have only {', '.join(_WHEN)}")
    for name, value in when.items():
        _check(f"{where}.when.{name}", _WHEN[name], value)
    times = entry.get("times")
    if times is not None:
        _check(f"{where}.times", "int", times)
        if times < 1:
            raise SpecError(f"{where}.times must be at least 1")
    params = {name: entry[name] for name in _PARAMS[kind] if name in entry}
    for name, value in params.items():
        _check(f"{where}.{name}", _PARAMS[kind][name], value)
    key = hashlib.sha256(json.dumps(entry, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    return Fault(
        kind=kind,
        ops=ops,
        when=tuple(sorted(when.items())),
        times=times,
        params=params,
        key=f"{index:03d}-{key}",
    )


def _parse_ops(where: str, kind: str, value: object) -> tuple[str, ...]:
    if value is None:
        return ("align",) if kind == "alignment_error" else ("synthesize",)
    ops = (value,) if isinstance(value, str) else value
    if not isinstance(ops, list | tuple) or not ops or not all(isinstance(op, str) for op in ops):
        raise SpecError(f"{where}.op must be an op name, a list of them, or '*'")
    for op in ops:
        if op != "*" and op not in FAKE_OPS:
            raise SpecError(f"{where}.op: {op!r} is not an op of the fake worker")
    if kind in CONTENT_KINDS and not set(ops) <= set(CONTENT_OPS):
        raise SpecError(f"{where}: {kind} changes what a take says, so its op must be synthesize or design")
    return tuple(ops)


def _check(where: str, kind: str, value: object) -> None:
    if kind == "int":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "number":
        ok = isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    elif kind == "str":
        ok = isinstance(value, str)
    elif kind == "code":
        ok = value in WORKER_ERROR_CODES
    else:  # words
        ok = isinstance(value, list) and bool(value) and all(isinstance(w, str) and w.strip() for w in value)
    if not ok:
        raise SpecError(f"{where} has the wrong type ({kind} expected)")


class SpecSource:
    """Reads the spec file named by ``NARRATION_FAKE_SPEC``, again whenever the file changes."""

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        value = (os.environ if environ is None else environ).get(SPEC_ENV, "")
        self.path = Path(value) if value else None
        self._stamp: tuple[int, int] | None = None
        self._spec = FaultSpec()

    def current(self) -> FaultSpec:
        """The spec in force; raises ``SpecError`` for a missing or invalid file."""
        if self.path is None:
            return self._spec
        try:
            stat = self.path.stat()
        except OSError as exc:
            raise SpecError(f"{SPEC_ENV} names {self.path}, which cannot be read: {exc}") from exc
        stamp = (stat.st_mtime_ns, stat.st_size)
        if stamp != self._stamp:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise SpecError(f"{self.path} is not a readable JSON file: {exc}") from exc
            self._spec = parse_spec(data)
            self._stamp = stamp
        return self._spec


class Tally:
    """How often each fault with ``times`` has fired, shared by every fake process on the store.

    Firing claims a marker file created exclusively, so a crash fault with ``times: 1`` does not crash the
    worker the daemon starts next.
    """

    def __init__(self, store_root: Path) -> None:
        self.root = fake_dir(store_root) / "fired"

    def claim(self, fault: Fault) -> bool:
        """True if the fault may fire now (and records the firing)."""
        if fault.times is None:
            return True
        self.root.mkdir(parents=True, exist_ok=True)
        for n in range(fault.times):
            try:
                fd = os.open(self.root / f"{fault.key}.{n}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            os.close(fd)
            return True
        return False
