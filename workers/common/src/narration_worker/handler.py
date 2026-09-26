"""The base class each worker role subclasses (design Appendix A; plan.md P1 and WP16).

A role's handler implements one method per op, ``op_<name>(request) -> dict``, for every op of
``protocol.OPS_BY_ROLE[role]`` except ``hello`` (implemented here) and ``shutdown`` (the request loop
replies to it and calls ``shutdown()`` first). A method returns the reply's members other than ``id`` and
``ok``, or raises ``OpError`` to reply ``ok: false``. Any other exception becomes ``INTERNAL``, or
``GPU_OOM`` when ``classify`` recognises CUDA running out of memory.

**Contract every role keeps** (the shared contract tests check it):

- the constructor (``__init__``) is cheap and must not touch the GPU: no CUDA, no model, no torch import
  (``self.torch()`` imports torch on first use; models load in ``load``). An ``ImportError`` from it (a
  missing dependency) stops the worker with ``EXIT_START_FAILED``, which the daemon reports as
  ``BACKEND_NOT_INSTALLED`` and does not retry; any other exception crashes the worker like any other crash;
- an op that needs a model replies ``NOT_LOADED`` before ``load``, and after ``unload``, before it reads
  any file or checks its other fields;
- ``load`` checks that each snapshot directory it is given exists before anything else, and replies
  ``BACKEND_NOT_INSTALLED`` for a missing one;
- ``synthesize`` and ``design`` take a required ``max_new_tokens``, the call's own generation cap (design
  section 10.1, DC-4), and check it with ``require_max_new_tokens``: an integer from 2 to the loaded ceiling
  (``load``'s ``settings.generation.max_new_tokens``). A missing or out-of-range cap is ``INVALID_REQUEST``,
  and like every argument error it comes before ``VOICE_NOT_PREPARED``. The reply echoes the cap and
  reports ``hit_token_cap`` and ``new_tokens`` exactly as ``protocol.AudioReply`` defines them (an end token
  on the cap-th step is not a hit). The cap only truncates, so a render that ends under it is the same
  under any cap;
- a worker returns raw outputs only; every verdict, threshold and flag is the server's (plan.md P1);
- file paths in a request are absolute and inside ``<store_root>``; outputs are written only there.

The helpers below validate request members and paths, so each role reports a bad request the same way:
``INVALID_REQUEST`` with ``details.field``. A role that uses torch gets it from ``self.torch()``, which applies
the CPU thread cap once (section 4.1); its ``load`` applies the request's switches with
``determinism.apply_determinism(determinism.parse_determinism(request["determinism"]), self.torch())``, and each
render seeds with ``determinism.seed_everything(seed, torch=self.torch())`` (sections 10.1, 10.3).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Final

from .determinism import import_optional
from .errors import OpError
from .fingerprint import collect
from .protocol import OPS_BY_ROLE, PROTOCOL_VERSION, Capabilities, Controls, Fingerprint, WorkerRole
from .threads import cap_torch_threads

Request = Mapping[str, Any]
"""A decoded request: ``id``, ``op`` and the op's members."""


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkerContext:
    """What the worker was started with: its role, the store it may use, and the CPU thread cap."""

    role: WorkerRole
    store_root: Path
    cpu_threads: int


class WorkerHandler:
    """Base class of a role's handler. Subclasses set ``role`` and implement ``op_<name>`` per op."""

    role: ClassVar[WorkerRole]
    controls: ClassVar[Controls | None] = None
    """The generation controls a synthesis role supports (``hello``'s ``capabilities.controls``)."""
    fingerprint_packages: ClassVar[tuple[str, ...]] = ()
    """Distributions whose versions go in the fingerprint (``narration-worker`` is always there)."""
    uses_torch: ClassVar[bool] = True
    """Whether the role uses torch (and the fingerprint reads torch and NVML). The fake sets it False."""

    def __init__(self, context: WorkerContext) -> None:
        if context.role != self.role:
            raise ValueError(f"{type(self).__name__} serves role {self.role!r}, not {context.role!r}")
        self.context = context
        self._torch: Any | None = None

    def torch(self) -> Any | None:
        """torch, imported on first use with the CPU thread cap applied (section 4.1); None if the role does
        not use torch or it is not installed."""
        if not self.uses_torch:
            return None
        if self._torch is None:
            module = import_optional("torch")
            if module is None:
                return None
            cap_torch_threads(self.context.cpu_threads, module)
            self._torch = module
        return self._torch

    # ------------------------------------------------------------------ the framework's side
    @property
    def ops(self) -> tuple[str, ...]:
        """The ops of this handler's role."""
        return OPS_BY_ROLE[self.role]

    @classmethod
    def missing_ops(cls) -> list[str]:
        """The role's ops this class has no ``op_<name>`` method for (``hello`` and ``shutdown`` excepted)."""
        return [
            op
            for op in OPS_BY_ROLE[cls.role]
            if op not in ("hello", "shutdown") and not callable(getattr(cls, f"op_{op}", None))
        ]

    def handle(self, op: str, request: Request) -> dict[str, Any]:
        """Run one request's op; the request loop has already checked ``id`` and that ``op`` is the role's."""
        method = getattr(self, f"op_{op}", None)
        if method is None:
            raise OpError("INVALID_REQUEST", f"the {self.role} worker does not implement {op}", {"op": op})
        return method(request)

    def before_request(self, op: str, request: Request) -> bytes | None:
        """Called for every well-formed request, ``hello`` and ``shutdown`` included, before it runs.

        Real workers leave it alone. A test double may return bytes, which the loop writes to the protocol
        stream verbatim (a planted protocol break), or raise ``OpError``.
        """
        return None

    def shutdown(self) -> None:
        """Release models before the process exits (on ``shutdown`` and at the end of input)."""

    def classify(self, exc: Exception) -> OpError | None:
        """Turn an unexpected exception into a protocol error, or ``None`` for ``INTERNAL``.

        Recognises CUDA running out of memory (``torch.OutOfMemoryError``, or a ``RuntimeError`` whose
        message says so) as ``GPU_OOM``; the daemon then unloads, waits and retries once (section 4).
        """
        kind = type(exc)
        text = str(exc)
        if (
            kind.__name__ == "OutOfMemoryError" and kind.__module__.startswith("torch")
        ) or "CUDA out of memory" in text:
            return OpError(
                "GPU_OOM", text[:2000] or "CUDA out of memory", {"type": f"{kind.__module__}.{kind.__name__}"}
            )
        return None

    # ------------------------------------------------------------------ ops every role shares
    def op_hello(self, request: Request) -> dict[str, Any]:
        """The role, the protocol version, the role's ops and controls, and the fingerprint."""
        capabilities = Capabilities(ops=list(self.ops))
        if self.controls is not None:
            capabilities["controls"] = self.controls
        return {
            "role": self.role,
            "protocol": PROTOCOL_VERSION,
            "capabilities": capabilities,
            "fingerprint": self.fingerprint(),
        }

    def fingerprint(self) -> Fingerprint:
        """The fingerprint (``fingerprint.collect``), with torch and NVML facts when ``uses_torch``."""
        nvml = import_optional("pynvml") if self.uses_torch else None
        return collect(
            cpu_threads=self.context.cpu_threads, packages=self.fingerprint_packages, torch=self.torch(), nvml=nvml
        )

    # ------------------------------------------------------------------ paths inside the store
    def input_file(self, request: Request, name: str) -> Path:
        """An existing file named by a request member: absolute and inside the store.

        ``INVALID_REQUEST`` for a path that is not absolute or is outside the store; ``UNSUPPORTED_AUDIO``
        for one that is not a regular file.
        """
        path = self._store_path(request, name)
        if not path.is_file():
            raise OpError("UNSUPPORTED_AUDIO", f"{name}: no such file: {path}", {"field": name, "path": str(path)})
        return path

    def output_file(self, request: Request, name: str) -> Path:
        """A file to write, named by a request member: absolute and inside the store. Its folder is created."""
        path = self._store_path(request, name)
        if path.is_dir():
            raise OpError("INVALID_REQUEST", f"{name} names a folder, not a file: {path}", {"field": name})
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def output_dir(self, request: Request, name: str) -> Path:
        """A folder to write into, named by a request member: absolute and inside the store. It is created."""
        path = self._store_path(request, name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _store_path(self, request: Request, name: str) -> Path:
        value = require_str(request, name)
        raw = Path(value)
        if not raw.is_absolute():
            raise OpError("INVALID_REQUEST", f"{name} must be an absolute path", {"field": name})
        path = raw.resolve()
        root = self.context.store_root.resolve()
        if not path.is_relative_to(root):
            raise OpError("INVALID_REQUEST", f"{name} is outside the store", {"field": name, "path": value})
        return path


# ---------------------------------------------------------------------- request members


def _missing(name: str) -> OpError:
    return OpError("INVALID_REQUEST", f"the request has no {name}", {"field": name})


def _wrong(name: str, what: str) -> OpError:
    return OpError("INVALID_REQUEST", f"{name} must be {what}", {"field": name})


def require_str(request: Request, name: str, *, allow_empty: bool = False) -> str:
    """A string member (non-empty unless ``allow_empty``)."""
    if name not in request:
        raise _missing(name)
    value = request[name]
    if not isinstance(value, str) or (not value and not allow_empty):
        raise _wrong(name, "a string" if allow_empty else "a non-empty string")
    return value


def require_bool(request: Request, name: str) -> bool:
    """A boolean member."""
    if name not in request:
        raise _missing(name)
    value = request[name]
    if not isinstance(value, bool):
        raise _wrong(name, "true or false")
    return value


def require_int(request: Request, name: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    """An integer member (a JSON boolean is not an integer), optionally within bounds."""
    if name not in request:
        raise _missing(name)
    value = request[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise _wrong(name, "an integer")
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise _wrong(name, f"an integer in {minimum}..{maximum}")
    return value


def require_number(request: Request, name: str) -> float:
    """A finite number member."""
    if name not in request:
        raise _missing(name)
    value = request[name]
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise _wrong(name, "a finite number")
    return float(value)


def require_str_list(request: Request, name: str) -> list[str]:
    """A list-of-strings member."""
    if name not in request:
        raise _missing(name)
    value = request[name]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise _wrong(name, "a list of strings")
    return list(value)


MIN_MAX_NEW_TOKENS: Final = 2
"""The smallest call cap: qwen-tts fixes ``min_new_tokens`` at 2, so a cap of 1 could not be honoured."""


def require_max_new_tokens(request: Request, ceiling: int) -> int:
    """The ``max_new_tokens`` of a ``synthesize`` or ``design`` call: required, an integer in 2..``ceiling``.

    It is the call's own generation cap (design section 10.1, DC-4). The daemon computes it from the text the
    call speaks, ``min(ceiling, max(floor, ceil(per_char * len(text))))`` with the engine profile's
    ``max_new_tokens_per_char`` and ``max_new_tokens_floor`` (``narration.contracts.names.max_new_tokens_for``),
    so a runaway render stops early. ``ceiling`` is the loaded ``settings.generation.max_new_tokens`` (8192,
    the snapshots' value). A missing cap, one below ``MIN_MAX_NEW_TOKENS`` or one above the ceiling is
    ``INVALID_REQUEST`` with ``details.field`` ``max_new_tokens``: an audio-changing setting is never left to a
    default.
    """
    cap = require_int(request, "max_new_tokens", minimum=MIN_MAX_NEW_TOKENS)
    if cap > ceiling:
        raise OpError(
            "INVALID_REQUEST",
            f"max_new_tokens ({cap}) is above the loaded ceiling ({ceiling}); the daemon's cap is at most that",
            {"field": "max_new_tokens", "ceiling": ceiling},
        )
    return cap


def require_one_of(request: Request, name: str, choices: tuple[str, ...]) -> str:
    """A string member with one of ``choices``."""
    value = require_str(request, name)
    if value not in choices:
        raise _wrong(name, "one of " + ", ".join(choices))
    return value
