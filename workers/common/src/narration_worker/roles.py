"""Which handler class serves a role.

- ``fake`` is built in (``narration_worker.fake``).
- ``qwen3`` and ``qa`` live in their worker projects (``workers/qwen3tts``, ``workers/qa``), which register
  their handler class as an entry point in the group ``narration_worker.roles``, named by the role::

      [project.entry-points."narration_worker.roles"]
      qwen3 = "narration_qwen3tts.worker:Qwen3Handler"

- ``--handler module:Class`` names a class directly (development and tests).

A handler module that cannot be imported, whether it raises ``ImportError`` or ``OSError`` (on Windows, a
native library such as torch's DLLs failing to load), is ``RoleUnavailable``: the worker's env is broken.
A transient load failure (``errors.is_transient_load_error``: a file another process holds) propagates
instead, so the worker crashes and a retry can get past it.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from typing import Final

from .errors import is_transient_load_error
from .handler import WorkerHandler
from .protocol import WorkerRole

ENTRY_POINT_GROUP: Final = "narration_worker.roles"
BUILTIN_HANDLERS: Final[dict[str, str]] = {"fake": "narration_worker.fake:FakeHandler"}


class RoleUnavailable(LookupError):
    """No handler class can be found or loaded for a role."""


def resolve_handler(role: WorkerRole, spec: str | None = None) -> type[WorkerHandler]:
    """The handler class for ``role``: ``spec`` if given, else the built-in, else the entry point."""
    target = spec or BUILTIN_HANDLERS.get(role) or _entry_point(role)
    if target is None:
        raise RoleUnavailable(
            f"no handler is installed for the {role!r} worker role: sync that worker's project "
            f"(its venv registers one under the entry point group {ENTRY_POINT_GROUP!r}), "
            "or pass --handler module:Class"
        )
    module_name, _, attr = target.partition(":")
    if not module_name or not attr:
        raise RoleUnavailable(f"{target!r} is not of the form module:Class")
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, attr)
    except (ImportError, AttributeError, OSError) as exc:
        if is_transient_load_error(exc):
            raise
        raise RoleUnavailable(f"cannot load the {role!r} handler {target!r}: {type(exc).__name__}: {exc}") from exc
    if not (isinstance(cls, type) and issubclass(cls, WorkerHandler)):
        raise RoleUnavailable(f"{target!r} is not a WorkerHandler subclass")
    if cls.role != role:
        raise RoleUnavailable(f"{target!r} serves the {cls.role!r} role, not {role!r}")
    missing = cls.missing_ops()
    if missing:
        raise RoleUnavailable(f"{target!r} does not implement: {', '.join(missing)}")
    return cls


def _entry_point(role: str) -> str | None:
    for entry in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        if entry.name == role:
            return entry.value
    return None
