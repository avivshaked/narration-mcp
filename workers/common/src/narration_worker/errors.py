"""The worker-side error a handler raises to reply ``ok: false`` (design Appendix A)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from .protocol import WORKER_ERROR_CODES, WorkerErrorCode

EXIT_START_FAILED: Final = 2
"""The worker's exit code when it cannot start: a bad argument, a missing store, or no loadable handler for
its role. The daemon reports it as ``BACKEND_NOT_INSTALLED`` (the worker env is missing or broken, section
14) and does not respawn the worker."""


class OpError(Exception):
    """Reply ``{"ok": false, "error": {"code", "message", "details"?}}`` to the current request.

    ``code`` is one of ``protocol.WORKER_ERROR_CODES``. ``details`` must be JSON-serialisable; it carries the
    facts behind the error (the field at fault, the exception type, an alignment's frame counts).
    """

    def __init__(self, code: WorkerErrorCode, message: str, details: Mapping[str, object] | None = None) -> None:
        if code not in WORKER_ERROR_CODES:
            raise ValueError(f"{code!r} is not a worker error code; use one of {', '.join(WORKER_ERROR_CODES)}")
        super().__init__(f"{code}: {message}")
        self.code: WorkerErrorCode = code
        self.message = message
        self.details: dict[str, object] | None = dict(details) if details else None

    def to_error(self) -> dict[str, object]:
        """The ``error`` member of the reply."""
        error: dict[str, object] = {"code": self.code, "message": self.message}
        if self.details:
            error["details"] = self.details
        return error
