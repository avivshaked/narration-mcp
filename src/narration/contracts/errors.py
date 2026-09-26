"""Exceptions that cross work-package seams.

``NarrationError`` carries a tool error (design section 14): the front-end turns it into ``isError: true``
with ``structuredContent: {"error": Error}``. The worker exceptions are what ``WorkerClient`` raises
(Appendix A); the daemon turns them into flags or job errors.
"""

from __future__ import annotations

from typing import Any

from . import codes
from .models import Error


class NarrationError(Exception):
    """A tool execution error with a code from section 14.

    ``retryable`` defaults to the code's table value (False for ``INTERNAL`` unless given); ``hint``
    defaults to the code's default hint. ``retry_after_s`` (DC-2) should be set on every retryable error.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        field: str | None = None,
        hint: str | None = None,
        details: dict[str, Any] | None = None,
        retryable: bool | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        spec = codes.error_code(code)
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.field = field
        self.hint = hint if hint is not None else spec.hint
        self.details = details
        self.retryable = retryable if retryable is not None else bool(spec.retryable)
        self.retry_after_s = retry_after_s

    @property
    def error(self) -> Error:
        """The Error record (section 7.2) for ``structuredContent``."""
        return Error(
            code=self.code,
            message=self.message,
            retryable=self.retryable,
            hint=self.hint,
            field=self.field,
            details=self.details,
            retry_after_s=self.retry_after_s,
        )


class UnsupportedPlatform(NarrationError):
    """An OS-specific operation on a platform v1 does not support (plan.md Q2: Windows first).

    ``narration.platform`` raises it from every method on any other OS. The daemon cannot run there, so the
    code is ``DAEMON_UNAVAILABLE``; it is not retryable, and ``narration-admin doctor`` reports the same.
    """

    HINT = (
        "narration-mcp v1 runs on Windows only; other platforms are planned. "
        "Run `narration-admin doctor` for what this machine supports."
    )

    def __init__(self, operation: str, platform: str) -> None:
        super().__init__(
            "DAEMON_UNAVAILABLE",
            f"{operation} is not supported on {platform}",
            hint=self.HINT,
            details={"operation": operation, "platform": platform},
            retryable=False,
        )
        self.operation = operation
        self.platform = platform


class ConfigError(ValueError):
    """The configuration file is missing, malformed, or has an unknown key or a value out of range."""


class WorkerFailure(Exception):
    """A worker replied ``ok: false`` (Appendix A). ``code`` is one of ``worker.WORKER_ERROR_CODES``."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = details or {}


class WorkerCrashed(Exception):
    """The worker process exited, or wrote something that is not a protocol message (``WORKER_CRASHED``)."""

    def __init__(self, message: str, *, exit_code: int | None = None, stderr_tail: str = "") -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr_tail = stderr_tail


class WorkerTimeout(Exception):
    """A request got no reply within its timeout; the client has stopped the worker."""
