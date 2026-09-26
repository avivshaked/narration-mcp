"""Tool results: ``structuredContent``, its text copy, ``resource_link`` items, and tool errors (sections 5, 14).

A successful call returns the backend's structured result, a JSON text copy of it for clients that read only
``content``, and a ``file:///`` ``resource_link`` for every file the result names (audio, pictures, the
report); audio is never inlined. A tool error is ``isError: true`` with ``structuredContent: {"error":
Error}`` and a text copy. Both must validate against the tool's ``outputSchema``, because the SDK client
checks ``structuredContent`` against it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Final

import mcp_types as types
from jsonschema import Draft202012Validator

from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.models import Error
from narration.contracts.serial import to_json

from .validation import field_path

LINKED_KEYS: Final = frozenset({"path", "report_md"})
"""Keys whose string value is a file the caller may open: a take's delivery, a clip, a measurement's JSON,
the job report. Pictures are linked from inside a ``pictures`` object."""
PICTURES_KEY: Final = "pictures"
MIME_TYPES: Final = {
    ".wav": "audio/wav",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".md": "text/markdown",
    ".json": "application/json",
}


def _absolute(value: str) -> PurePath | None:
    """The path as an absolute pure path of its own flavour (a Windows path is absolute on Linux too)."""
    for flavour in (PureWindowsPath, PurePosixPath):
        path = flavour(value)
        if path.is_absolute():
            return path
    return None


def _file_values(value: Any, *, in_pictures: bool = False) -> Iterator[str]:
    """Every file path in a result, in document order."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(item, str) and (in_pictures or key in LINKED_KEYS):
                yield item
            elif isinstance(item, (Mapping, list)):
                yield from _file_values(item, in_pictures=key == PICTURES_KEY)
    elif isinstance(value, list):
        for item in value:
            yield from _file_values(item)


def resource_links(structured: Mapping[str, Any]) -> list[types.ResourceLink]:
    """A ``file:///`` link for every absolute file path the result names, once each (section 5)."""
    links: list[types.ResourceLink] = []
    seen: set[str] = set()
    for value in _file_values(structured):
        path = _absolute(value)
        if path is None:
            continue
        uri = path.as_uri()
        if uri in seen:
            continue
        seen.add(uri)
        name = "/".join(path.parts[-2:]) if len(path.parts) > 2 else path.name
        links.append(
            types.ResourceLink(type="resource_link", uri=uri, name=name, mime_type=MIME_TYPES.get(path.suffix.lower()))
        )
    return links


def _text(structured: Mapping[str, Any]) -> types.TextContent:
    return types.TextContent(type="text", text=json.dumps(structured, ensure_ascii=False))


def success(structured: dict[str, Any]) -> types.CallToolResult:
    """A successful call: ``structuredContent``, its text copy, then a link for every file it names."""
    content: list[types.ContentBlock] = [_text(structured), *resource_links(structured)]
    return types.CallToolResult(content=content, structured_content=structured, is_error=False)


def tool_error(error: Error) -> types.CallToolResult:
    """A tool execution error: ``isError: true`` with ``structuredContent: {"error": Error}`` and a text copy."""
    structured = {"error": to_json(error)}
    return types.CallToolResult(content=[_text(structured)], structured_content=structured, is_error=True)


def from_narration_error(exc: NarrationError) -> types.CallToolResult:
    """The tool error for a ``NarrationError`` the backend or the validator raised."""
    return tool_error(exc.error)


def internal_error(tool: str, exc: BaseException, *, log_path: Path | None) -> Error:
    """The ``INTERNAL`` error for an exception inside a tool call (ADR 0001; section 14).

    The message names the exception's type only: its text may carry the caller's values or local paths,
    and it goes to the log instead, whose path is in ``details.log``.
    """
    details: dict[str, Any] = {"tool": tool, "exception": type(exc).__name__}
    if log_path is not None:
        details["log"] = str(log_path)
    return Error(
        code=codes.INTERNAL,
        message=f"{tool} failed inside the service ({type(exc).__name__})",
        retryable=False,
        hint=codes.error_code(codes.INTERNAL).hint,
        details=details,
    )


def output_failures(validator: Draft202012Validator, structured: Mapping[str, Any]) -> list[dict[str, str]]:
    """Where a result breaks its tool's ``outputSchema``: each failure's path and rule, never its value."""
    return [
        {"path": field_path(e.absolute_path) or "$", "rule": str(e.validator)}
        for e in validator.iter_errors(dict(structured))
    ]


def output_mismatch_error(tool: str, failures: list[dict[str, str]], *, log_path: Path | None) -> Error:
    """The ``INTERNAL`` error for a backend result that does not match the tool's ``outputSchema``."""
    details: dict[str, Any] = {"tool": tool, "output_schema_failures": failures[:20]}
    if log_path is not None:
        details["log"] = str(log_path)
    return Error(
        code=codes.INTERNAL,
        message=f"{tool} produced a result that does not match its outputSchema",
        retryable=False,
        hint=codes.error_code(codes.INTERNAL).hint,
        details=details,
    )
