"""The daemon holds no OS-specific code (AGENTS.md section 6; WP30 review, finding 6): what differs by OS
is asked of ``narration.platform`` (``ProcessPlatform``), which the test platform
(``narration.platform.testing``) also implements."""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

import narration.daemon

DAEMON = Path(narration.daemon.__file__).parent
OS_NAMES = {("os", "name"), ("sys", "platform")}
OS_LITERALS = {
    "CREATE_NO_WINDOW",
    "BELOW_NORMAL_PRIORITY_CLASS",
    "NoDefaultCurrentDirectoryInExePath",
    "python.exe",
    "pythonw.exe",
    "nt",
    "win32",
}


def os_specific(source: str) -> list[str]:
    """``os.name``/``sys.platform`` tests and Windows names used as values (strings in docstrings are prose)."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if (node.value.id, node.attr) in OS_NAMES:
                found.append(f"{node.value.id}.{node.attr} (line {node.lineno})")
            if node.attr in ("CREATE_NO_WINDOW", "BELOW_NORMAL_PRIORITY_CLASS"):
                found.append(f"{node.value.id}.{node.attr} (line {node.lineno})")
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.STRING:
            try:
                value = ast.literal_eval(token.string)
            except (ValueError, SyntaxError):
                continue
            if isinstance(value, str) and value in OS_LITERALS:
                found.append(f"{token.string} (line {token.start[0]})")
    return found


def test_the_daemon_asks_the_platform_for_everything_that_differs_by_os_s6() -> None:
    offenders = {
        path.name: hits
        for path in sorted(DAEMON.glob("*.py"))
        if (hits := os_specific(path.read_text(encoding="utf-8")))
    }
    assert offenders == {}
