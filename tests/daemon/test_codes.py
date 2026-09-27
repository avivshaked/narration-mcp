"""Names are the design's names (AGENTS.md section 6; WP30 review, finding 5): the daemon takes every error
code from ``narration.contracts.codes`` and never retypes one as a string."""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path
from typing import get_args

from narration_worker.protocol import WorkerErrorCode

import narration.daemon
from narration.contracts import codes

DAEMON = Path(narration.daemon.__file__).parent
CODES = {value for name, value in vars(codes).items() if name.isupper() and isinstance(value, str)}
CODES |= set(get_args(WorkerErrorCode))


def retyped_codes(source: str) -> list[str]:
    found: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.STRING:
            continue
        try:
            value = ast.literal_eval(token.string)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, str) and value in CODES:
            found.append(f"{token.string} (line {token.start[0]})")
    return found


def test_the_daemon_retypes_no_error_code_s14() -> None:
    assert {"BACKEND_NOT_INSTALLED", "INTERNAL", "INVALID_ARGUMENT"} <= CODES
    offenders = {
        path.name: hits
        for path in sorted(DAEMON.glob("*.py"))
        if (hits := retyped_codes(path.read_text(encoding="utf-8")))
    }
    assert offenders == {}
