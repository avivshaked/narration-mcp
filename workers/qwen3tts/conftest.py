"""pytest set-up for the qwen3tts worker's tests, run from this project's own venv.

Tests write only inside the checkout (AGENTS.md rule 3): pytest's base temporary directory is
``<checkout>/.pytest-tmp`` (gitignored), and so is Python's ``tempfile`` directory, for this process and for
any subprocess a test starts. The repository root's conftest does the same for the server's suite; this
project has its own rootdir, so it needs its own.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

CHECKOUT = Path(__file__).resolve().parents[2]


def pytest_configure(config: pytest.Config) -> None:
    if config.option.basetemp is None:
        config.option.basetemp = CHECKOUT / ".pytest-tmp" / f"qwen3tts-{os.getpid()}"
    scratch = CHECKOUT / ".pytest-tmp" / "tempfile"
    scratch.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(scratch)
    for var in ("TMP", "TEMP", "TMPDIR"):
        os.environ[var] = str(scratch)
