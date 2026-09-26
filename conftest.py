"""Repository-wide pytest set-up.

Tests write only inside the checkout they run in (AGENTS.md rule 3): pytest's base temporary directory is
``<checkout>/.pytest-tmp`` (gitignored), and so is Python's ``tempfile`` directory, for this process and for
any subprocess a test starts.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent


def pytest_configure(config: pytest.Config) -> None:
    if config.option.basetemp is None:
        config.option.basetemp = ROOT / ".pytest-tmp" / f"run-{os.getpid()}"
    scratch = ROOT / ".pytest-tmp" / "tempfile"
    scratch.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(scratch)
    for var in ("TMP", "TEMP", "TMPDIR"):
        os.environ[var] = str(scratch)
