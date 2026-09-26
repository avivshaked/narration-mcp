"""Fixtures for the QA tests.

``bakeoff_root`` is the bake-off's local evidence, read only, for tests marked ``evidence``. It skips the test
cleanly when ``NARRATION_BAKEOFF_ROOT`` is unset or names no folder, so the default suite never needs it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def bakeoff_root() -> Path:
    value = os.environ.get("NARRATION_BAKEOFF_ROOT")
    if not value:
        pytest.skip("NARRATION_BAKEOFF_ROOT is not set, so the bake-off's evidence is not available")
    root = Path(value)
    if not root.is_dir():
        pytest.skip("NARRATION_BAKEOFF_ROOT does not name a folder")
    return root
