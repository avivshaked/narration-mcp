"""Set-up for the QA worker's tests, which run from the worker's own venv (``workers/qa``).

- ADAPTER: ``workers/qa`` is not an installed package yet (``package = false``; WP22 decides its packaging,
  see ``status/WP16.md``), so its ``src`` folder is put on ``sys.path`` here. Remove this once it is one.
- Tests write only inside the checkout (AGENTS.md rule 3): pytest's base temporary directory and Python's
  ``tempfile`` directory are under ``<checkout>/.pytest-tmp``, as the repository's own ``conftest.py`` does.
- The CPU thread cap (design section 4.1) and the offline switches are set before torch or transformers
  load, and no test may see a GPU: the aligner runs on the CPU only.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

QA_ROOT = Path(__file__).resolve().parents[1]
CHECKOUT = QA_ROOT.parents[1]
THREADS = "4"

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = THREADS
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(QA_ROOT / "src"))


def pytest_configure(config: pytest.Config) -> None:
    if config.option.basetemp is None:
        config.option.basetemp = CHECKOUT / ".pytest-tmp" / f"qa-run-{os.getpid()}"
    scratch = CHECKOUT / ".pytest-tmp" / "tempfile"
    scratch.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(scratch)
    for var in ("TMP", "TEMP", "TMPDIR"):
        os.environ[var] = str(scratch)
