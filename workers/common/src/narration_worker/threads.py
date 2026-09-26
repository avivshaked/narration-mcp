"""The CPU thread cap (design section 4.1).

``[workers] cpu_threads`` (default 8) caps every worker. The daemon puts the cap in the worker's environment
(``thread_env``), and the worker applies it again at start-up, before any native library loads, and to
torch once torch is imported (``cap_torch_threads``). torch is never imported here: a handler passes its
module in, so this package imports without torch.
"""

from __future__ import annotations

import logging
import os
from collections.abc import MutableMapping
from typing import Any, Final

log = logging.getLogger(__name__)

DEFAULT_CPU_THREADS: Final = 8
"""The design's default for ``[workers] cpu_threads`` (section 16)."""

THREAD_ENV_VARS: Final = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
"""The variables through which the native thread pools (OpenMP, MKL, OpenBLAS, numexpr, Accelerate) read
their size. They are read once, when the library loads, so they must be set before that."""

MAX_INTEROP_THREADS: Final = 4
"""``torch.set_num_interop_threads(min(n, 4))`` (section 4.1)."""


def thread_env(cpu_threads: int) -> dict[str, str]:
    """The environment variables that cap native thread pools at ``cpu_threads``."""
    if cpu_threads < 1:
        raise ValueError(f"cpu_threads must be at least 1, not {cpu_threads}")
    return dict.fromkeys(THREAD_ENV_VARS, str(cpu_threads))


def cap_threads_env(cpu_threads: int, environ: MutableMapping[str, str] | None = None) -> None:
    """Set the thread variables in this process's environment (call before importing numpy or torch)."""
    (os.environ if environ is None else environ).update(thread_env(cpu_threads))


def cap_torch_threads(cpu_threads: int, torch: Any) -> dict[str, int | None]:
    """Apply the cap to an imported torch module; returns the intra-op and inter-op sizes now in force.

    ``set_num_interop_threads`` may be called only once, before any inter-op work; if torch refuses it, the
    refusal is logged and the size torch reports is returned.
    """
    torch.set_num_threads(cpu_threads)
    interop = min(cpu_threads, MAX_INTEROP_THREADS)
    try:
        torch.set_num_interop_threads(interop)
    except RuntimeError as exc:
        log.warning("torch kept its inter-op thread count: %s", exc)
    return {
        "intra_op": _int_or_none(torch.get_num_threads()),
        "inter_op": _int_or_none(torch.get_num_interop_threads()),
    }


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None
