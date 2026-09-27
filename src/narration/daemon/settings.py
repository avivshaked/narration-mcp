"""The daemon's settings (design section 16: ``[daemon]`` and ``[workers]``).

Standard library and ``narration.config`` only: the entry point reads these before it caps the thread
pools, so nothing here may import numpy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Final

from narration.config import Config

DEFAULT_POLL_S: Final = 0.5
"""How often the daemon looks for commands, and for work when it has none."""
LOG_NAME: Final = "daemon.log"
"""The daemon's log file, in the store's ``logs`` folder: ``<store_root>/logs/daemon.log`` (rotated). The daemon
writes it, and the job engine names it in an ``INTERNAL`` error's ``details.log`` (section 14); this is the one
place the name is set."""

SAFE_PATH_FLAG: Final = "-P"
"""Python's ``-P``: do not put the working directory (for ``-m``) first on ``sys.path``. The daemon runs in the
store root, and nothing there may shadow a module it imports (section 17). The daemon's and the workers'
command lines both carry it."""
IMPORT_ENV_VARS: Final = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")
"""The ``PYTHON*`` variables that change what a Python process imports or runs at start. They are removed
from the environment the daemon and its workers start with (section 17); the thread caps, the offline flags
and the rest are kept."""


def scrub_python_env(env: Mapping[str, str]) -> dict[str, str]:
    """``env`` without ``IMPORT_ENV_VARS`` (matched without regard to case, as Windows reads names)."""
    return {name: value for name, value in env.items() if name.upper() not in IMPORT_ENV_VARS}


def isolated(argv: Sequence[str]) -> tuple[str, ...]:
    """A Python command line with ``SAFE_PATH_FLAG`` right after the interpreter (once)."""
    if len(argv) > 1 and argv[1] == SAFE_PATH_FLAG:
        return tuple(argv)
    return (argv[0], SAFE_PATH_FLAG, *argv[1:])


@dataclass(frozen=True, slots=True, kw_only=True)
class DaemonSettings:
    """What one daemon process runs with.

    ``idle_unload_s`` and ``idle_exit_s`` come from ``[daemon] idle_unload_s`` and ``idle_exit_min``
    (section 4: 120 s and 15 min by default); ``cpu_threads`` and ``below_normal`` from ``[workers]``.
    The rest are the daemon's own:

    - ``poll_s``: how often it looks for commands, and for work when it has none;
    - ``fake_workers``: every worker group runs the ``fake`` role (no model, no GPU), for development and
      tests;
    - ``worker_close_s``: how long a worker gets to exit after ``shutdown`` before it is killed;
    - ``stop_now_grace_s``: how long ``stop_now`` waits for the runner to give its job back after the
      workers are killed;
    - ``takeover_wait_s``: how long a daemon that finds its store's singleton held waits, while the holder
      says it is ``stopping``, before it gives up;
    - ``command_wait_s``: how long ``release_gpu`` waits for the runner to finish a step that holds no job.
    """

    store_root: Path
    idle_unload_s: float = 120.0
    idle_exit_s: float = 900.0
    poll_s: float = DEFAULT_POLL_S
    cpu_threads: int = 8
    below_normal: bool = True
    fake_workers: bool = False
    worker_close_s: float = 10.0
    stop_now_grace_s: float = 30.0
    takeover_wait_s: float = 60.0
    command_wait_s: float = 5.0

    def __post_init__(self) -> None:
        for spec in fields(self):
            value = getattr(self, spec.name)
            if spec.name in ("store_root", "fake_workers", "below_normal"):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{spec.name} must be a finite number, not {value!r}")
            if value < 0:
                raise ValueError(f"{spec.name} must not be negative, not {value!r}")
        if self.poll_s <= 0:
            raise ValueError(f"poll_s must be positive, not {self.poll_s!r}")
        if self.cpu_threads < 1:
            raise ValueError(f"cpu_threads must be at least 1, not {self.cpu_threads!r}")

    @classmethod
    def from_config(cls, config: Config, **overrides: Any) -> DaemonSettings:
        """The settings ``config`` gives, with every override that is not None applied on top."""
        values: dict[str, Any] = {
            "store_root": config.server.store_root,
            "idle_unload_s": float(config.daemon.idle_unload_s),
            "idle_exit_s": 60.0 * config.daemon.idle_exit_min,
            "cpu_threads": config.workers.cpu_threads,
            "below_normal": config.workers.priority == "below_normal",
        }
        values.update({name: value for name, value in overrides.items() if value is not None})
        return cls(**values)
