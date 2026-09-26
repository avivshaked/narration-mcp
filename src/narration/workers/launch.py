"""How the daemon starts a worker: its command line and its environment (design sections 4, 4.1, 10.1).

A worker runs as its venv's own Python, never through uv or a generated launcher::

    <python> -m narration_worker --role <role> --store <store_root> --cpu-threads <n>

which is also its process-identity marker (section 4.1). The ``fake`` role runs with the server's own
interpreter, which carries ``narration_worker``.

The environment is the daemon's, with:

- ``[workers] env`` added;
- ``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1`` always, whatever the config says (section 4);
- ``CUBLAS_WORKSPACE_CONFIG`` always set before CUDA starts (section 10.1): to the engine profile's pinned
  value when the caller passes it, else to ``[workers] env``'s, else to ``:4096:8``; never to whatever the
  daemon happened to inherit, and an operator's ``env`` table that leaves it out cannot drop it;
- the thread-pool variables capped at ``[workers] cpu_threads`` (section 4.1);
- UTF-8 I/O and unbuffered stderr, so a crash's last log lines reach the daemon;
- and without the variables that would let the server's Python leak into a worker venv (``PYTHONPATH``,
  ``PYTHONHOME``, ``VIRTUAL_ENV``, ``PYTHONSTARTUP``, ``PYTHONSAFEPATH``).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from narration_worker.determinism import CUBLAS_WORKSPACE_CONFIG, CUBLAS_WORKSPACE_CONFIG_VAR
from narration_worker.threads import thread_env

from narration.config import Config, WorkersConfig
from narration.contracts.errors import WorkerFailure
from narration.contracts.names import WorkerRole

WORKER_PROJECT_DIRS: Final[dict[str, str]] = {"qwen3": "qwen3tts", "qa": "qa"}
"""Each real role's uv project under ``<service_root>/workers/`` (section 4)."""
FORCED_ENV: Final = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
DROPPED_ENV: Final = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONSTARTUP", "PYTHONSAFEPATH")
IO_ENV: Final = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
INSTALL_HINT: Final = "sync the worker's venv (narration-admin install)"
"""What to do when a worker's environment is missing or broken (``BACKEND_NOT_INSTALLED``)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkerCommand:
    """What starts one worker: its argument list (never a shell string) and its whole environment."""

    role: WorkerRole
    argv: tuple[str, ...]
    env: Mapping[str, str] = field(repr=False)


def venv_python(project: Path) -> Path:
    """The Python of a uv project's ``.venv`` (``Scripts\\python.exe`` on Windows, ``bin/python`` elsewhere)."""
    return project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def worker_project(config: Config, role: WorkerRole) -> Path | None:
    """The worker's uv project: ``[workers.<role>] project``, else ``<service_root>/workers/<dir>``."""
    configured = config.workers.qwen3 if role == "qwen3" else config.workers.qa if role == "qa" else None
    if configured is not None:
        return configured.project
    if role in WORKER_PROJECT_DIRS and config.service_root is not None:
        return config.service_root / "workers" / WORKER_PROJECT_DIRS[role]
    return None


def worker_env(
    workers: WorkersConfig, *, base: Mapping[str, str] | None = None, cublas_workspace_config: str | None = None
) -> dict[str, str]:
    """The environment a worker starts with (see the module docstring). ``base`` defaults to ours.

    ``cublas_workspace_config`` is the engine profile's pinned value (``determinism.cublas_workspace_config``,
    section 10.1); when it is given, it wins over ``[workers] env``.
    """
    env = {
        k: v
        for k, v in (os.environ if base is None else base).items()
        if k not in DROPPED_ENV and k != CUBLAS_WORKSPACE_CONFIG_VAR
    }
    env.update(workers.env)
    env.update(FORCED_ENV)
    pinned = cublas_workspace_config or workers.env.get(CUBLAS_WORKSPACE_CONFIG_VAR) or CUBLAS_WORKSPACE_CONFIG
    env[CUBLAS_WORKSPACE_CONFIG_VAR] = pinned
    env.update(thread_env(workers.cpu_threads))
    env.update(IO_ENV)
    return env


def worker_command(
    config: Config,
    role: WorkerRole,
    *,
    python: Path | None = None,
    base_env: Mapping[str, str] | None = None,
    cublas_workspace_config: str | None = None,
) -> WorkerCommand:
    """The command that starts ``role`` for this configuration.

    ``cublas_workspace_config`` is the engine profile's pinned value, for a worker that will load one (see
    ``worker_env``).

    Raises ``WorkerFailure`` (``BACKEND_NOT_INSTALLED``) when a real role's worker venv cannot be found.
    """
    if python is None:
        if role == "fake":
            python = Path(sys.executable)
        else:
            project = worker_project(config, role)
            if project is None:
                raise WorkerFailure(
                    "BACKEND_NOT_INSTALLED",
                    f"no project is configured for the {role} worker: set [workers.{role}] project, then "
                    f"{INSTALL_HINT}",
                    {"role": role},
                )
            python = venv_python(project)
    if not python.is_file():
        raise WorkerFailure(
            "BACKEND_NOT_INSTALLED",
            f"the {role} worker's Python is missing at {python}: {INSTALL_HINT}",
            {"role": role, "python": str(python)},
        )
    store = config.server.store_root.resolve()
    argv = (
        str(python),
        "-m",
        "narration_worker",
        "--role",
        role,
        "--store",
        str(store),
        "--cpu-threads",
        str(config.workers.cpu_threads),
    )
    env = worker_env(config.workers, base=base_env, cublas_workspace_config=cublas_workspace_config)
    return WorkerCommand(role=role, argv=argv, env=env)
