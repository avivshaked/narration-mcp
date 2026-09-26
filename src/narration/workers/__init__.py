"""The daemon's side of the model workers (design Appendix A, sections 4 and 4.1; plan.md WP16).

- ``launch``: the command line and environment a worker starts with.
- ``client``: ``SubprocessWorkerClient``, which implements ``contracts.interfaces.WorkerClient``.

The workers themselves are the ``narration_worker`` package (``workers/common``) and the worker projects.
"""

from __future__ import annotations

from .client import SpawnHook, SubprocessWorkerClient
from .launch import WorkerCommand, venv_python, worker_command, worker_env, worker_project

__all__ = [
    "SpawnHook",
    "SubprocessWorkerClient",
    "WorkerCommand",
    "venv_python",
    "worker_command",
    "worker_env",
    "worker_project",
]
