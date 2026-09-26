"""The determinism switches of design section 10.1, applied inside a worker.

- ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` must be in the environment **before CUDA initialises**, or cuBLAS
  is not deterministic. The daemon puts it in the worker's environment (``[workers] env``), and
  ``prepare_cuda_env`` makes sure of it at start-up, before any handler imports torch.
- The switches a ``load`` request carries (``determinism``): TF32 off, cuDNN deterministic, cuDNN benchmark
  off, and ``torch.use_deterministic_algorithms`` on, off or warn-only. ``apply_determinism`` sets them on
  an imported torch module.
- ``seed_everything`` seeds ``random``, numpy, torch and torch.cuda with a request's seed (section 10.3).

torch and numpy are imported lazily, or passed in, so this package imports and its tests run without them.
"""

from __future__ import annotations

import importlib
import os
import random
from collections.abc import MutableMapping
from typing import Any, Final, cast

from .errors import OpError
from .protocol import DeterminismSettings

CUBLAS_WORKSPACE_CONFIG_VAR: Final = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_WORKSPACE_CONFIG: Final = ":4096:8"
"""The value section 10.1 pins; required for deterministic cuBLAS."""

DETERMINISM_KEYS: Final = ("tf32", "cudnn_deterministic", "cudnn_benchmark", "deterministic_algorithms")
DETERMINISTIC_ALGORITHMS_MODES: Final = ("warn_only", "on", "off")


def prepare_cuda_env(environ: MutableMapping[str, str] | None = None) -> str:
    """Make sure ``CUBLAS_WORKSPACE_CONFIG`` is set; call it before anything imports torch.

    A value the daemon passed is kept; otherwise the pinned value is set. Returns the value in force.
    """
    env = os.environ if environ is None else environ
    return env.setdefault(CUBLAS_WORKSPACE_CONFIG_VAR, CUBLAS_WORKSPACE_CONFIG)


def parse_determinism(value: object) -> DeterminismSettings:
    """Check a ``load`` request's ``determinism`` member; raises ``OpError`` (``INVALID_REQUEST``)."""
    if not isinstance(value, dict):
        raise OpError("INVALID_REQUEST", "determinism must be an object", {"field": "determinism"})
    settings = cast(dict[str, object], value)
    missing = [k for k in DETERMINISM_KEYS if k not in settings]
    unknown = sorted(set(settings) - set(DETERMINISM_KEYS))
    if missing or unknown:
        raise OpError(
            "INVALID_REQUEST",
            "determinism must have exactly the keys " + ", ".join(DETERMINISM_KEYS),
            {"field": "determinism", "missing": missing, "unknown": unknown},
        )
    for key in ("tf32", "cudnn_deterministic", "cudnn_benchmark"):
        if not isinstance(settings[key], bool):
            raise OpError(
                "INVALID_REQUEST", f"determinism.{key} must be true or false", {"field": f"determinism.{key}"}
            )
    if settings["deterministic_algorithms"] not in DETERMINISTIC_ALGORITHMS_MODES:
        raise OpError(
            "INVALID_REQUEST",
            "determinism.deterministic_algorithms must be one of " + ", ".join(DETERMINISTIC_ALGORITHMS_MODES),
            {"field": "determinism.deterministic_algorithms"},
        )
    return cast(DeterminismSettings, settings)


def import_optional(name: str) -> Any | None:
    """Import a module if it is installed, else ``None`` (for torch, numpy and NVML).

    A module that is installed but cannot be loaded raises ``OpError`` (``BACKEND_NOT_INSTALLED``, with the
    error in ``details.error``): the worker's env is broken, and saying "not installed" would hide it. That
    is an ``OSError`` from the import (on Windows, torch raises one when its DLLs fail to load), or an
    ``ImportError`` for anything but the module itself being absent (a missing dependency, a failed
    extension).
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name is not None and (name == exc.name or name.startswith(f"{exc.name}.")):
            return None
        broken: Exception = exc
    except (ImportError, OSError) as exc:
        broken = exc
    error = f"{type(broken).__name__}: {broken}"
    raise OpError(
        "BACKEND_NOT_INSTALLED",
        f"{name} is installed but cannot be loaded ({error}): sync the worker's venv (narration-admin install)",
        {"module": name, "error": error},
    ) from broken


def apply_determinism(settings: DeterminismSettings, torch: Any | None = None) -> dict[str, object]:
    """Set the section 10.1 switches on torch; returns what was applied (``{"torch": False}`` without torch).

    ``torch`` is the imported module; when omitted, torch is imported if it is installed. With deterministic
    algorithms on (or warn-only), ``CUBLAS_WORKSPACE_CONFIG`` must already be set; ``OpError`` (``INTERNAL``)
    otherwise, since setting it now could be after CUDA started.
    """
    module = torch if torch is not None else import_optional("torch")
    if module is None:
        return {"torch": False}
    mode = settings["deterministic_algorithms"]
    if mode != "off" and not os.environ.get(CUBLAS_WORKSPACE_CONFIG_VAR):
        raise OpError(
            "INTERNAL",
            f"{CUBLAS_WORKSPACE_CONFIG_VAR} is not set; it must be in the worker's environment before CUDA starts",
            {"variable": CUBLAS_WORKSPACE_CONFIG_VAR},
        )
    module.backends.cuda.matmul.allow_tf32 = settings["tf32"]
    module.backends.cudnn.allow_tf32 = settings["tf32"]
    module.backends.cudnn.deterministic = settings["cudnn_deterministic"]
    module.backends.cudnn.benchmark = settings["cudnn_benchmark"]
    module.use_deterministic_algorithms(mode != "off", warn_only=mode == "warn_only")
    return {
        "torch": True,
        "tf32": settings["tf32"],
        "cudnn_deterministic": settings["cudnn_deterministic"],
        "cudnn_benchmark": settings["cudnn_benchmark"],
        "deterministic_algorithms": mode,
        CUBLAS_WORKSPACE_CONFIG_VAR: os.environ.get(CUBLAS_WORKSPACE_CONFIG_VAR),
    }


def seed_everything(seed: int, *, torch: Any | None = None, numpy: Any | None = None) -> list[str]:
    """Seed ``random``, numpy, torch and torch.cuda with ``seed`` (section 10.3); returns what was seeded.

    Modules not passed in are imported if installed. torch.cuda is seeded only when CUDA is available, and
    ``manual_seed_all`` does not start CUDA by itself (it is deferred until CUDA starts).
    """
    if not 0 <= seed <= 0xFFFFFFFF:
        raise OpError("INVALID_REQUEST", f"seed must be in 0..2**32-1, not {seed}", {"field": "seed"})
    random.seed(seed)
    seeded = ["random"]
    np_module = numpy if numpy is not None else import_optional("numpy")
    if np_module is not None:
        np_module.random.seed(seed)
        seeded.append("numpy")
    torch_module = torch if torch is not None else import_optional("torch")
    if torch_module is not None:
        torch_module.manual_seed(seed)
        seeded.append("torch")
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(seed)
            seeded.append("torch.cuda")
    return seeded
