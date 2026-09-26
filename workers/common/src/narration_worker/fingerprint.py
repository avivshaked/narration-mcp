"""The ``hello`` fingerprint: what a worker observes of its environment (design section 10.1, Appendix A).

The daemon compares it with the engine profile's pins (``ENGINE_DRIFT``). Every fact is observed, never
assumed: a package that is not installed is left out, and a CUDA fact that cannot be read is ``None``.

CUDA facts come from torch and NVML, and only when the handler passes them in. Neither is imported here.
Reading them never starts a CUDA context: the GPU name and driver come from NVML (``pynvml``, from the
``nvidia-ml-py`` package) when it is installed, and otherwise from torch only once torch has started CUDA
itself. Before that, the GPU name is ``None``.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import platform
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

from .protocol import Fingerprint
from .threads import THREAD_ENV_VARS

log = logging.getLogger(__name__)

FINGERPRINT_ENV_VARS: Final = (
    "CUBLAS_WORKSPACE_CONFIG",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    *THREAD_ENV_VARS,
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "PYTORCH_CUDA_ALLOC_CONF",
    "NVIDIA_TF32_OVERRIDE",
)
"""The environment variables that can change a worker's output or its use of the machine. Those that are
set appear in the fingerprint's ``env``."""

WORKER_PACKAGE: Final = "narration-worker"


def package_versions(names: Sequence[str]) -> dict[str, str]:
    """The installed version of each named distribution; a distribution that is not installed is left out."""
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def collect(
    *,
    cpu_threads: int,
    packages: Sequence[str] = (),
    torch: Any | None = None,
    nvml: Any | None = None,
    environ: Mapping[str, str] | None = None,
) -> Fingerprint:
    """Build the fingerprint.

    ``packages`` are the distributions whose versions matter to the role (``narration-worker`` is always
    included, and ``torch`` when the torch module is passed). ``torch`` and ``nvml`` are the imported modules
    or ``None``.
    """
    env = os.environ if environ is None else environ
    names = [WORKER_PACKAGE, *packages]
    if torch is not None and "torch" not in names:
        names.append("torch")
    cuda: str | None = None
    cudnn: str | None = None
    gpu: str | None = None
    driver: str | None = None
    if torch is not None:
        cuda = _probe("torch.version.cuda", lambda: _str_or_none(torch.version.cuda))
        cudnn = _probe("cudnn version", lambda: _cudnn_version(torch))
    if nvml is not None:
        gpu, driver = _probe("NVML", lambda: _nvml_facts(nvml, env)) or (None, None)
    if gpu is None and torch is not None:
        gpu = _probe("torch GPU name", lambda: _torch_gpu_name(torch))
    return Fingerprint(
        python=platform.python_version(),
        platform=platform.platform(),
        packages=package_versions(names),
        cuda=cuda,
        cudnn=cudnn,
        gpu=gpu,
        driver=driver,
        cpu_threads=cpu_threads,
        env={name: env[name] for name in FINGERPRINT_ENV_VARS if name in env},
    )


def _probe[T](what: str, read: Callable[[], T]) -> T | None:
    try:
        return read()
    except Exception as exc:  # a fact we cannot read is reported as unknown, never as a failed hello
        log.debug("fingerprint: could not read %s: %s: %s", what, type(exc).__name__, exc)
        return None


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


def _cudnn_version(torch: Any) -> str | None:
    if not torch.backends.cudnn.is_available():
        return None
    return _str_or_none(torch.backends.cudnn.version())


def _torch_gpu_name(torch: Any) -> str | None:
    if not torch.cuda.is_initialized():
        return None
    return _str_or_none(torch.cuda.get_device_name())


def _nvml_facts(nvml: Any, env: Mapping[str, str]) -> tuple[str | None, str | None]:
    visible = env.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    index = int(visible) if visible.isdigit() else 0
    nvml.nvmlInit()
    try:
        driver = _str_or_none(nvml.nvmlSystemGetDriverVersion())
        name = _str_or_none(nvml.nvmlDeviceGetName(nvml.nvmlDeviceGetHandleByIndex(index)))
    finally:
        nvml.nvmlShutdown()
    return name, driver
