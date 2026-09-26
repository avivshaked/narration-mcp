"""The GPU side of the scheduler (design section 4): one resident group, the free-VRAM check, the wait.

**One resident group** (section 4 item 1): Qwen (Base, or VoiceDesign for design jobs) or the QA group
(Whisper and WavLM). Before a group is loaded, whatever other group is on the GPU is unloaded; a group already
loaded with the same engine profile on the same worker process is used as it is, across jobs too, so work
that needs the resident group starts without a load (affinity).

**The free-VRAM check** (section 4 item 2): before loading, NVML must show free VRAM of at least the group's
need plus ``[gpu] min_free_margin_mb`` (1 GB). Otherwise the job's phase is ``waiting_for_gpu``, the check
is made again every 15 s, and after ``[gpu] wait_timeout_min`` (30 min) the job fails with
``GPU_UNAVAILABLE`` (retryable, with ``retry_after_s`` and the facts behind it). The service **never**
kills, throttles or inspects another process: NVML is asked for the device's memory totals only.

The wait is spread over steps: each step makes one check and sleeps at most one recheck interval (through
``host.sleep``, so a stop ends it at once), then returns, so a cancel or a higher-priority job is seen within
15 s. A probe that cannot read the GPU (no NVML, no NVIDIA driver, the fake workers) makes no check: the
worker's own ``load`` then reports what is missing.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol

from narration.config import GpuConfig
from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.names import GpuHolder, JobPhase
from narration.store.store import utc_iso

from .admission import GPU_RECHECK_S, GPU_UNAVAILABLE_RETRY_S
from .host import GpuFacts, RunnerHost

log = logging.getLogger(__name__)

LOAD_TIMEOUT_S: Final = 900.0
"""How long a ``load`` may take before the worker is stopped (a cold model load is well under this)."""
UNLOAD_TIMEOUT_S: Final = 120.0
_DEVICE: Final = re.compile(r"cuda(?::(\d+))?")


@dataclass(frozen=True, slots=True)
class VramReading:
    """The device's memory as NVML reports it, in MiB, and its name."""

    name: str | None
    total_mb: int | None
    free_mb: int | None


class VramProbe(Protocol):
    """Reads the GPU's memory; None when it cannot be read (no NVML, no driver, no such device)."""

    def read(self) -> VramReading | None:
        """The device's memory now, or None when it cannot be read."""
        ...


class NoProbe:
    """A probe that never reads anything: no free-VRAM check is made (the fake workers, CPU-only work)."""

    def read(self) -> VramReading | None:
        """Nothing: the memory is never known."""
        return None


class NvmlProbe:
    """Free VRAM through NVML (``nvidia-ml-py``), for the device ``[gpu] device`` names (``cuda:N``).

    NVML numbers devices in PCI order, as CUDA does when ``CUDA_DEVICE_ORDER=PCI_BUS_ID``; on a machine with
    one GPU the two agree. Only the device's memory totals are read, never its processes. Any NVML error makes
    the reading unknown (None), and the next ``read`` tries again.
    """

    def __init__(self, device: str) -> None:
        match = _DEVICE.fullmatch(device.strip())
        self._index = int(match.group(1) or 0) if match else None
        self._ready = False

    def read(self) -> VramReading | None:
        """The device's name and memory totals from NVML, or None (see the class docstring)."""
        if self._index is None:
            return None
        try:
            import pynvml  # nvidia-ml-py; imported here so a machine without the driver still starts

            if not self._ready:
                pynvml.nvmlInit()
                self._ready = True
            handle = pynvml.nvmlDeviceGetHandleByIndex(self._index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            raw_name: Any = pynvml.nvmlDeviceGetName(handle)
        except Exception as exc:  # NVMLError, a missing library, an index out of range
            log.debug("NVML cannot read device %s: %s", self._index, exc)
            self._ready = False
            return None
        name = raw_name.decode("utf-8", "replace") if isinstance(raw_name, bytes) else str(raw_name)
        return VramReading(name=name, total_mb=int(memory.total) // (1 << 20), free_mb=int(memory.free) // (1 << 20))


@dataclass(frozen=True, slots=True, kw_only=True)
class GroupNeed:
    """What a phase needs loaded: the group, which models (``key``: the engine profile hash or the QA pins),
    the ``load`` payload, the VRAM it needs, and the engine profile's cuBLAS pin (section 10.1)."""

    group: GpuHolder
    key: str
    label: str
    payload: Mapping[str, Any]
    need_mb: int
    cublas_workspace_config: str | None = None
    gpu: bool = True


Readiness = Literal["ready", "loaded", "waiting"]
"""``ready``: already loaded, nothing done; ``loaded``: loaded by this call; ``waiting``: waiting for VRAM."""


@dataclass(slots=True)
class _Wait:
    since: float
    since_iso: str
    group: GpuHolder
    last: VramReading | None = None


@dataclass(slots=True)
class Residency:
    """Which group is loaded with which models, and the wait for free VRAM (see the module docstring).

    ``clock`` measures the wait (monotonic seconds); ``wall`` gives the ISO time of ``waiting_since``. Both
    can be replaced in tests. ``need_mb`` is reported by group in the GPU facts (``admission.gpu.need_mb``).
    """

    gpu: GpuConfig
    probe: VramProbe
    clock: Callable[[], float] = time.monotonic
    wall: Callable[[], str] = field(default=lambda: utc_iso(time.time()))
    need_mb: dict[str, int] = field(default_factory=dict)
    _loaded: dict[GpuHolder, tuple[str, int | None]] = field(default_factory=dict)
    _wait: _Wait | None = None
    _reading: VramReading | None = None

    # ------------------------------------------------------------------ what is loaded
    def is_ready(self, host: RunnerHost, need: GroupNeed) -> bool:
        """Whether ``need``'s models are loaded on the group's current worker (no call is sent)."""
        mine = self._loaded.get(need.group)
        if mine is None or mine[0] != need.key or need.group not in host.workers.loaded():
            return False
        client = host.workers.client(need.group, cublas_workspace_config=need.cublas_workspace_config)
        return client.pid == mine[1]

    def forget(self, group: GpuHolder) -> None:
        """The group's models are gone (unloaded, or its worker died)."""
        self._loaded.pop(group, None)

    def unload(self, host: RunnerHost, group: GpuHolder) -> None:
        """Unload the group's models (the worker keeps running)."""
        self._loaded.pop(group, None)
        host.workers.unload(group, timeout_s=UNLOAD_TIMEOUT_S)

    @property
    def waiting(self) -> bool:
        """Whether a job is waiting for free VRAM."""
        return self._wait is not None

    def reset_wait(self, host: RunnerHost | None = None) -> None:
        """Forget a wait (the job changed, or its work no longer needs the GPU)."""
        if self._wait is not None:
            self._wait = None
            if host is not None:
                self.publish(host)

    # ------------------------------------------------------------------ making a group ready
    def ensure(self, host: RunnerHost, need: GroupNeed, *, phase: Callable[[JobPhase], None]) -> Readiness:
        """Make ``need`` loaded, or wait one recheck interval for free VRAM.

        Raises ``NarrationError(GPU_UNAVAILABLE)`` once the wait has lasted ``wait_timeout_min``, and lets the
        pool's ``WorkerFailure``/``WorkerCrashed``/``WorkerTimeout`` through.
        """
        if self.is_ready(host, need):
            self.reset_wait(host)
            return "ready"
        pool = host.workers
        holder = pool.gpu_holder
        if need.gpu and holder is not None and holder != need.group:
            log.info("unloading the %s group to load %s", holder, need.label)
            self.unload(host, holder)
        if need.group in pool.loaded():  # the same group with other models (Base and VoiceDesign)
            self.unload(host, need.group)
        if need.gpu and not self._vram_ok(host, need):
            if self._wait is None or self._wait.group != need.group:
                self._wait = _Wait(since=self.clock(), since_iso=self.wall(), group=need.group, last=self._reading)
            self._wait.last = self._reading
            waited = self.clock() - self._wait.since
            limit = 60.0 * self.gpu.wait_timeout_min
            self.publish(host)
            if waited >= limit:
                free = self._reading.free_mb if self._reading else None
                self.reset_wait(host)
                raise NarrationError(
                    codes.GPU_UNAVAILABLE,
                    f"waited {waited / 60:.0f} min for {need.need_mb + self.gpu.min_free_margin_mb} MB of free VRAM "
                    f"to load {need.label}; the GPU had {free} MB free",
                    details={"free_mb": free, "need_mb": need.need_mb, "waited_s": round(waited, 1)},
                    retry_after_s=GPU_UNAVAILABLE_RETRY_S,
                )
            phase("waiting_for_gpu")
            host.sleep(min(GPU_RECHECK_S, max(0.0, limit - waited)))
            return "waiting"
        self.reset_wait(host)
        phase("loading_model")
        client = pool.client(need.group, cublas_workspace_config=need.cublas_workspace_config)
        pool.load(
            need.group,
            need.payload,
            gpu=need.gpu,
            timeout_s=LOAD_TIMEOUT_S,
            cublas_workspace_config=need.cublas_workspace_config,
        )
        self._loaded[need.group] = (need.key, client.pid)
        self._reading = self.probe.read()
        self.publish(host)
        log.info("loaded %s on the %s worker (pid %s)", need.label, need.group, client.pid)
        return "loaded"

    def _vram_ok(self, host: RunnerHost, need: GroupNeed) -> bool:
        self._reading = self.probe.read()
        if self._reading is None or self._reading.free_mb is None:
            return True
        return self._reading.free_mb >= need.need_mb + self.gpu.min_free_margin_mb

    def publish(self, host: RunnerHost) -> None:
        """Tell the daemon what is known of the GPU (``run/daemon.json``; DC-2's ``admission.gpu``)."""
        reading = self._reading
        host.set_gpu_facts(
            GpuFacts(
                name=reading.name if reading else None,
                total_mb=reading.total_mb if reading else None,
                free_mb=reading.free_mb if reading else None,
                need_mb=dict(self.need_mb),
                waiting_since=self._wait.since_iso if self._wait else None,
            )
        )


__all__ = [
    "LOAD_TIMEOUT_S",
    "UNLOAD_TIMEOUT_S",
    "GroupNeed",
    "NoProbe",
    "NvmlProbe",
    "Readiness",
    "Residency",
    "VramProbe",
    "VramReading",
]
