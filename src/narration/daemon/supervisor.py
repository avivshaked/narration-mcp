"""The worker supervisor (design sections 4 and 4.1; plan.md WP30): one worker process per model group.

``WorkerSupervisor`` implements ``seam.WorkerPool``. It starts each worker with WP16's client
(``narration.workers``): the worker venv's own Python, run directly (never through uv), with the offline
environment and the CPU thread cap of ``launch.worker_env``. Each worker process:

- is created at below-normal priority (``[workers] priority``) and without a console window, and is
  added to the daemon's kill-on-close group (``Platform.kill_on_close_group``) in the client's
  ``on_spawn``, before ``hello``, so it dies with the daemon however the daemon ends;
- is started again on the next use after it crashed, up to ``CRASH_LIMIT`` crashes in ``CRASH_WINDOW_S``;
- is never started again, for the rest of the daemon's life, once it could not start at all
  (``BACKEND_NOT_INSTALLED``: its venv is missing or broken; section 14 makes that not retryable).
  ``narration-admin install`` (WP37) stops the daemon after it repairs a worker; the next use starts a
  fresh daemon, which tries again.

It tracks which groups have models loaded, and which one is on the GPU. With ``[gpu]
one_group_at_a_time`` it refuses a second GPU group (``ResidencyError``); which group to load, and when to
swap, is the job engine's choice (WP31).

**Where a worker runs (section 17).** A worker's working directory is its worker project folder
(``[workers.<role>] project``), never the store root, and its environment gets this OS's
``ProcessPlatform.hardening_env`` (on Windows ``NoDefaultCurrentDirectoryInExePath=1``). Importing
``qwen_tts`` imports ``sox``, which runs ``os.popen("sox -h")``; on Windows that goes through ``cmd.exe``,
which looks for the program in the current directory before ``PATH`` (KNOW, WP20's reading of qwen-tts
0.1.1). The variable turns that search off, and the project folder is the operator's own. ``PATH`` itself is
kept: it is the operator's, and a scrubbed ``PATH`` can break DLL loading. Its creation flags come from
``ProcessPlatform.worker_creationflags`` (on Windows no console window, and below-normal priority). Its
command line carries ``-P`` (its working folder is not put on ``sys.path``), and its environment has none of
the ``PYTHON*`` variables that change imports (``settings.IMPORT_ENV_VARS``).

**Which GPU a worker means (``GPU_ORDER_ENV``).** ``[gpu] device`` (``cuda:1``) names one GPU for two readers:
the job engine's free-VRAM check, which asks NVML in the daemon, and the worker's load, which asks CUDA. NVML
numbers GPUs by PCI bus; CUDA by default numbers them fastest first. So every worker starts with
``CUDA_DEVICE_ORDER=PCI_BUS_ID``, replacing any value the operator's environment had, and on a machine with
two GPUs the check reads the GPU the models go on.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from narration.config import Config
from narration.contracts import codes
from narration.contracts.errors import WorkerCrashed, WorkerFailure, WorkerTimeout
from narration.contracts.models import WorkerInfo
from narration.contracts.names import GpuHolder, WorkerRole
from narration.platform import ProcessPlatform
from narration.workers import SubprocessWorkerClient, WorkerCommand, worker_command, worker_project

from .seam import GROUP_ROLES, ResidencyError
from .settings import isolated, scrub_python_env

log = logging.getLogger(__name__)

CRASH_LIMIT: Final = 3
"""How many crashes of one group's worker within ``CRASH_WINDOW_S`` stop it being started again."""
CRASH_WINDOW_S: Final = 300.0
FAKE_ROLES: Final[dict[GpuHolder, WorkerRole]] = {"qwen": "fake", "qa": "fake"}
"""Every group served by the fake worker (``DaemonSettings.fake_workers``)."""
GPU_ORDER_ENV: Final[dict[str, str]] = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
"""Set in every worker's environment, so CUDA numbers GPUs as NVML does (see the module docstring)."""


def worker_cwd(config: Config, group: GpuHolder, role: WorkerRole, python: Path) -> Path:
    """A worker's working directory: its worker project folder, never the store root (section 17).

    The fake role has no project of its own, so it runs in the project of the role it stands in for
    (``GROUP_ROLES[group]``) when that folder exists, else in the folder of the Python it runs.
    """
    candidates = [worker_project(config, role)]
    if role == "fake":
        candidates.append(worker_project(config, GROUP_ROLES[group]))
    for folder in candidates:
        if folder is not None and folder.is_dir():
            return folder
    return python.parent


CommandFactory = Callable[[WorkerRole, str | None], WorkerCommand]
"""Builds a worker's command from its role and the engine profile's cuBLAS pin."""
ClientFactory = Callable[..., SubprocessWorkerClient]


class SupervisorClosed(WorkerCrashed):
    """The daemon is stopping: its workers were stopped, and none will start again."""


@dataclass(slots=True)
class _Slot:
    """One group's worker: the client, how it was started, and its failures."""

    role: WorkerRole
    client: SubprocessWorkerClient | None = None
    starting: bool = False
    """True while ``client`` is inside ``start()``: its start judges its own failure, so ``poll`` leaves it."""
    cublas: str | None = None
    crashes: deque[float] = field(default_factory=deque)
    failure: WorkerFailure | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class WorkerSupervisor:
    """The daemon's ``WorkerPool`` (see the module docstring). Use it as a context manager: entering opens the
    kill-on-close group, and leaving closes every worker, then the group.

    ``roles`` maps each group to the role that serves it (``seam.GROUP_ROLES`` by default; ``FAKE_ROLES``
    for tests). ``base_env`` is the environment workers start from (this process's by default); every
    worker's environment then gets ``platform.hardening_env()``, and it runs in ``worker_cwd``. ``on_change`` is
    called, without arguments and on the thread that made it, after every change a status would show: a
    worker started, stopped or found dead; models loaded or unloaded.
    """

    def __init__(
        self,
        config: Config,
        platform: ProcessPlatform,
        *,
        roles: Mapping[GpuHolder, WorkerRole] | None = None,
        below_normal: bool = True,
        base_env: Mapping[str, str] | None = None,
        close_timeout_s: float = 10.0,
        on_change: Callable[[], None] | None = None,
        command_factory: CommandFactory | None = None,
        client_factory: ClientFactory = SubprocessWorkerClient,
        client_options: Mapping[str, Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._platform = platform
        self._slots: dict[GpuHolder, _Slot] = {
            group: _Slot(role=role) for group, role in (roles or GROUP_ROLES).items()
        }
        self._below_normal = below_normal
        self._close_timeout_s = close_timeout_s
        self._on_change = on_change
        self._command_factory = command_factory or self._default_command
        self._base_env = dict(os.environ if base_env is None else base_env)
        self._client_factory = client_factory
        self._client_options = dict(client_options or {})
        self._clock = clock
        self._lock = threading.RLock()
        self._loaded: set[GpuHolder] = set()
        self._gpu_holder: GpuHolder | None = None
        self._closed = False
        self._group: AbstractContextManager[Callable[[int], None]] | None = None
        self._add: Callable[[int], None] | None = None
        self._group_lock = threading.Lock()

    # ------------------------------------------------------------------ lifetime
    def __enter__(self) -> WorkerSupervisor:
        group = self._platform.kill_on_close_group()
        self._add = group.__enter__()
        self._group = group
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Stop every worker (``shutdown``, then a kill after ``close_timeout_s``), then close the group,
        which kills anything left in it. No worker starts afterwards."""
        with self._lock:
            self._closed = True
            clients = [(group, slot.client) for group, slot in self._slots.items() if slot.client is not None]
        for group, client in clients:
            try:
                client.close(timeout_s=self._close_timeout_s)
            except Exception:
                log.exception("closing the %s worker failed; the kill-on-close group will end it", group)
        self._close_group()
        self._forget_all()

    def kill_all(self) -> None:
        """Kill every worker at once (``stop_now``, section 4.1: terminate the Job Object). Requests in
        flight fail with ``WorkerCrashed``, and no worker starts afterwards."""
        with self._lock:
            self._closed = True
            clients = [slot.client for slot in self._slots.values() if slot.client is not None]
        self._close_group()
        for client in clients:
            try:
                client.close(timeout_s=1.0)  # reaps the process, which the group has already ended
            except Exception:
                log.exception("reaping a killed %s worker failed", client.role)
        self._forget_all()

    def _close_group(self) -> None:
        with self._group_lock:
            group, self._group = self._group, None
        if group is not None:
            group.__exit__(None, None, None)

    def _forget_all(self) -> None:
        with self._lock:
            for slot in self._slots.values():
                slot.client = None
            self._loaded.clear()
            self._gpu_holder = None
        self._changed()

    # ------------------------------------------------------------------ what a status shows
    @property
    def gpu_holder(self) -> GpuHolder | None:
        with self._lock:
            return self._gpu_holder

    def loaded(self) -> frozenset[GpuHolder]:
        with self._lock:
            return frozenset(self._loaded)

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def workers(self) -> tuple[WorkerInfo, ...]:
        """The running workers, by group order: role and pid (a venv launcher's pid, section 4.1)."""
        with self._lock:
            return tuple(
                WorkerInfo(role=slot.role, pid=pid)
                for slot in self._slots.values()
                if slot.client is not None and (pid := slot.client.pid) is not None and slot.client.is_alive()
            )

    def failure(self, group: GpuHolder) -> WorkerFailure | None:
        """The ``BACKEND_NOT_INSTALLED`` failure that keeps ``group``'s worker from starting, if any."""
        with self._lock:
            return self._slot(group).failure

    def poll(self) -> None:
        """Notice workers that died since the last look: count the crash, forget their models, and report
        the change. The next use starts them again.

        A worker still starting is left to its start, which tells a worker that cannot start
        (``BACKEND_NOT_INSTALLED``, for good) from one that crashed. Closing it here would fail its ``hello``
        as closed before its exit code was read, and a broken venv would pass for a crash."""
        dead: list[tuple[GpuHolder, SubprocessWorkerClient]] = []
        with self._lock:
            for group, slot in self._slots.items():
                client = slot.client
                if client is not None and not slot.starting and client.pid is not None and not client.is_alive():
                    dead.append((group, client))
                    self._crashed(group, slot)
        for group, client in dead:
            log.warning("the %s worker (pid %s) exited; it starts again when next needed", group, client.pid)
            client.close(timeout_s=1.0)
        if dead:
            self._changed()

    # ------------------------------------------------------------------ the WorkerPool
    def client(self, group: GpuHolder, *, cublas_workspace_config: str | None = None) -> SubprocessWorkerClient:
        """The group's worker, started if need be (see ``seam.WorkerPool.client``)."""
        slot = self._slot(group)
        with slot.lock:  # one start at a time per group; other groups and status readers are not held up
            with self._lock:
                self._check_open()
                if slot.failure is not None:
                    raise slot.failure
                current = slot.client
                if current is not None and current.is_alive():
                    if cublas_workspace_config is None or cublas_workspace_config == slot.cublas:
                        return current
                elif current is not None:
                    self._crashed(group, slot)
            if current is not None:
                if current.is_alive():
                    log.info("restarting the %s worker with another cuBLAS workspace pin", group)
                current.close(timeout_s=self._close_timeout_s)
                with self._lock:
                    if slot.client is current:
                        slot.client = None
                    self._forget_models(group)
                self._changed()
            return self._start(group, slot, cublas_workspace_config)

    def load(
        self,
        group: GpuHolder,
        payload: Mapping[str, Any],
        *,
        gpu: bool = True,
        timeout_s: float,
        cublas_workspace_config: str | None = None,
    ) -> dict[str, Any]:
        """Send ``load`` (see ``seam.WorkerPool.load``)."""
        with self._lock:
            holder = self._gpu_holder
            if gpu and self._config.gpu.one_group_at_a_time and holder is not None and holder != group:
                raise ResidencyError(
                    f"the {holder} group's models are on the GPU; unload it before loading the {group} group "
                    "([gpu] one_group_at_a_time)"
                )
        client = self.client(group, cublas_workspace_config=cublas_workspace_config)
        reply = client.request("load", payload, timeout_s=timeout_s)
        with self._lock:
            self._loaded.add(group)
            if gpu:
                self._gpu_holder = group
        self._changed()
        return reply

    def unload(self, group: GpuHolder, *, timeout_s: float) -> None:
        """Send ``unload``. If the worker fails to answer, it is stopped: either way, afterwards the group
        has no models loaded."""
        slot = self._slot(group)
        with self._lock:
            client = slot.client
            loaded = group in self._loaded
        if client is not None and loaded and client.is_alive():
            try:
                client.request("unload", {}, timeout_s=timeout_s)
            except (WorkerFailure, WorkerCrashed, WorkerTimeout) as exc:
                log.warning("the %s worker failed to unload (%s); stopping it", group, exc)
                self.stop(group)
                return
        with self._lock:
            self._forget_models(group)
        self._changed()

    def stop(self, group: GpuHolder) -> None:
        """Close the group's worker: ``shutdown`` (which releases its models), a kill if it does not exit."""
        slot = self._slot(group)
        with slot.lock:
            with self._lock:
                client = slot.client
            if client is not None:
                client.close(timeout_s=self._close_timeout_s)
            with self._lock:
                if slot.client is client:
                    slot.client = None
                self._forget_models(group)
        self._changed()

    def release_all(self) -> GpuHolder | None:
        """Stop every worker (idle unload, ``release_gpu``); returns the group that was on the GPU before."""
        holder = self.gpu_holder
        for group in list(self._slots):
            with self._lock:
                running = self._slots[group].client is not None
            if running:
                self.stop(group)
        return holder

    # ------------------------------------------------------------------ internals
    def _slot(self, group: GpuHolder) -> _Slot:
        try:
            return self._slots[group]
        except KeyError:
            raise ValueError(f"no worker group {group!r}; the groups are {', '.join(self._slots)}") from None

    def _check_open(self) -> None:
        if self._closed:
            raise SupervisorClosed("the daemon is stopping; no worker starts")

    def _crashed(self, group: GpuHolder, slot: _Slot) -> None:
        """Record a crash of the slot's current client (the caller holds ``_lock``)."""
        now = self._clock()
        slot.crashes.append(now)
        while slot.crashes and now - slot.crashes[0] > CRASH_WINDOW_S:
            slot.crashes.popleft()
        slot.client = None
        self._forget_models(group)

    def _forget_models(self, group: GpuHolder) -> None:
        self._loaded.discard(group)
        if self._gpu_holder == group:
            self._gpu_holder = None

    def _start(self, group: GpuHolder, slot: _Slot, cublas: str | None) -> SubprocessWorkerClient:
        with self._lock:
            self._check_open()
            now = self._clock()
            recent = [t for t in slot.crashes if now - t <= CRASH_WINDOW_S]
            if len(recent) >= CRASH_LIMIT:
                wait_s = CRASH_WINDOW_S - (now - recent[0])
                raise WorkerCrashed(
                    f"the {group} worker ({slot.role}) crashed {len(recent)} times in {CRASH_WINDOW_S:.0f} s; it is "
                    f"not started again for another {wait_s:.0f} s (see the daemon's log for its stderr)"
                )
        try:
            command = self._command_factory(slot.role, cublas)
        except WorkerFailure as exc:
            self._fail_for_good(group, slot, exc)
            raise
        command = dataclasses.replace(
            command,
            argv=isolated(command.argv),
            env=scrub_python_env({**command.env, **GPU_ORDER_ENV, **self._platform.hardening_env()}),
        )
        client = self._client_factory(
            command,
            cwd=worker_cwd(self._config, group, slot.role, Path(command.argv[0])),
            on_spawn=self._on_spawn,
            creationflags=self._platform.worker_creationflags(below_normal=self._below_normal),
            **self._client_options,
        )
        with self._lock:
            self._check_open()
            slot.client = client
            slot.cublas = cublas
            slot.starting = True
        try:
            client.start()
        except WorkerFailure as exc:
            with self._lock:
                slot.client = None
                slot.starting = False
            if exc.code == codes.BACKEND_NOT_INSTALLED:
                self._fail_for_good(group, slot, exc)
            raise
        except BaseException as exc:
            with self._lock:
                slot.client = None
                slot.starting = False
                closed = self._closed
                if not closed and isinstance(exc, WorkerCrashed | WorkerTimeout):
                    slot.crashes.append(self._clock())
            if closed:
                raise SupervisorClosed(f"the daemon is stopping; the {group} worker was not started") from exc
            raise
        with self._lock:
            slot.starting = False
        log.info("started the %s worker (%s, pid %s)", group, slot.role, client.pid)
        self._changed()
        return client

    def _fail_for_good(self, group: GpuHolder, slot: _Slot, exc: WorkerFailure) -> None:
        with self._lock:
            slot.failure = exc
        log.error(
            "the %s worker (%s) cannot start; it is not tried again until the daemon restarts: %s",
            group,
            slot.role,
            exc,
        )

    def _on_spawn(self, pid: int) -> None:
        """Right after a worker starts, before ``hello``: into the kill-on-close group, then the priority."""
        add = self._add
        if add is None:
            raise SupervisorClosed("the daemon's kill-on-close group is not open")
        add(pid)
        if self._below_normal:
            try:
                self._platform.set_below_normal_priority(pid)
            except OSError as exc:  # priority is a courtesy to the machine, not a safety property
                log.warning("could not lower the priority of worker %d: %s", pid, exc)

    def _default_command(self, role: WorkerRole, cublas: str | None) -> WorkerCommand:
        # The fake role runs with the server's own interpreter, as a console program (it speaks over stdio).
        python = self._platform.python_for(Path(sys.executable), console=True) if role == "fake" else None
        return worker_command(
            self._config, role, python=python, base_env=self._base_env, cublas_workspace_config=cublas
        )

    def _changed(self) -> None:
        if self._on_change is not None:
            try:
                self._on_change()
            except Exception:
                log.exception("reporting a worker change failed")
