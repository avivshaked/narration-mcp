"""``narration-admin doctor``: check this machine and the configuration, and say what to fix (design
sections 4, 7.1, 15, 16 and 17.8; plan.md WP37 and WP44).

Each check gives a ``Finding``: ``ok``, ``info``, ``warn`` or ``fail``, what was found, and, for a warning
or a failure, what to do next. The checks, in order:

- **platform**: v1 runs on Windows only (plan.md Q2);
- **config**: the configuration file is found (``find_config``, the rule ``narration-mcp`` uses) and loads;
- **store**: ``[server] store_root`` is writable (or can be created), with at least ``[limits]
  min_free_disk_gb`` free;
- **models**: each pinned model's snapshot is under ``[server] models_root``, and its files are the ones
  the install recorded (hashed again unless ``--quick``; ``narration.admin.models``). The pins are the
  engine's (WP32's ``narration.engine.models``); until that module is in this build they are unknown, and
  the models the install record lists are checked instead;
- **workers**: each worker's uv project is there, its venv has its Python, and ``uv sync --check`` says the
  venv matches its ``uv.lock``;
- **gpu**: NVML finds the configured NVIDIA GPU, with its memory and driver. A machine without one is told
  plainly that it cannot run jobs (WP44);
- **engine**: the store has a pinned engine profile for Base and for VoiceDesign, with its determinism tier
  and canary (``engine pin``, section 10.1). Whether the installed stack still matches the pin is the engine
  module's to say; until it is in this build that is reported as unknown.

The exit code is 1 when any check fails, else 0. ``doctor`` changes nothing, except that it writes and
removes one empty file to test that the store root is writable.
"""

from __future__ import annotations

import argparse
import json
import os
import platform as os_platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from narration.config import Config
from narration.contracts.errors import UnsupportedPlatform
from narration.contracts.names import WorkerRole
from narration.platform import ProcessPlatform, is_supported
from narration.workers.launch import WORKER_PROJECT_DIRS, venv_python, worker_project

from .cli import EXIT_FAILED, EXIT_OK, PROGRAM, Admin, AdminError, Subparsers, names_the_module
from .models import check_files, pinned_revisions, read_manifest, snapshot_dir

Level = Literal["ok", "info", "warn", "fail"]
ENGINE_MODULE: Final = "narration.engine.admin"
WORKER_ROLES: Final[tuple[WorkerRole, ...]] = ("qwen3", "qa")
UV_CHECK_TIMEOUT_S: Final = 120.0
GIB: Final = 1024**3


@dataclass(frozen=True, slots=True)
class Finding:
    """One check's result: its area, its level, what was found and (for ``warn`` and ``fail``) what to do."""

    area: str
    level: Level
    summary: str
    next_step: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class GpuFacts:
    """What NVML says of the configured GPU."""

    index: int
    name: str
    total_mb: int
    free_mb: int
    driver: str
    count: int


class GpuUnavailable(Exception):
    """NVML cannot read the configured GPU; the message says why. ``no_driver`` is True when there is no
    NVIDIA driver at all (the machine has no usable NVIDIA GPU); ``next_step`` is what to do, when the cause
    says it."""

    def __init__(self, message: str, *, no_driver: bool = False, next_step: str | None = None) -> None:
        super().__init__(message)
        self.no_driver = no_driver
        self.next_step = next_step


NO_DRIVER_ERRORS: Final = ("NVMLError_LibraryNotFound", "NVMLError_DriverNotLoaded")
"""NVML's errors that mean no NVIDIA driver is there to use."""
RESTART_ERRORS: Final = ("NVMLError_LibRmVersionMismatch",)
"""NVML's errors that a restart cures: the driver was updated and the loaded one no longer matches its library."""


def _nvml_start_failure(exc: Exception) -> GpuUnavailable:
    """What NVML's start failure means, in its own words and with what to do."""
    kind = type(exc).__name__
    if kind in NO_DRIVER_ERRORS:
        return GpuUnavailable(f"NVML could not start ({exc}): no NVIDIA driver is loaded", no_driver=True)
    if kind in RESTART_ERRORS:
        return GpuUnavailable(
            f"NVML could not start ({exc}): the NVIDIA driver in use does not match its library, as after a driver "
            "update",
            next_step="Restart the machine to load the updated driver, then run doctor again.",
        )
    return GpuUnavailable(
        f"NVML could not start ({kind}: {exc})",
        next_step=(
            "Check the NVIDIA driver (nvidia-smi shows what it says), reinstall it if needed, then run doctor again."
        ),
    )


@dataclass(frozen=True, slots=True)
class SyncCheck:
    """Whether a venv matches its lock: True, False, or None when that could not be checked."""

    synced: bool | None
    detail: str


def read_gpu(index: int) -> GpuFacts:
    """The facts of GPU ``index`` from NVML (``nvidia-ml-py``); ``GpuUnavailable`` when NVML cannot tell.
    This starts no CUDA context and looks at no process."""
    try:
        import pynvml  # pyright: ignore[reportMissingTypeStubs]
    except ImportError as exc:
        raise GpuUnavailable(f"the NVML binding (nvidia-ml-py) is not installed: {exc}") from exc
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:
        raise _nvml_start_failure(exc) from exc
    try:
        count = int(pynvml.nvmlDeviceGetCount())
        if index >= count:
            raise GpuUnavailable(f"there is no GPU {index}: NVML sees {count}")
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return GpuFacts(
            index=index,
            name=_text(pynvml.nvmlDeviceGetName(handle)),
            total_mb=int(memory.total) // (1024 * 1024),
            free_mb=int(memory.free) // (1024 * 1024),
            driver=_text(pynvml.nvmlSystemGetDriverVersion()),
            count=count,
        )
    except pynvml.NVMLError as exc:
        raise GpuUnavailable(f"NVML could not read GPU {index}: {exc}") from exc
    finally:
        pynvml.nvmlShutdown()


def _text(value: object) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def uv_sync_check(project: Path, *, uv: str | None = None) -> SyncCheck:
    """Ask ``uv sync --check --locked --offline`` whether ``project``'s venv matches its ``uv.lock``. It
    changes nothing and needs no network."""
    uv = uv or os.environ.get("UV") or shutil.which("uv")
    if not uv:
        return SyncCheck(None, "uv was not found, so whether the venv matches its uv.lock is unknown")
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}  # this is the server's venv, not the worker's
    env["UV_PYTHON_DOWNLOADS"] = "never"
    try:
        done = subprocess.run(
            [uv, "sync", "--check", "--locked", "--offline", "--project", str(project)],
            cwd=project,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=UV_CHECK_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return SyncCheck(None, f"uv could not check it ({exc})")
    if done.returncode == 0:
        return SyncCheck(True, "its venv matches its uv.lock")
    lines = [line.strip() for line in (done.stderr + "\n" + done.stdout).splitlines() if line.strip()]
    return SyncCheck(
        False, f"its venv does not match its uv.lock ({lines[-1] if lines else f'uv exit {done.returncode}'})"
    )


def engine_module_present() -> bool:
    """Whether the engine's commands (WP32's ``narration.engine.admin``) are in this build."""
    import importlib.util

    try:
        return importlib.util.find_spec(ENGINE_MODULE) is not None
    except ModuleNotFoundError as exc:
        # Absent only when the engine package itself is missing; one that fails on a dependency is in this
        # build (broken), and ``narration-admin engine`` says why.
        return not names_the_module(exc, ENGINE_MODULE)


@dataclass(frozen=True, slots=True, kw_only=True)
class Probes:
    """What ``diagnose`` asks the machine; tests replace them."""

    gpu: Callable[[int], GpuFacts] = read_gpu
    venv_synced: Callable[[Path], SyncCheck] = uv_sync_check
    pins: Callable[[], dict[str, str] | None] = pinned_revisions
    engine_module: Callable[[], bool] = engine_module_present
    supported: Callable[[], bool] = is_supported


# ---------------------------------------------------------------- the checks
def check_platform(supported: bool) -> Finding:
    """v1 runs on Windows only."""
    here = f"{os_platform.system() or sys.platform} ({sys.platform}), Python {os_platform.python_version()}"
    if supported:
        return Finding("platform", "ok", here)
    return Finding(
        "platform",
        "fail",
        f"{here}: narration-mcp v1 runs on Windows only, so the daemon, and every job, cannot run here",
        "Run the service on Windows 10 or 11; other platforms are planned.",
    )


def check_store(config: Config, platform: ProcessPlatform) -> list[Finding]:
    """The store root is writable (or can be created there), with enough free disk."""
    root = config.server.store_root
    if root.exists() and not root.is_dir():
        return [Finding("store", "fail", f"{root} is not a folder", "Point [server] store_root at a folder.")]
    existing = root
    while not existing.exists() and existing.parent != existing:
        existing = existing.parent
    try:
        handle, probe = tempfile.mkstemp(dir=existing, prefix=".narration-doctor-")
        os.close(handle)
        os.unlink(probe)
    except OSError as exc:
        return [
            Finding(
                "store",
                "fail",
                f"cannot write in {existing} ({exc})",
                "Give this account write access there, or point [server] store_root at a folder it can write.",
            )
        ]
    where = (
        f"{root} is writable" if existing == root else f"{root} will be created at first use ({existing} is writable)"
    )
    findings = [Finding("store", "ok", where)]
    try:
        free = platform.free_disk_bytes(existing)
    except UnsupportedPlatform:
        free = shutil.disk_usage(existing).free
    need = config.limits.min_free_disk_gb
    if free < need * GIB:
        findings.append(
            Finding(
                "disk",
                "fail",
                f"{free / GIB:.1f} GB free where the store is; the service takes no new work below "
                f"[limits] min_free_disk_gb = {need}",
                f"Free space on that drive (`{PROGRAM} gc --apply` removes what retention no longer keeps), or "
                "move [server] store_root.",
            )
        )
    else:
        findings.append(Finding("disk", "ok", f"{free / GIB:.0f} GB free where the store is"))
    return findings


def check_models(config: Config, pins: dict[str, str] | None, *, hash_files: bool) -> list[Finding]:
    """Each pinned model is installed, and its files are the ones the install recorded."""
    root = config.server.models_root
    install = f"Run `{PROGRAM} install`, which downloads the pinned models there and checks their hashes."
    if not root.is_dir():
        return [Finding("models", "fail", f"the models root {root} does not exist", install)]
    try:
        records = read_manifest(root)
    except ValueError as exc:
        return [Finding("models", "fail", str(exc), f"Run `{PROGRAM} install`, which writes it again.")]
    findings: list[Finding] = []
    if pins is None:
        findings.append(
            Finding(
                "models",
                "info",
                "which revisions this build pins is unknown (the engine's pins, narration.engine.models, are not "
                "in it), so the models the install record lists are checked",
            )
        )
        wanted = [(m.repo, m.revision) for m in records.values()]
        if not wanted:
            findings.append(Finding("models", "warn", f"no model is recorded as installed in {root}", install))
    else:
        wanted = sorted(pins.items())
        if config.alignment.model not in pins:
            findings.append(
                Finding(
                    "models",
                    "warn",
                    f"[alignment] model is {config.alignment.model}, which this build does not pin",
                    "Set [alignment] model to a pinned aligner (narration.example.toml names the default).",
                )
            )
    for repo, revision in wanted:
        name = f"{repo}@{revision[:12]}"
        if not snapshot_dir(root, repo, revision).is_dir():
            findings.append(Finding("models", "fail", f"{name} is not installed", install))
            continue
        record = records.get(f"{repo}@{revision}")
        if record is None:
            findings.append(
                Finding(
                    "models",
                    "warn",
                    f"{name} is there, but not in the install record, so its files cannot be checked",
                    f"Run `{PROGRAM} install`, which checks the files and records them.",
                )
            )
            continue
        problems = check_files(record, hash_files=hash_files)
        if problems:
            shown = "; ".join(problems[:3]) + (f"; and {len(problems) - 3} more" if len(problems) > 3 else "")
            findings.append(
                Finding("models", "fail", f"{name}: {shown}", f"Run `{PROGRAM} install`, which repairs it.")
            )
        else:
            how = "their hashes match the install record" if hash_files else "present (not hashed: --quick)"
            findings.append(Finding("models", "ok", f"{name}: {len(record.files_sha256)} files, {how}"))
    return findings


def check_workers(config: Config, venv_synced: Callable[[Path], SyncCheck]) -> list[Finding]:
    """Each worker's project is there and its venv matches its lock."""
    findings: list[Finding] = []
    for role in WORKER_ROLES:
        area = f"worker {role}"
        project = worker_project(config, role)
        if project is None:
            findings.append(
                Finding(
                    area,
                    "fail",
                    "no project is configured, and the service's folder is not known",
                    f"Set [workers.{role}] project to the service's workers/{WORKER_PROJECT_DIRS[role]} folder.",
                )
            )
            continue
        if not (project / "pyproject.toml").is_file():
            findings.append(
                Finding(
                    area,
                    "fail",
                    f"{project} is not the worker's uv project (it has no pyproject.toml)",
                    f"Set [workers.{role}] project to the service's workers/{WORKER_PROJECT_DIRS[role]} folder.",
                )
            )
            continue
        sync = f"Run `{PROGRAM} install`, or `uv sync --locked --project {project}`."
        python = venv_python(project)
        if not python.is_file():
            findings.append(Finding(area, "fail", f"its venv is not set up (no {python})", sync))
            continue
        check = venv_synced(project)
        if check.synced is True:
            findings.append(Finding(area, "ok", f"{project}: {check.detail}"))
        elif check.synced is False:
            findings.append(Finding(area, "fail", f"{project}: {check.detail}", sync))
        else:
            findings.append(Finding(area, "warn", f"{project}: {check.detail}", sync))
    return findings


_CUDA_DEVICE = re.compile(r"cuda:([0-9]+)")


def check_gpu(config: Config | None, read: Callable[[int], GpuFacts]) -> Finding:
    """NVML finds the configured GPU."""
    device = config.gpu.device if config is not None else "cuda:0"
    match = _CUDA_DEVICE.fullmatch(device)
    if match is None:
        return Finding(
            "gpu", "fail", f"[gpu] device is {device!r}", "Set [gpu] device to cuda:<n> (cuda:0 is the first)."
        )
    try:
        facts = read(int(match.group(1)))
    except GpuUnavailable as exc:
        if exc.no_driver:
            return Finding(
                "gpu",
                "fail",
                f"no NVIDIA GPU can be used here: {exc}. narration-mcp renders and checks audio on an NVIDIA GPU, "
                "so this machine cannot run jobs; the operator commands that need no GPU still work",
                "If this machine has an NVIDIA GPU, install or update its driver, then run doctor again.",
            )
        return Finding(
            "gpu",
            "fail",
            f"the NVIDIA GPU cannot be used now: {exc}. Jobs cannot run until it can",
            exc.next_step or "Check the NVIDIA driver and the [gpu] device, then run doctor again.",
        )
    return Finding(
        "gpu",
        "ok",
        f"{device}: {facts.name}, {facts.total_mb} MB ({facts.free_mb} MB free now), driver {facts.driver}",
    )


def check_engine(admin: Admin, *, engine_module: bool) -> list[Finding]:
    """The store has a pinned engine profile for Base and VoiceDesign."""
    pin = f"Run `{PROGRAM} engine pin` once the models and workers are installed."
    if not engine_module:
        pin += " (The engine commands are not in this build yet.)"
    if not admin.store_exists():
        return [Finding("engine", "warn", "not pinned: the store has not been used yet", pin)]
    try:
        store = admin.store()
        profiles = {kind: store.current_engine_profile(kind) for kind in ("base", "design")}
    except Exception as exc:
        return [
            Finding(
                "engine",
                "fail",
                f"the store cannot be read ({type(exc).__name__}: {exc})",
                f"Run `{PROGRAM} verify`, which checks the store's database and files; if it reports damage, keep "
                "its report and tell the maintainers.",
            )
        ]
    findings: list[Finding] = []
    for kind, profile in profiles.items():
        if profile is None:
            findings.append(Finding("engine", "warn", f"no {kind} engine profile is pinned", pin))
            continue
        tier = profile.tier or "not measured"
        canary = "canary pinned" if profile.canary is not None else "no canary"
        level: Level = "ok" if profile.tier is not None and profile.canary is not None else "warn"
        findings.append(
            Finding(
                "engine",
                level,
                f"{kind}: {profile.engine_profile_id}, tier {tier}, {canary}",
                None if level == "ok" else pin,
            )
        )
    if not engine_module:
        findings.append(
            Finding(
                "engine",
                "info",
                "whether the installed models and workers still match the pinned profile is unknown in this build "
                "(the engine module is not in it)",
            )
        )
    return findings


def diagnose(admin: Admin, *, probes: Probes | None = None, hash_models: bool = True) -> list[Finding]:
    """Every check (see the module docstring), in order."""
    probes = probes or Probes()
    findings = [check_platform(probes.supported())]
    try:
        path = admin.config_path()
        config = admin.config()
    except AdminError as exc:
        findings.append(Finding("config", "fail", exc.message.rstrip("."), "Then run doctor again."))
        findings.append(
            Finding("checks", "info", "the store, models, workers and engine need a configuration: not checked")
        )
        findings.append(check_gpu(None, probes.gpu))
        return findings
    findings.append(Finding("config", "ok", str(path)))
    findings += check_store(config, admin.platform())
    try:
        pins = probes.pins()
    except ImportError as exc:
        findings.append(
            Finding(
                "models",
                "fail",
                f"the engine's model pins are in this build but cannot be loaded ({exc})",
                "Sync the server's venv (uv sync --locked), then run doctor again.",
            )
        )
    else:
        findings += check_models(config, pins, hash_files=hash_models)
    findings += check_workers(config, probes.venv_synced)
    findings.append(check_gpu(config, probes.gpu))
    findings += check_engine(admin, engine_module=probes.engine_module())
    return findings


# ---------------------------------------------------------------- the command
@dataclass(frozen=True, slots=True)
class _Report:
    findings: list[Finding] = field(default_factory=list[Finding])

    @property
    def failed(self) -> int:
        return sum(1 for f in self.findings if f.level == "fail")

    @property
    def warned(self) -> int:
        return sum(1 for f in self.findings if f.level == "warn")


def register(subparsers: Subparsers) -> None:
    """Add ``doctor`` (``narration.admin.cli``)."""
    parser = subparsers.add_parser(
        "doctor",
        help="check this machine and the configuration, and say what to fix",
        description=(
            "Check the platform, the configuration, the store, the models (hashed again), the worker venvs, the "
            "GPU and the engine pin, and say what to do about each problem. Exit code 1 if a check fails."
        ),
    )
    parser.add_argument("--quick", action="store_true", help="check the models' files are there, without hashing them")
    parser.add_argument("--json", action="store_true", help="print the findings as JSON")
    parser.set_defaults(handler=run_doctor)


def run_doctor(admin: Admin, args: argparse.Namespace, *, probes: Probes | None = None) -> int:
    """``doctor``: see the module docstring."""
    report = _Report(diagnose(admin, probes=probes, hash_models=not args.quick))
    if args.json:
        payload: dict[str, Any] = {"ok": report.failed == 0, "findings": [asdict(f) for f in report.findings]}
        admin.say(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        width = max(len(f.area) for f in report.findings)
        for finding in report.findings:
            mark = {"ok": "ok  ", "info": "info", "warn": "WARN", "fail": "FAIL"}[finding.level]
            admin.say(f"{mark}  {finding.area.ljust(width)}  {finding.summary}")
            if finding.next_step:
                admin.say(f"{'':4}  {'':{width}}  -> {finding.next_step}")
        if report.failed:
            admin.say(f"\n{report.failed} check(s) failed, {report.warned} warning(s).")
        elif report.warned:
            admin.say(f"\nNo check failed; {report.warned} warning(s).")
        else:
            admin.say("\nEverything checked is ready.")
    return EXIT_FAILED if report.failed else EXIT_OK
