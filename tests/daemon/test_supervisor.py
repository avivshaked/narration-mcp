"""The worker supervisor (sections 4, 4.1, 17): real fake-worker processes, the platform stand-in.

Every worker a test starts is closed by the supervisor's context (a ``shutdown``, then a kill), and the
stand-in's kill-on-close group kills whatever is left when it closes.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import psutil
import pytest

from narration.config import Config
from narration.contracts.errors import WorkerCrashed, WorkerFailure
from narration.contracts.names import WorkerRole
from narration.daemon.settings import NO_CWD_EXE_SEARCH
from narration.daemon.supervisor import (
    CRASH_LIMIT,
    FAKE_ROLES,
    ResidencyError,
    SupervisorClosed,
    WorkerSupervisor,
    console_python,
    harden_env,
    worker_creationflags,
    worker_cwd,
)
from narration.store import NarrationStore
from narration.workers import SubprocessWorkerClient, WorkerCommand, worker_command

from .conftest import fake_env
from .standin import StandInPlatform

pytestmark = pytest.mark.timeout(120)

SupervisorFactory = Callable[..., WorkerSupervisor]
LOAD: dict[str, Any] = {"device": "cpu"}


@pytest.fixture
def supervisors(
    config: Config, platform: StandInPlatform, store: NarrationStore, tmp_path: Path
) -> Iterator[SupervisorFactory]:
    """Opens fake-worker supervisors on the store; every one is closed at teardown."""
    opened: list[WorkerSupervisor] = []

    def make(spec: dict[str, Any] | None = None, **options: Any) -> WorkerSupervisor:
        options.setdefault("roles", FAKE_ROLES)
        options.setdefault("close_timeout_s", 5.0)
        supervisor = WorkerSupervisor(config, platform, base_env=fake_env(tmp_path, spec), **options)
        opened.append(supervisor.__enter__())
        return supervisor

    try:
        yield make
    finally:
        for supervisor in opened:
            supervisor.close()


def _alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


# ---------------------------------------------------------------- how a worker starts (sections 4, 4.1, 17)
def test_a_worker_joins_the_group_and_is_lowered_before_hello_s4_1(
    supervisors: SupervisorFactory, platform: StandInPlatform
) -> None:
    supervisor = supervisors()
    client = supervisor.client("qwen")
    assert client.pid is not None
    assert platform.added == [client.pid]
    assert platform.lowered == [client.pid]
    assert client.hello is not None and client.hello["role"] == "fake"


def test_normal_priority_when_the_config_says_so_s16(supervisors: SupervisorFactory, platform: StandInPlatform) -> None:
    client = supervisors(below_normal=False).client("qwen")
    assert platform.added == [client.pid]
    assert platform.lowered == []


def test_the_creation_flags_give_no_window_and_below_normal_s4_1() -> None:
    if os.name == "nt":
        assert worker_creationflags(below_normal=True) == (
            subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
        )
        assert worker_creationflags(below_normal=False) == subprocess.CREATE_NO_WINDOW
    else:
        assert worker_creationflags(below_normal=True) == 0


def test_a_worker_runs_in_its_project_folder_with_the_cmd_search_off_s17(
    config: Config, platform: StandInPlatform, store: NarrationStore, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    class Capture(SubprocessWorkerClient):
        def __init__(self, command: WorkerCommand, **options: Any) -> None:
            seen.update(command=command, **options)
            super().__init__(command, **options)

    project = config.service_root / "workers" / "qwen3tts" if config.service_root else None
    assert project is not None and project.is_dir()
    with WorkerSupervisor(
        config, platform, roles=FAKE_ROLES, base_env=fake_env(tmp_path), client_factory=Capture
    ) as sup:
        client = sup.client("qwen")
        assert seen["cwd"] == project
        env = seen["command"].env
        if os.name == "nt":
            assert env[NO_CWD_EXE_SEARCH] == "1"
        else:
            assert NO_CWD_EXE_SEARCH not in env
        assert env["PATH"] == os.environ["PATH"], "the operator's PATH is kept"
        # The process itself: its working directory and environment, read from the worker this test started.
        assert client.pid is not None
        process = psutil.Process(client.pid)
        assert os.path.normcase(process.cwd()) == os.path.normcase(str(project))
        if os.name == "nt":  # Windows keeps names case-insensitively (this one arrives upper-cased)
            names = {name.upper(): value for name, value in process.environ().items()}
            assert names.get(NO_CWD_EXE_SEARCH.upper()) == "1"
        assert config.server.store_root not in (Path(process.cwd()),)


def test_worker_cwd_is_never_the_store_root_s17(config: Config, tmp_path: Path) -> None:
    python = tmp_path / "venv" / "python.exe"
    assert config.service_root is not None
    assert worker_cwd(config, "qa", "qa", python) == config.service_root / "workers" / "qa"
    assert worker_cwd(config, "qa", "fake", python) == config.service_root / "workers" / "qa"
    bare = Config.for_tests(tmp_path / "store")  # no config file: no service root, no projects
    assert worker_cwd(bare, "qwen", "fake", python) == python.parent


def test_harden_env_adds_only_the_cmd_switch_s17() -> None:
    env = {"PATH": "a;b", "X": "1"}
    hardened = harden_env(env)
    extra = {NO_CWD_EXE_SEARCH: "1"} if os.name == "nt" else {}
    assert hardened == {**env, **extra}


def test_the_fake_worker_speaks_through_a_console_python_s4_1() -> None:
    assert console_python().name.lower() in ("python.exe", "python", Path(sys.executable).name.lower())


# ---------------------------------------------------------------- crashes and start failures (section 14)
def test_a_crashed_worker_starts_again_on_its_next_use_appA(supervisors: SupervisorFactory) -> None:
    supervisor = supervisors({"faults": [{"kind": "crash", "op": "unload", "times": 1}]})
    first = supervisor.client("qwen")
    supervisor.load("qwen", LOAD, timeout_s=30)
    with pytest.raises(WorkerCrashed):
        first.request("unload", {}, timeout_s=30)
    supervisor.poll()
    assert supervisor.loaded() == frozenset(), "a dead worker's models are gone"
    second = supervisor.client("qwen")
    assert second is not first and second.pid != first.pid
    assert second.request("unload", {}, timeout_s=30)["ok"] is True


def test_a_worker_that_keeps_crashing_is_not_started_in_a_loop_appA(
    supervisors: SupervisorFactory, platform: StandInPlatform
) -> None:
    supervisor = supervisors({"faults": [{"kind": "crash", "op": "unload"}]})
    for _ in range(CRASH_LIMIT):
        client = supervisor.client("qwen")
        with pytest.raises(WorkerCrashed):
            client.request("unload", {}, timeout_s=30)
    started = len(platform.added)
    with pytest.raises(WorkerCrashed, match="not started again"):
        supervisor.client("qwen")
    assert len(platform.added) == started


def test_a_worker_that_cannot_start_is_backend_not_installed_for_good_s14(
    supervisors: SupervisorFactory, platform: StandInPlatform
) -> None:
    supervisor = supervisors({"faults": "not a list"})  # an invalid spec: the fake exits 2 at start
    with pytest.raises(WorkerFailure) as first:
        supervisor.client("qa")
    assert first.value.code == "BACKEND_NOT_INSTALLED"
    started = len(platform.added)
    with pytest.raises(WorkerFailure) as again:
        supervisor.client("qa")
    assert again.value is first.value, "the same failure, without another attempt"
    assert len(platform.added) == started
    assert supervisor.failure("qa") is first.value


def test_a_missing_worker_venv_is_backend_not_installed_without_a_start_s14(
    config: Config, platform: StandInPlatform
) -> None:
    calls: list[WorkerRole] = []

    def command(role: WorkerRole, cublas: str | None) -> WorkerCommand:
        calls.append(role)
        return worker_command(config, role)  # qwen3's venv does not exist here

    with WorkerSupervisor(config, platform, command_factory=command) as supervisor:
        for _ in range(2):
            with pytest.raises(WorkerFailure) as info:
                supervisor.client("qwen")
            assert info.value.code == "BACKEND_NOT_INSTALLED"
    assert calls == ["qwen3"], "never retried"
    assert platform.added == []


# ---------------------------------------------------------------- residency (section 4, GPU scheduler item 1)
def test_one_gpu_group_at_a_time_and_the_choice_is_the_callers_s4(supervisors: SupervisorFactory) -> None:
    supervisor = supervisors()
    supervisor.load("qwen", LOAD, timeout_s=30)
    assert supervisor.gpu_holder == "qwen"
    with pytest.raises(ResidencyError, match="unload"):
        supervisor.load("qa", LOAD, timeout_s=30)
    supervisor.load("qa", LOAD, gpu=False, timeout_s=30)  # a CPU load (the aligner) is not a GPU group
    assert supervisor.loaded() == {"qwen", "qa"}
    assert supervisor.gpu_holder == "qwen"
    supervisor.unload("qwen", timeout_s=30)
    assert supervisor.gpu_holder is None
    supervisor.load("qa", LOAD, timeout_s=30)
    assert supervisor.gpu_holder == "qa"


def test_two_gpu_groups_when_the_config_allows_it_s16(
    config: Config, platform: StandInPlatform, store: NarrationStore, tmp_path: Path
) -> None:
    import dataclasses

    from narration.config import GpuConfig

    relaxed = dataclasses.replace(config, gpu=GpuConfig(one_group_at_a_time=False))
    with WorkerSupervisor(relaxed, platform, roles=FAKE_ROLES, base_env=fake_env(tmp_path)) as supervisor:
        supervisor.load("qwen", LOAD, timeout_s=30)
        supervisor.load("qa", LOAD, timeout_s=30)
        assert supervisor.loaded() == {"qwen", "qa"}


def test_release_all_stops_every_worker_and_names_the_holder_s7_6(supervisors: SupervisorFactory) -> None:
    changes: list[int] = []
    supervisor = supervisors(on_change=lambda: changes.append(1))
    supervisor.load("qwen", LOAD, timeout_s=30)
    qa = supervisor.client("qa")
    assert {w.role for w in supervisor.workers()} == {"fake"} and len(supervisor.workers()) == 2
    assert supervisor.release_all() == "qwen"
    assert supervisor.workers() == ()
    assert supervisor.loaded() == frozenset() and supervisor.gpu_holder is None
    assert qa.exit_code is not None
    assert changes, "every change is reported"


def test_a_new_cublas_pin_restarts_the_worker_s10_1(supervisors: SupervisorFactory) -> None:
    supervisor = supervisors()
    first = supervisor.client("qwen", cublas_workspace_config=":4096:8")
    assert supervisor.client("qwen") is first, "no pin asked: the running worker serves"
    assert supervisor.client("qwen", cublas_workspace_config=":4096:8") is first
    second = supervisor.client("qwen", cublas_workspace_config=":16:8")
    assert second is not first and first.exit_code is not None


# ---------------------------------------------------------------- stopping (section 4.1)
def test_kill_all_ends_every_worker_and_starts_no_more_s4_1(supervisors: SupervisorFactory) -> None:
    supervisor = supervisors()
    pids = [supervisor.client(group).pid for group in ("qwen", "qa")]
    supervisor.kill_all()
    assert supervisor.closed
    assert not any(pid is not None and _alive(pid) for pid in pids)
    with pytest.raises(SupervisorClosed):
        supervisor.client("qwen")


def test_close_shuts_the_workers_down_and_closes_the_group_s4(
    supervisors: SupervisorFactory, platform: StandInPlatform
) -> None:
    supervisor = supervisors()
    client = supervisor.client("qwen")
    supervisor.close()
    assert client.exit_code == 0, "a clean shutdown, not a kill"
    assert platform.groups_closed == platform.groups_opened
