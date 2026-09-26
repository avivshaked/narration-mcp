"""Determinism switches (section 10.1), the CPU thread cap (section 4.1) and the fingerprint, without torch.

torch is not installed where these run by default; each test passes a stand-in module, so what is checked
is that the right switches are set, not torch itself. The worker venvs' own tests check the real thing.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from narration_worker import determinism, fingerprint, threads
from narration_worker import handler as handler_module
from narration_worker.errors import OpError, is_transient_load_error
from narration_worker.handler import WorkerContext, WorkerHandler


class _FakeTorch:
    """Records what the worker sets on it."""

    def __init__(self, *, cuda_available: bool = False, interop_locked: bool = False) -> None:
        self.backends = SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(
                allow_tf32=True,
                deterministic=False,
                benchmark=True,
                is_available=lambda: True,
                version=lambda: 91002,
            ),
        )
        self.version = SimpleNamespace(cuda="12.8")
        self.deterministic: tuple[bool, bool] | None = None
        self.threads = 0
        self.interop = 0
        self.interop_locked = interop_locked
        self.seeds: list[tuple[str, int]] = []
        self.cuda = SimpleNamespace(
            is_available=lambda: cuda_available,
            is_initialized=lambda: False,
            manual_seed_all=lambda s: self.seeds.append(("cuda", s)),
            get_device_name=lambda: "never read before CUDA starts",
        )

    def use_deterministic_algorithms(self, mode: bool, *, warn_only: bool = False) -> None:
        self.deterministic = (mode, warn_only)

    def set_num_threads(self, n: int) -> None:
        self.threads = n

    def set_num_interop_threads(self, n: int) -> None:
        if self.interop_locked:
            raise RuntimeError("Error: cannot set number of interop threads after parallel work has started")
        self.interop = n

    def get_num_threads(self) -> int:
        return self.threads

    def get_num_interop_threads(self) -> int:
        return self.interop

    def manual_seed(self, seed: int) -> None:
        self.seeds.append(("cpu", seed))


SWITCHES = {
    "tf32": False,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "deterministic_algorithms": "warn_only",
}


def test_prepare_cuda_env_sets_the_pinned_cublas_workspace_s10_1() -> None:
    env: dict[str, str] = {}
    assert determinism.prepare_cuda_env(env) == ":4096:8"
    assert env == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    kept = {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}
    assert determinism.prepare_cuda_env(kept) == ":16:8"


def test_the_load_switches_are_applied_to_torch_s10_1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = _FakeTorch()
    applied = determinism.apply_determinism(determinism.parse_determinism(dict(SWITCHES)), torch)
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.deterministic == (True, True)
    assert applied["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


@pytest.mark.parametrize(("mode", "expected"), [("on", (True, False)), ("off", (False, False))])
def test_deterministic_algorithms_modes_s10_1(
    monkeypatch: pytest.MonkeyPatch, mode: str, expected: tuple[bool, bool]
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = _FakeTorch()
    determinism.apply_determinism(determinism.parse_determinism({**SWITCHES, "deterministic_algorithms": mode}), torch)
    assert torch.deterministic == expected


def test_deterministic_algorithms_refuse_to_run_without_the_cublas_workspace_s10_1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(OpError) as caught:
        determinism.apply_determinism(determinism.parse_determinism(dict(SWITCHES)), _FakeTorch())
    assert caught.value.code == "INTERNAL"


@pytest.fixture
def broken_modules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Modules that are installed but cannot be loaded, the ways real ones fail."""
    sources = {
        "narration_test_dll_fails": "raise OSError('[WinError 126] The specified module could not be found')\n",
        "narration_test_dependency_missing": "import narration_test_absent_dependency\n",
        "narration_test_extension_fails": "raise ImportError('DLL load failed while importing _C')\n",
    }
    for name, source in sources.items():
        (tmp_path / f"{name}.py").write_text(source, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))


def test_a_file_another_process_holds_is_transient_not_backend_not_installed_s4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    in_use = "The process cannot access the file because it is being used by another process"
    (tmp_path / "narration_test_dll_in_use.py").write_text(
        f"raise ImportError('DLL load failed while importing _C: {in_use}.')\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(OpError) as caught:
        determinism.import_optional("narration_test_dll_in_use")
    assert caught.value.code == "INTERNAL" and caught.value.details["transient"] is True
    assert is_transient_load_error(ImportError(f"DLL load failed while importing _C: {in_use}."))
    assert is_transient_load_error(OSError(f"[WinError 32] {in_use}: 'c10.dll'"))
    assert not is_transient_load_error(ImportError("DLL load failed while importing _C: not found"))
    assert not is_transient_load_error(OSError("[WinError 126] The specified module could not be found"))


def test_a_broken_nvml_binding_leaves_gpu_and_driver_unknown_not_a_failed_hello_s10_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NVML is optional: a binding that cannot load is an unreadable fact, not a broken worker."""
    (tmp_path / "pynvml.py").write_text(
        "raise OSError('[WinError 126] nvml.dll could not be found')\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "pynvml", raising=False)
    monkeypatch.setattr(handler_module, "import_optional", _only_nvml_is_real)

    class _TorchQa(WorkerHandler):
        role = "qa"

    fingerprint = _TorchQa(WorkerContext(role="qa", store_root=tmp_path, cpu_threads=2)).fingerprint()
    assert (fingerprint["gpu"], fingerprint["driver"]) == (None, None)
    assert fingerprint["cuda"] == "12.8"  # torch's facts are still read


def _only_nvml_is_real(name: str) -> Any:
    """The real ``import_optional`` for NVML; a stand-in torch for torch (not installed here)."""
    return determinism.import_optional(name) if name == "pynvml" else _FakeTorch()


def test_a_module_that_is_not_installed_is_none_s4(broken_modules: None) -> None:
    assert determinism.import_optional("narration_test_not_installed_at_all") is None
    assert determinism.import_optional("narration_test_not_installed_at_all.sub") is None


@pytest.mark.parametrize(
    ("name", "error"),
    [
        ("narration_test_dll_fails", "OSError: [WinError 126] The specified module could not be found"),
        (
            "narration_test_dependency_missing",
            "ModuleNotFoundError: No module named 'narration_test_absent_dependency'",
        ),
        ("narration_test_extension_fails", "ImportError: DLL load failed while importing _C"),
    ],
)
def test_a_module_installed_but_unloadable_is_backend_not_installed_s4(
    broken_modules: None, name: str, error: str
) -> None:
    """torch's DLLs failing on Windows raise OSError: that is a broken env, not "no torch" and not INTERNAL."""
    with pytest.raises(OpError) as caught:
        determinism.import_optional(name)
    assert caught.value.code == "BACKEND_NOT_INSTALLED"
    assert caught.value.details == {"module": name, "error": error}
    assert "narration-admin install" in caught.value.message


def test_without_torch_nothing_is_applied_s10_1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(determinism, "import_optional", lambda name: None)
    assert determinism.apply_determinism(determinism.parse_determinism(dict(SWITCHES))) == {"torch": False}


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        {**SWITCHES, "extra": 1},
        {k: v for k, v in SWITCHES.items() if k != "tf32"},
        {**SWITCHES, "tf32": 0},
        {**SWITCHES, "deterministic_algorithms": True},
    ],
)
def test_malformed_determinism_is_invalid_request_appA(bad: Any) -> None:
    with pytest.raises(OpError) as caught:
        determinism.parse_determinism(bad)
    assert caught.value.code == "INVALID_REQUEST"


def test_seed_everything_seeds_random_numpy_torch_and_cuda_s10_3() -> None:
    torch = _FakeTorch(cuda_available=True)
    numpy = SimpleNamespace(random=SimpleNamespace(seed=lambda s: torch.seeds.append(("numpy", s))))
    seeded = determinism.seed_everything(1834112093, torch=torch, numpy=numpy)
    assert seeded == ["random", "numpy", "torch", "torch.cuda"]
    assert torch.seeds == [("numpy", 1834112093), ("cpu", 1834112093), ("cuda", 1834112093)]
    first = random.random()
    determinism.seed_everything(1834112093, torch=torch, numpy=numpy)
    assert random.random() == first


def test_the_thread_cap_sets_every_pool_variable_s4_1() -> None:
    env: dict[str, str] = {}
    threads.cap_threads_env(6, env)
    assert env == dict.fromkeys(threads.THREAD_ENV_VARS, "6")
    assert {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"} <= set(env)
    with pytest.raises(ValueError):
        threads.thread_env(0)


@pytest.mark.parametrize(("cap", "interop"), [(8, 4), (2, 2)])
def test_the_thread_cap_is_applied_to_torch_s4_1(cap: int, interop: int) -> None:
    torch = _FakeTorch()
    assert threads.cap_torch_threads(cap, torch) == {"intra_op": cap, "inter_op": interop}


def test_torch_refusing_the_interop_size_is_logged_not_fatal_s4_1() -> None:
    torch = _FakeTorch(interop_locked=True)
    assert threads.cap_torch_threads(8, torch) == {"intra_op": 8, "inter_op": 0}


def test_the_fingerprint_without_torch_reports_unknown_cuda_s10_1() -> None:
    env = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "HF_HUB_OFFLINE": "1", "UNRELATED": "x", "OMP_NUM_THREADS": "8"}
    fp = fingerprint.collect(cpu_threads=8, packages=("no-such-distribution-here",), environ=env)
    assert set(fp) == {"python", "platform", "packages", "cuda", "cudnn", "gpu", "driver", "cpu_threads", "env"}
    assert fp["python"].startswith("3.12")
    assert "narration-worker" in fp["packages"]
    assert "no-such-distribution-here" not in fp["packages"]
    assert (fp["cuda"], fp["cudnn"], fp["gpu"], fp["driver"]) == (None, None, None, None)
    assert fp["env"] == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "8"}


def test_the_fingerprint_reads_cuda_facts_without_starting_cuda_s10_1() -> None:
    fp = fingerprint.collect(cpu_threads=4, torch=_FakeTorch(), environ={})
    assert (fp["cuda"], fp["cudnn"], fp["gpu"], fp["driver"]) == ("12.8", "91002", None, None)


def test_the_fingerprint_takes_gpu_and_driver_from_nvml_s10_1() -> None:
    calls: list[str] = []
    nvml = SimpleNamespace(
        nvmlInit=lambda: calls.append("init"),
        nvmlShutdown=lambda: calls.append("shutdown"),
        nvmlSystemGetDriverVersion=lambda: b"591.86",
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetName=lambda handle: f"GPU {handle}",
    )
    fp = fingerprint.collect(cpu_threads=4, torch=_FakeTorch(), nvml=nvml, environ={"CUDA_VISIBLE_DEVICES": "1"})
    assert (fp["gpu"], fp["driver"]) == ("GPU 1", "591.86")
    assert calls == ["init", "shutdown"]


def test_an_unreadable_fact_is_unknown_not_a_failure_s10_1() -> None:
    def broken() -> None:
        raise RuntimeError("NVML not found")

    nvml = SimpleNamespace(nvmlInit=broken, nvmlShutdown=broken)
    fp = fingerprint.collect(cpu_threads=4, nvml=nvml, environ={})
    assert (fp["gpu"], fp["driver"]) == (None, None)


def test_a_torch_role_gets_torch_with_the_thread_cap_applied_once_s4_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch = _FakeTorch()
    imports: list[str] = []

    def fake_import(name: str) -> Any:
        imports.append(name)
        return torch if name == "torch" else None

    monkeypatch.setattr(handler_module, "import_optional", fake_import)

    class _TorchQa(WorkerHandler):
        role = "qa"

    worker = _TorchQa(WorkerContext(role="qa", store_root=tmp_path, cpu_threads=6))
    assert worker.torch() is torch
    assert worker.torch() is torch
    assert (torch.threads, torch.interop) == (6, 4)
    assert imports.count("torch") == 1
    fingerprint = worker.fingerprint()
    assert (fingerprint["cuda"], fingerprint["cudnn"]) == ("12.8", "91002")
    assert imports.count("torch") == 1
