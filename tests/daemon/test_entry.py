"""The entry point, ``python -m narration.daemon`` (sections 4, 4.1): run in subprocesses this test starts and
waits for (each exits by itself within seconds)."""

from __future__ import annotations

import importlib
import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from narration_worker.threads import THREAD_ENV_VARS

from narration.config import load_config
from narration.daemon.__main__ import DEFAULT_RUNNER, launch_time, warn_without_safe_path
from narration.daemon.seam import JobRunner, NullRunner
from narration.daemon.start import daemon_argv
from narration.jobs.runner import EngineRunner, log_path

pytestmark = pytest.mark.timeout(120)

# A MetaPathFinder that records the thread variables at the moment numpy is first imported, then runs the
# daemon's entry point as ``python -m narration.daemon`` would.
WATCH = textwrap.dedent(
    """
    import importlib.abc, json, os, runpy, sys

    OUT = sys.argv[1]
    VARS = json.loads(sys.argv[2])

    class Watch(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "numpy" and not os.path.exists(OUT):
                with open(OUT, "w", encoding="utf-8") as out:
                    json.dump({var: os.environ.get(var) for var in VARS}, out)
            return None

    sys.meta_path.insert(0, Watch())
    sys.argv = ["narration.daemon", *sys.argv[3:]]
    runpy.run_module("narration.daemon", run_name="__main__", alter_sys=True)
    """
)


def write_config(folder: Path, *, cpu_threads: int = 3, store: str = "store") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "narration.toml"
    path.write_text(
        f"[server]\nstore_root = '{store}'\nmodels_root = 'models'\n\n[workers]\ncpu_threads = {cpu_threads}\n",
        encoding="utf-8",
    )
    return path


def run(argv: list[str], cwd: Path, *, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in THREAD_ENV_VARS}
    env.update(extra_env or {})
    return subprocess.run(
        argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=90, check=False
    )


def test_importing_the_package_loads_no_numpy_s4_1(tmp_path: Path) -> None:
    done = run([sys.executable, "-c", "import sys, narration.daemon; print('numpy' in sys.modules)"], tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False"


def test_the_thread_cap_is_set_before_numpy_is_imported_s4_1(tmp_path: Path) -> None:
    config = write_config(tmp_path / "service", cpu_threads=3)
    script = tmp_path / "watch.py"
    script.write_text(WATCH, encoding="utf-8")
    seen = tmp_path / "at-numpy-import.json"
    store = tmp_path / "service" / "store"
    done = run(
        [
            sys.executable,
            str(script),
            str(seen),
            json.dumps(list(THREAD_ENV_VARS)),
            "--store",
            str(store),
            "--config",
            str(config),
            "--idle-exit-s",
            "0",
            "--poll-s",
            "0.05",
        ],
        tmp_path,
        extra_env={"OPENBLAS_NUM_THREADS": "64"},  # the daemon's own cap must win over what it inherits
    )
    assert seen.exists(), f"numpy was never imported: {done.stderr}"
    assert json.loads(seen.read_text(encoding="utf-8")) == dict.fromkeys(THREAD_ENV_VARS, "3")
    if sys.platform == "win32":
        assert done.returncode == 0, done.stderr


def test_a_module_planted_in_the_store_root_is_never_imported_s17(tmp_path: Path) -> None:
    """The daemon runs in the store root (``start_detached``'s cwd). ``python -m`` puts the working directory
    first on ``sys.path``; with ``-P`` it does not, so a ``narration`` package planted there is never run."""
    config = write_config(tmp_path / "service")
    store = tmp_path / "service" / "store"
    marker = tmp_path / "the planted package ran"
    (store / "narration").mkdir(parents=True)
    (store / "narration" / "__init__.py").write_text(f"open({str(marker)!r}, 'w').close()\n", encoding="utf-8")
    argv = daemon_argv(store, config, python=Path(sys.executable), extra=["--idle-exit-s", "0", "--poll-s", "0.05"])
    done = run(argv, store)
    assert not marker.exists(), done.stderr
    if sys.platform == "win32":
        assert done.returncode == 0, done.stderr


def test_a_bad_configuration_is_a_usage_error_s16(tmp_path: Path) -> None:
    bad = tmp_path / "narration.toml"
    bad.write_text(
        "[server]\nstore_root = 'store'\nmodels_root = 'models'\n[daemon]\nidle_exit_min = -1\n", encoding="utf-8"
    )
    done = run(
        [sys.executable, "-m", "narration.daemon", "--store", str(tmp_path / "store"), "--config", str(bad)], tmp_path
    )
    assert done.returncode == 2
    assert "idle_exit_min" in done.stderr
    assert not (tmp_path / "store").exists(), "nothing was created"


def test_the_store_must_be_the_configurations_s16(tmp_path: Path) -> None:
    config = write_config(tmp_path / "service")
    done = run(
        [sys.executable, "-m", "narration.daemon", "--store", str(tmp_path / "elsewhere"), "--config", str(config)],
        tmp_path,
    )
    assert done.returncode == 2
    assert "store_root" in done.stderr


def test_the_launch_time_is_the_launchers_unless_it_is_later_than_the_process_began_s4_1() -> None:
    assert launch_time(None, 100.0) == 100.0, "no launcher: when this process began"
    assert launch_time(99.5, 100.0) == 99.5, "the launcher's clock, read just before the spawn"
    assert launch_time(250.0, 100.0) == 100.0, "a launch after the process began is not believed"


@pytest.mark.parametrize("given", [float("nan"), float("inf"), float("-inf")])
def test_a_launch_time_that_is_not_finite_is_ignored_and_logged_s4_1(
    given: float, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="narration.daemon"):
        assert launch_time(given, 100.0) == 100.0, "when this process began, as with no --launched-at"
    assert any("not a finite time" in record.getMessage() for record in caplog.records)


def test_a_daemon_started_without_safe_path_says_so_once_s17(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="narration.daemon"):
        assert warn_without_safe_path(True) is False
        assert caplog.records == [], "started with -P: nothing to say"
        assert warn_without_safe_path(False) is True
    (record,) = caplog.records
    message = record.getMessage()
    assert "without -P" in message and "sys.path" in message
    assert "narration-admin daemon start" in message and "start_detached" in message


def test_the_default_runner_is_the_job_engine_s4() -> None:
    module, _, attr = DEFAULT_RUNNER.partition(":")
    assert (module, attr) == ("narration.jobs.runner", "default_runner")
    runner = getattr(importlib.import_module(module), attr)()  # as the entry point loads it
    assert isinstance(runner, JobRunner) and isinstance(runner, EngineRunner)
    assert isinstance(NullRunner(), JobRunner)  # still there, for --runner narration.daemon.seam:NullRunner


@pytest.mark.skipif(sys.platform != "win32", reason="the daemon runs on Windows only in v1 (plan.md Q2)")
def test_an_unknown_runner_is_a_usage_error_and_is_logged_s4(tmp_path: Path) -> None:
    config = write_config(tmp_path / "service")
    store = tmp_path / "service" / "store"
    done = run(
        [
            sys.executable,
            "-m",
            "narration.daemon",
            "--store",
            str(store),
            "--config",
            str(config),
            "--runner",
            "no_such_module:Runner",
        ],
        tmp_path,
    )
    assert done.returncode == 2
    # The file the daemon logs to is the one the job engine names in an INTERNAL error (section 14).
    assert "no_such_module" in log_path(load_config(config)).read_text(encoding="utf-8")


@pytest.mark.skipif(sys.platform == "win32", reason="checks the behaviour off Windows")
def test_off_windows_the_daemon_says_the_platform_is_unsupported_s4(tmp_path: Path) -> None:
    config = write_config(tmp_path / "service")
    done = run(
        [
            sys.executable,
            "-m",
            "narration.daemon",
            "--store",
            str(tmp_path / "service" / "store"),
            "--config",
            str(config),
        ],
        tmp_path,
    )
    assert done.returncode == 1
    assert "DAEMON_UNAVAILABLE" in done.stderr
