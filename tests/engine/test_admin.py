"""``narration-admin engine`` as the operator runs it (design sections 7.1 and 10.1; plan.md WP32, WP37): the
commands registered on a parser, their output, their exit codes and what they say to do next."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from narration.config import Config
from narration.contracts.interfaces import WorkerClient
from narration.engine.admin import CONFIG_ENV, EXIT_FAILED, EXIT_OK, EXIT_USAGE, Environment, register
from narration.engine.models import QWEN_BASE
from narration.engine.pinning import SubprocessStarter
from narration.jobs.gpu import NoProbe, VramProbe, VramReading
from tests.store.standin import StandInPlatform

from .support import make_install

WORKER_PACKAGES = ("narration-worker",)


class _Platform(StandInPlatform):
    """The stand-in platform, with the daemon's singleton free (``held``) or taken."""

    def __init__(self, held: bool = True) -> None:
        super().__init__()
        self.held = held

    def singleton(self, store_root: Path) -> contextlib.AbstractContextManager[bool]:
        return contextlib.nullcontext(self.held)


class _Fake:
    def __init__(self, config: Config) -> None:
        base = {k: v for k, v in os.environ.items() if k != "NARRATION_FAKE_SPEC"}
        self.inner = SubprocessStarter(config, role="fake", base_env=base)

    def start(self, role: Any, *, cublas_workspace_config: str | None = None) -> WorkerClient:
        return self.inner.start(role, cublas_workspace_config=cublas_workspace_config)


class _Probe:
    def __init__(self, free_mb: int) -> None:
        self.free_mb = free_mb

    def read(self) -> VramReading | None:
        return VramReading(name="Test GPU", total_mb=24_000, free_mb=self.free_mb)


def _env(*, held: bool = True, probe: VramProbe | None = None) -> Environment:
    return Environment(
        platform=lambda: _Platform(held),
        starter=_Fake,
        probe=lambda config: probe if probe is not None else NoProbe(),
        packages=WORKER_PACKAGES,
    )


def _run(env: Environment, *argv: str) -> int:
    parser = argparse.ArgumentParser(prog="narration-admin")
    register(parser.add_subparsers(dest="group", required=True), env)
    args = parser.parse_args(list(argv))
    return args.handler(args)


@pytest.fixture
def config_file(tmp_path: Path) -> Iterator[Path]:
    install = make_install(tmp_path, versions={"narration-worker": importlib.metadata.version("narration-worker")})
    path = tmp_path / "narration.toml"
    path.write_text(
        "\n".join(
            [
                "[server]",
                f"store_root = {json.dumps(str(install.store_root))}",
                f"models_root = {json.dumps(str(install.models_root))}",
                "[gpu]",
                'device = "cpu"',
                "[workers.qwen3]",
                f"project = {json.dumps(str(install.project))}",
            ]
        ),
        encoding="utf-8",
    )
    yield path


def test_engine_pin_prints_what_it_pinned_s10_1(config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(_env(), "engine", "pin", "--config", str(config_file)) == EXIT_OK
    out = capsys.readouterr().out
    assert "qwen3-design-1.7b.p1  new  sha256:" in out and "qwen3-base-1.7b.p1  new  sha256:" in out
    assert "tier bit_exact" in out and "canary threshold" in out

    assert _run(_env(), "engine", "pin", "--config", str(config_file), "--json") == EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert [e["action"] for e in data["engines"]] == ["keep", "keep"]


def test_engine_show_lists_the_profiles_in_use_s10_1(config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(_env(), "engine", "show", "--config", str(config_file)) == EXIT_OK
    assert "engine pin" in capsys.readouterr().out  # nothing pinned yet: it says what to run
    _run(_env(), "engine", "pin", "--config", str(config_file))
    capsys.readouterr()
    assert _run(_env(), "engine", "show", "--config", str(config_file), "--json") == EXIT_OK
    rows = {r["engine_profile_id"]: r for r in json.loads(capsys.readouterr().out)["profiles"]}
    assert rows["qwen3-base-1.7b.p1"]["in_use_for"] == "base"
    assert rows["qwen3-design-1.7b.p1"]["in_use_for"] == "design"
    assert rows["qwen3-base-1.7b.p1"]["canary"]["threshold"] is not None


def test_the_config_can_come_from_the_environment(
    config_file: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CONFIG_ENV, str(config_file))
    assert _run(_env(), "engine", "show") == EXIT_OK


def test_no_config_is_a_usage_error_that_says_how_to_give_one(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    assert _run(_env(), "engine", "pin") == EXIT_USAGE
    assert "--config" in capsys.readouterr().err


def test_a_bad_config_is_a_usage_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "narration.toml"
    path.write_text("[server]\n", encoding="utf-8")
    assert _run(_env(), "engine", "pin", "--config", str(path)) == EXIT_USAGE
    assert "store_root" in capsys.readouterr().err


def test_pin_refuses_while_a_daemon_holds_the_store_s4(config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(_env(held=False), "engine", "pin", "--config", str(config_file)) == EXIT_FAILED
    err = capsys.readouterr().err
    assert "daemon is running" in err and "next: Stop it (narration-admin daemon stop" in err


def test_pin_refuses_without_the_vram_a_qwen_load_needs_s4(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(_env(probe=_Probe(free_mb=2000)), "engine", "pin", "--config", str(config_file)) == EXIT_FAILED
    err = capsys.readouterr().err
    assert "2000 MB free" in err and "next:" in err


def test_a_changed_installation_is_refused_and_repin_makes_the_new_profile_s10_1(
    config_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(_env(), "engine", "pin", "--config", str(config_file))
    capsys.readouterr()
    (QWEN_BASE.snapshot_dir(tmp_path / "models") / "model.safetensors").write_bytes(b"new weights")

    assert _run(_env(), "engine", "pin", "--config", str(config_file)) == EXIT_FAILED
    err = capsys.readouterr().err
    assert "differs from the pinned engine profile qwen3-base-1.7b.p1 in: weights" in err
    assert "next: To make a new profile the one in use, run narration-admin engine repin" in err

    assert _run(_env(), "engine", "repin", "--config", str(config_file)) == EXIT_OK
    out = capsys.readouterr().out
    assert "qwen3-base-1.7b.p2  new" in out and "changed: weights" in out and "measured again" in out

    assert (
        _run(_env(), "engine", "bridge", "qwen3-base-1.7b.p1", "qwen3-base-1.7b.p2", "--config", str(config_file)) == 0
    )
    out = capsys.readouterr().out
    assert "qwen3-base-1.7b.p1 -> qwen3-base-1.7b.p2" in out and "cannot render here: weights" in out


def test_a_worker_that_is_not_installed_says_to_run_doctor(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = Environment(
        platform=lambda: _Platform(True),
        starter=SubprocessStarter,  # the real qwen3 and QA workers, whose venvs this installation lacks
        probe=lambda config: NoProbe(),
        packages=WORKER_PACKAGES,
    )
    assert _run(env, "engine", "pin", "--config", str(config_file)) == EXIT_FAILED
    assert "narration-admin doctor" in capsys.readouterr().err
