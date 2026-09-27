"""``narration-admin engine`` as the operator runs it (design sections 7.1 and 10.1; plan.md WP32, WP37): the
commands registered on a parser and run as the dispatcher runs them (``handler(admin, args)``), their output,
their exit codes and what they say to do next.

``_Admin`` stands in for the dispatcher's ``narration.admin.cli.Admin`` (the ``AdminContext`` protocol) until
WP37 is merged; the tests then run through its ``main``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from narration.config import Config, GpuConfig
from narration.engine.admin import EXIT_FAILED, EXIT_OK, Environment, register
from narration.engine.models import QWEN_BASE
from narration.engine.pinning import SubprocessStarter
from narration.jobs.gpu import NoProbe, VramProbe, VramReading
from narration.platform.testing import StandInPlatform
from narration.store import NarrationStore
from narration.store.layout import DB_NAME

from .support import FAKE_WORKER_PACKAGES, CountingStarter, Install, fake_install


@dataclasses.dataclass
class _Admin:
    """The dispatcher's context for one command: a configuration, a platform, the store on first use, and
    what the command printed."""

    configured: Config
    platform_: StandInPlatform = dataclasses.field(default_factory=StandInPlatform)
    out: list[str] = dataclasses.field(default_factory=list)
    err: list[str] = dataclasses.field(default_factory=list)
    _store: NarrationStore | None = None

    def config(self) -> Config:
        return self.configured

    def platform(self) -> StandInPlatform:
        return self.platform_

    def store_exists(self) -> bool:
        return (self.configured.server.store_root / DB_NAME).is_file()

    def store(self) -> NarrationStore:
        if self._store is None:
            self._store = NarrationStore.from_config(self.configured, self.platform())
        return self._store

    def say(self, text: str = "") -> None:
        self.out.append(text)

    def warn(self, text: str) -> None:
        self.err.append(text)

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None

    @property
    def stdout(self) -> str:
        return "\n".join(self.out)

    @property
    def stderr(self) -> str:
        return "\n".join(self.err)


class _Probe:
    def __init__(self, free_mb: int) -> None:
        self.free_mb = free_mb

    def read(self) -> VramReading | None:
        return VramReading(name="Test GPU", total_mb=24_000, free_mb=self.free_mb)


def _env(*, probe: VramProbe | None = None) -> Environment:
    return Environment(
        starter=CountingStarter,
        probe=lambda config: probe if probe is not None else NoProbe(),
        packages=FAKE_WORKER_PACKAGES,
    )


def _run(config: Config, *argv: str, env: Environment | None = None, held: bool = True) -> tuple[int, _Admin]:
    """Parse ``argv`` as ``narration-admin`` would, and run its handler as the dispatcher does."""
    parser = argparse.ArgumentParser(prog="narration-admin")
    register(parser.add_subparsers(dest="group", required=True), env if env is not None else _env())
    args = parser.parse_args(list(argv))
    admin = _Admin(config)
    try:
        with contextlib.ExitStack() as stack:
            if not held:  # a daemon holds the store's singleton meanwhile
                stack.enter_context(admin.platform_.hold(config.server.store_root))
            return args.handler(admin, args), admin
    finally:
        admin.close()


@pytest.fixture
def install(tmp_path: Path) -> Install:
    return fake_install(tmp_path)


@pytest.fixture
def config(install: Install) -> Config:
    return dataclasses.replace(install.config, gpu=GpuConfig(device="cpu"))


def test_every_command_sets_a_handler_the_dispatcher_can_run_wp37(config: Config) -> None:
    parser = argparse.ArgumentParser(prog="narration-admin")
    register(parser.add_subparsers(dest="group", required=True))  # as the dispatcher calls it
    for argv in (["engine", "pin"], ["engine", "repin"], ["engine", "bridge", "a", "b"], ["engine", "show"]):
        handler: Any = parser.parse_args(argv).handler
        assert callable(handler) and handler.__code__.co_argcount == 2  # handler(admin, args)


def test_engine_pin_prints_what_it_pinned_s10_1(config: Config) -> None:
    code, admin = _run(config, "engine", "pin")
    assert code == EXIT_OK, admin.stderr
    assert "qwen3-design-1.7b.p1  new  sha256:" in admin.stdout and "qwen3-base-1.7b.p1  new  sha256:" in admin.stdout
    assert "tier bit_exact" in admin.stdout and "canary threshold" in admin.stdout

    code, admin = _run(config, "engine", "pin", "--json")
    assert code == EXIT_OK
    assert [e["action"] for e in json.loads(admin.stdout)["engines"]] == ["keep", "keep"]


def test_engine_show_lists_the_profiles_in_use_s10_1(config: Config) -> None:
    code, admin = _run(config, "engine", "show")
    assert code == EXIT_OK and "engine pin" in admin.stdout  # nothing pinned yet: it says what to run
    assert not (config.server.store_root / DB_NAME).exists()  # and showing created no store

    _run(config, "engine", "pin")
    code, admin = _run(config, "engine", "show", "--json")
    assert code == EXIT_OK
    rows = {r["engine_profile_id"]: r for r in json.loads(admin.stdout)["profiles"]}
    assert rows["qwen3-base-1.7b.p1"]["in_use_for"] == "base"
    assert rows["qwen3-design-1.7b.p1"]["in_use_for"] == "design"
    assert rows["qwen3-base-1.7b.p1"]["canary"]["threshold"] is not None


def test_pin_refuses_while_a_daemon_holds_the_store_s4(config: Config) -> None:
    code, admin = _run(config, "engine", "pin", held=False)
    assert code == EXIT_FAILED
    assert "daemon is running" in admin.stderr and "next: Stop it (narration-admin daemon stop" in admin.stderr


def test_pin_refuses_without_the_vram_a_qwen_load_needs_s4(config: Config) -> None:
    code, admin = _run(config, "engine", "pin", env=_env(probe=_Probe(free_mb=2000)))
    assert code == EXIT_FAILED
    assert "2000 MB free" in admin.stderr and "next:" in admin.stderr


def test_a_changed_installation_is_refused_and_repin_makes_the_new_profile_s10_1(
    config: Config, install: Install
) -> None:
    _run(config, "engine", "pin")
    (QWEN_BASE.snapshot_dir(install.models_root) / "model.safetensors").write_bytes(b"new weights")

    code, admin = _run(config, "engine", "pin")
    assert code == EXIT_FAILED
    assert "differs from the pinned engine profile qwen3-base-1.7b.p1 in: weights" in admin.stderr
    assert "next: To make a new profile the one in use, run narration-admin engine repin" in admin.stderr

    code, admin = _run(config, "engine", "repin")
    assert code == EXIT_OK
    out = admin.stdout
    assert "qwen3-base-1.7b.p2  new" in out and "changed: weights" in out and "measured again" in out

    code, admin = _run(config, "engine", "bridge", "qwen3-base-1.7b.p1", "qwen3-base-1.7b.p2")
    assert code == EXIT_OK
    assert "qwen3-base-1.7b.p1 -> qwen3-base-1.7b.p2" in admin.stdout
    assert "cannot render here: weights" in admin.stdout


def test_bridge_of_an_unknown_profile_says_how_to_list_them_s10_1(config: Config) -> None:
    _run(config, "engine", "pin")
    code, admin = _run(config, "engine", "bridge", "qwen3-base-1.7b.p1", "qwen3-base-1.7b.p9")
    assert code == EXIT_FAILED and "engine show" in admin.stderr


def test_a_worker_that_is_not_installed_says_to_run_doctor(config: Config) -> None:
    env = Environment(
        starter=SubprocessStarter,  # the real qwen3 and QA workers, whose venvs this installation lacks
        probe=lambda config: NoProbe(),
        packages=FAKE_WORKER_PACKAGES,
    )
    code, admin = _run(config, "engine", "pin", env=env)
    assert code == EXIT_FAILED and "narration-admin doctor" in admin.stderr
