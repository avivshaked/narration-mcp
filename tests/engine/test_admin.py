"""``narration-admin engine`` as the operator runs it (design sections 7.1 and 10.1; plan.md WP32, WP37): through
``narration-admin``'s own dispatcher (``narration.admin.__main__.main``), with the configuration found by its
one rule, its exit codes, what the commands print and what they say to do next; and ``doctor`` reading the
engine's pins and the pinned profiles.

The workers are the fake role, started through the daemon's supervisor as ``engine pin`` starts the real ones
(``ENVIRONMENT`` is replaced for these tests); the platform is the stand-in.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import narration.config
from narration.admin.__main__ import main
from narration.admin.cli import EXIT_FAILED, EXIT_OK, EXIT_USAGE, Admin
from narration.admin.doctor import Probes, SyncCheck, diagnose, engine_module_present
from narration.admin.models import pinned_revisions
from narration.contracts.errors import WorkerCrashed
from narration.engine import admin as engine_admin
from narration.engine.admin import Environment
from narration.engine.models import PINNED, QWEN_BASE
from narration.engine.pinning import SupervisedStarter
from narration.jobs.gpu import NoProbe, VramProbe, VramReading
from narration.platform.testing import StandInPlatform
from narration.store.layout import DB_NAME

from .support import FAKE_WORKER_PACKAGES, CountingStarter, Install, fake_install


@dataclass(frozen=True)
class Ran:
    code: int
    out: str
    err: str


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


@dataclass
class Cli:
    """``narration-admin`` in this process, on a miniature installation and the stand-in platform."""

    install: Install
    config_path: Path
    platform: StandInPlatform
    monkeypatch: pytest.MonkeyPatch

    def __call__(self, *argv: str, env: Environment | None = None, config: bool = True) -> Ran:
        self.monkeypatch.setattr(engine_admin, "ENVIRONMENT", env if env is not None else _env())
        out, err = io.StringIO(), io.StringIO()
        args = ["--config", str(self.config_path), *argv] if config else list(argv)
        code = main(args, out=out, err=err, platform=lambda: self.platform, environ={})
        return Ran(code, out.getvalue(), err.getvalue())

    @property
    def store_root(self) -> Path:
        return self.install.store_root


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Cli]:
    monkeypatch.setattr(narration.config, "service_root", lambda: None)  # never the checkout's own config
    install = fake_install(tmp_path)
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
    yield Cli(install=install, config_path=path, platform=StandInPlatform(), monkeypatch=monkeypatch)


def test_engine_pin_prints_what_it_pinned_s10_1(cli: Cli) -> None:
    ran = cli("engine", "pin")
    assert ran.code == EXIT_OK, ran.err
    assert "qwen3-design-1.7b.p1  new  sha256:" in ran.out and "qwen3-base-1.7b.p1  new  sha256:" in ran.out
    assert "tier bit_exact" in ran.out and "canary threshold" in ran.out

    ran = cli("engine", "pin", "--json")
    assert ran.code == EXIT_OK
    assert [e["action"] for e in json.loads(ran.out)["engines"]] == ["keep", "keep"]


def test_engine_show_lists_the_profiles_in_use_s10_1(cli: Cli) -> None:
    ran = cli("engine", "show")
    assert ran.code == EXIT_OK and "engine pin" in ran.out  # nothing pinned yet: it says what to run
    assert not (cli.store_root / DB_NAME).exists()  # and showing created no store

    cli("engine", "pin")
    ran = cli("engine", "show", "--json")
    assert ran.code == EXIT_OK
    rows = {r["engine_profile_id"]: r for r in json.loads(ran.out)["profiles"]}
    assert rows["qwen3-base-1.7b.p1"]["in_use_for"] == "base"
    assert rows["qwen3-design-1.7b.p1"]["in_use_for"] == "design"
    assert rows["qwen3-base-1.7b.p1"]["canary"]["threshold"] is not None


def test_the_configuration_is_the_dispatchers_one_rule_s16(cli: Cli) -> None:
    ran = cli("engine", "pin", config=False)  # no --config, no NARRATION_CONFIG, no service folder file
    assert ran.code == EXIT_USAGE and "narration.toml" in ran.err


def test_pin_refuses_while_a_daemon_holds_the_store_s4(cli: Cli) -> None:
    with cli.platform.hold(cli.store_root):
        ran = cli("engine", "pin")
    assert ran.code == EXIT_FAILED
    assert "daemon is running" in ran.err and "`narration-admin daemon stop`" in ran.err


def test_pin_refuses_without_the_vram_a_qwen_load_needs_s4(cli: Cli) -> None:
    ran = cli("engine", "pin", env=_env(probe=_Probe(free_mb=2000)))
    assert ran.code == EXIT_FAILED and "2000 MB free" in ran.err and "Wait until the GPU has room" in ran.err


def test_a_changed_installation_is_refused_and_repin_makes_the_new_profile_s10_1(cli: Cli) -> None:
    cli("engine", "pin")
    (QWEN_BASE.snapshot_dir(cli.install.models_root) / "model.safetensors").write_bytes(b"new weights")

    ran = cli("engine", "pin")
    assert ran.code == EXIT_FAILED
    assert "differs from the pinned engine profile qwen3-base-1.7b.p1 in: weights" in ran.err
    assert "run narration-admin engine repin" in ran.err

    ran = cli("engine", "repin")
    assert ran.code == EXIT_OK
    assert "qwen3-base-1.7b.p2  new" in ran.out and "changed: weights" in ran.out and "measured again" in ran.out

    ran = cli("engine", "bridge", "qwen3-base-1.7b.p1", "qwen3-base-1.7b.p2")
    assert ran.code == EXIT_OK
    assert "qwen3-base-1.7b.p1 -> qwen3-base-1.7b.p2" in ran.out and "cannot render here: weights" in ran.out


def test_a_repin_that_keeps_both_says_how_to_force_one_s10_1(cli: Cli) -> None:
    """A gate failing for a reason repin cannot see leaves the operator at "keep": the output names the way out."""
    cli("engine", "pin")
    ran = cli("engine", "repin")
    assert ran.code == EXIT_OK, ran.err
    assert "qwen3-design-1.7b.p1  keep" in ran.out and "qwen3-base-1.7b.p1  keep" in ran.out
    assert "Nothing changed in the installation or on this machine" in ran.out
    assert "engine repin --force" in ran.out
    assert "Nothing changed" not in cli("engine", "pin").out  # only a repin says it


def test_a_worker_that_stops_during_the_pin_pins_nothing_s10_1(cli: Cli, monkeypatch: pytest.MonkeyPatch) -> None:
    """The message says nothing was pinned, and it is so: VoiceDesign's canary measured, Base's did not."""
    from narration.engine import pinning

    real = pinning.embed

    def embed(qa: Any, wav: Path) -> tuple[float, ...]:
        if Path(wav).name.startswith("base-calibration"):
            raise WorkerCrashed("the qa worker exited", exit_code=1)
        return real(qa, wav)

    monkeypatch.setattr(pinning, "embed", embed)
    ran = cli("engine", "pin")
    assert ran.code == EXIT_FAILED
    assert "a worker stopped" in ran.err and "nothing was pinned" in ran.err
    shown = cli("engine", "show", "--json")
    assert shown.code == EXIT_OK and json.loads(shown.out)["profiles"] == []


def test_repin_force_repins_both_engines_s10_1(cli: Cli) -> None:
    cli("engine", "pin")
    ran = cli("engine", "repin", "--force")
    assert ran.code == EXIT_OK, ran.err
    assert "qwen3-design-1.7b.p2  new" in ran.out and "qwen3-base-1.7b.p2  new" in ran.out
    assert "changed: forced" in ran.out and "measured again" in ran.out


def test_engine_show_prints_each_gates_tier_and_threshold_s10_1(cli: Cli) -> None:
    pinned = cli("engine", "pin")
    assert "calibration " in pinned.out  # the similarities, as the pin measured them
    ran = cli("engine", "show")
    assert ran.code == EXIT_OK
    lines = {line.split()[0]: line for line in ran.out.splitlines() if line.startswith("qwen3-")}
    assert "tier bit_exact  canary threshold 0." in lines["qwen3-base-1.7b.p1"]
    # The fake designs an unrelated voice for another seed: VoiceDesign's threshold is the floor, and says so.
    assert "canary threshold 0.1000 (the floor)" in lines["qwen3-design-1.7b.p1"]
    assert "VoiceDesign's gate is a drift alarm" in ran.out and "engine pin --json" in ran.out


def test_bridge_of_an_unknown_profile_says_how_to_list_them_s10_1(cli: Cli) -> None:
    cli("engine", "pin")
    ran = cli("engine", "bridge", "qwen3-base-1.7b.p1", "qwen3-base-1.7b.p9")
    assert ran.code == EXIT_FAILED and "engine show" in ran.err


def test_a_worker_that_is_not_installed_says_to_run_doctor(cli: Cli) -> None:
    env = Environment(
        starter=SupervisedStarter,  # the real qwen3 and QA workers, whose venvs this installation lacks
        probe=lambda config: NoProbe(),
        packages=FAKE_WORKER_PACKAGES,
    )
    ran = cli("engine", "pin", env=env)
    assert ran.code == EXIT_FAILED and "narration-admin doctor" in ran.err


# ======================================================================== doctor reads the engine's pins


def test_doctor_checks_the_models_the_engine_pins_s17_8() -> None:
    pins = pinned_revisions()
    assert pins == {repo: model.revision for repo, model in PINNED.items()}
    assert engine_module_present()


def test_doctor_reports_each_pinned_profile_with_its_tier_and_canary_s10_1(cli: Cli) -> None:
    cli("engine", "pin")
    admin = Admin(config_path=cli.config_path, platform=lambda: cli.platform, environ={})
    try:
        probes = Probes(
            venv_synced=lambda project: SyncCheck(True, "in sync"),
            gpu=lambda index: pytest.fail("the GPU is not asked for here"),
            engine_module=engine_module_present,
        )
        findings = [f for f in diagnose(admin, probes=probes, hash_models=False) if f.area == "engine"]
    finally:
        admin.close()
    assert [(f.level, f.summary) for f in findings] == [
        ("ok", "base: qwen3-base-1.7b.p1, tier bit_exact, canary pinned"),
        ("ok", "design: qwen3-design-1.7b.p1, tier bit_exact, canary pinned"),
    ]
