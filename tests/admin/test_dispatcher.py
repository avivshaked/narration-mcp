"""``narration-admin``'s dispatcher: the configuration it uses, and command groups registered by module, with
a group that is not in this build saying so instead of breaking the others (design sections 7.1 and 16)."""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import pytest

import narration.config
from narration.admin.__main__ import COMMAND_GROUPS, CommandGroup, build_parser
from narration.admin.cli import EXIT_FAILED, EXIT_OK, EXIT_UNAVAILABLE, EXIT_USAGE

from .conftest import AdminRun, write_config

_packages = itertools.count()

GOOD = """
def register(subparsers):
    parser = subparsers.add_parser("hello", help="say hello")
    parser.add_argument("--name", default="world")

    def hello(admin, args):
        admin.say(f"hello {args.name}")
        return 0

    parser.set_defaults(handler=hello)
"""

SHOW_CONFIG = """
def register(subparsers):
    parser = subparsers.add_parser("show", help="show the configuration it got")
    parser.set_defaults(handler=lambda admin, args: admin.say(str(admin.config().path)) or 0)
"""

FAILING = """
from narration.admin.cli import AdminError
from narration.contracts.errors import NarrationError


def register(subparsers):
    parser = subparsers.add_parser("fail", help="fail")
    parser.add_argument("how", choices=["admin", "narration"])

    def fail(admin, args):
        if args.how == "admin":
            raise AdminError("the widget is missing. Put it back, then run this again.")
        raise NarrationError("DAEMON_UNAVAILABLE", "the daemon cannot start")

    parser.set_defaults(handler=fail)
"""


@pytest.fixture
def package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh importable package for the test's own command modules (a new name per test)."""
    name = f"admin_groups_{next(_packages)}"
    folder = tmp_path / "importable" / name
    folder.mkdir(parents=True)
    (folder / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(folder.parent))
    return folder


def module(package: Path, name: str, source: str) -> str:
    (package / f"{name}.py").write_text(source, encoding="utf-8")
    return f"{package.name}.{name}"


def test_every_operator_command_of_section_7_1_is_listed_s7_1() -> None:
    listed = build_parser().format_help()
    for name in ("install", "engine", "gc", "verify", "bench", "daemon", "doctor", "render"):
        assert name in listed
    assert [g.name for g in COMMAND_GROUPS] == [
        "install",
        "engine",
        "gc",
        "verify",
        "bench",
        "daemon",
        "doctor",
        "render",
    ]


def test_a_group_runs_through_the_module_that_registered_it_s7_1(admin: AdminRun, package: Path) -> None:
    groups = [CommandGroup("hello", module(package, "good", GOOD), "say hello")]
    ran = admin("hello", "--name", "operator", groups=groups)
    assert (ran.code, ran.out) == (EXIT_OK, "hello operator\n")


def test_a_group_not_in_this_build_says_so_and_the_others_still_run_s7_1(admin: AdminRun, package: Path) -> None:
    groups = [
        CommandGroup("ghost", f"{package.name}.not_written", "a group whose module is not in this build"),
        CommandGroup("hello", module(package, "good", GOOD), "say hello"),
    ]
    ran = admin("ghost", "pin", "--anything", groups=groups)
    assert ran.code == EXIT_UNAVAILABLE
    assert "narration-admin ghost is not available" in ran.err and "not in this build" in ran.err
    assert "[not available" in build_parser(groups).format_help()
    assert admin("hello", groups=groups).code == EXIT_OK


def test_a_group_whose_parent_package_is_missing_is_not_available_s7_1(admin: AdminRun) -> None:
    groups = [CommandGroup("ghost", "narration_no_such_package.admin", "missing")]
    ran = admin("ghost", groups=groups)
    assert ran.code == EXIT_UNAVAILABLE and "not in this build" in ran.err


def test_a_group_that_fails_to_import_says_why_and_the_others_still_run_s7_1(admin: AdminRun, package: Path) -> None:
    groups = [
        CommandGroup("broken", module(package, "broken", "raise RuntimeError('a dependency is missing')\n"), "b"),
        CommandGroup("hello", module(package, "good", GOOD), "say hello"),
    ]
    ran = admin("broken", groups=groups)
    assert ran.code == EXIT_FAILED
    assert "could not be loaded (RuntimeError: a dependency is missing)" in ran.err
    assert admin("hello", groups=groups).code == EXIT_OK


@pytest.mark.parametrize(
    ("source", "why"),
    [
        ("X = 1\n", "has no register(subparsers)"),
        ("def register(subparsers):\n    pass\n", "register added no 'quiet' command"),
        ("def register(subparsers):\n    raise ValueError('bad')\n", "could not be set up (ValueError: bad)"),
    ],
)
def test_a_module_that_does_not_register_its_group_is_reported_s7_1(
    admin: AdminRun, package: Path, source: str, why: str
) -> None:
    groups = [CommandGroup("quiet", module(package, "quiet", source), "q")]
    ran = admin("quiet", groups=groups)
    assert ran.code == EXIT_FAILED and why in ran.err


@pytest.mark.parametrize(
    ("group", "module_name"), [("engine", "narration.engine.admin"), ("bench", "narration.bench.admin")]
)
def test_the_engine_and_bench_groups_wait_for_their_packages_s7_1(
    admin: AdminRun, group: str, module_name: str
) -> None:
    try:
        present = importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        present = False
    if present:
        pytest.skip(f"{module_name} is in this build")
    ran = admin(group, "pin")
    assert ran.code == EXIT_UNAVAILABLE and f"{module_name} is not installed" in ran.err


def test_config_is_the_file_config_names_s16(admin: AdminRun, package: Path, config_path: Path) -> None:
    groups = [CommandGroup("show", module(package, "show", SHOW_CONFIG), "s")]
    ran = admin("--config", str(config_path), "show", groups=groups)
    assert (ran.code, ran.out.strip()) == (EXIT_OK, str(config_path.resolve()))


def test_without_config_the_service_folders_file_is_used_s16(
    admin: AdminRun, package: Path, config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(narration.config, "service_root", lambda: config_path.parent)
    groups = [CommandGroup("show", module(package, "show", SHOW_CONFIG), "s")]
    ran = admin("show", groups=groups)
    assert (ran.code, ran.out.strip()) == (EXIT_OK, str(config_path.resolve()))


def test_without_config_narration_config_names_the_file_s16(admin: AdminRun, package: Path, config_path: Path) -> None:
    groups = [CommandGroup("show", module(package, "show", SHOW_CONFIG), "s")]
    ran = admin("show", groups=groups, environ={"NARRATION_CONFIG": str(config_path)})
    assert (ran.code, ran.out.strip()) == (EXIT_OK, str(config_path.resolve()))


def test_with_no_configuration_a_command_that_needs_one_says_what_to_do_s16(admin: AdminRun, package: Path) -> None:
    groups = [CommandGroup("show", module(package, "show", SHOW_CONFIG), "s")]
    ran = admin("show", groups=groups)
    assert ran.code == EXIT_USAGE
    assert "narration-admin: no configuration given: pass --config" in ran.err and "NARRATION_CONFIG" in ran.err


def test_a_configuration_that_does_not_load_names_the_key_and_the_template_s16(
    admin: AdminRun, package: Path, tmp_path: Path
) -> None:
    path = write_config(tmp_path / "bad", "[daemon]\nidle_exit_minutes = 5\n")
    groups = [CommandGroup("show", module(package, "show", SHOW_CONFIG), "s")]
    ran = admin("--config", str(path), "show", groups=groups)
    assert ran.code == EXIT_USAGE
    assert "idle_exit_minutes" in ran.err and "narration.example.toml" in ran.err


def test_a_missing_config_file_given_by_name_is_refused_s16(admin: AdminRun, package: Path, tmp_path: Path) -> None:
    groups = [CommandGroup("show", module(package, "show", SHOW_CONFIG), "s")]
    ran = admin("--config", str(tmp_path / "nowhere.toml"), "show", groups=groups)
    assert ran.code == EXIT_USAGE and "no configuration file at" in ran.err


def test_errors_from_a_command_are_printed_with_what_to_do_next_s14(admin: AdminRun, package: Path) -> None:
    groups = [CommandGroup("fail", module(package, "failing", FAILING), "f")]
    ran = admin("fail", "admin", groups=groups)
    assert ran.code == EXIT_FAILED
    assert ran.err.strip() == "narration-admin: the widget is missing. Put it back, then run this again."
    ran = admin("fail", "narration", groups=groups)
    assert ran.code == EXIT_FAILED
    assert "the daemon cannot start (DAEMON_UNAVAILABLE)" in ran.err


def test_no_command_prints_the_help_s7_1(admin: AdminRun) -> None:
    ran = admin()
    assert ran.code == EXIT_USAGE and "<command>" in ran.err


def test_bad_arguments_are_a_usage_error_s7_1(admin: AdminRun, capsys: pytest.CaptureFixture[str]) -> None:
    assert admin("no-such-command").code == EXIT_USAGE
    assert "invalid choice" in capsys.readouterr().err

