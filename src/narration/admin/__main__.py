"""``narration-admin``: the operator CLI, for the machine and not for any use of it (design section 7.1).

::

    narration-admin [--config <path>] [-v] <command> ...

The configuration is found as ``narration-mcp`` finds it (``narration.config.find_config``): the file
``--config`` names, else the one ``NARRATION_CONFIG`` names, else ``narration.toml`` in the service's folder.

The commands are grouped, and each group is a module listed in ``COMMAND_GROUPS``. Each module exposes
``register(subparsers)`` (``narration.admin.cli``). A group whose module is not in this build still appears
in ``--help``, and running it says it is not available rather than failing. ``engine`` is WP32's module,
and ``bench`` waits for WP38. A module that is there but fails to import says why, and the other groups
still work.

None of the commands approves anything; the service has no approvals (section 17.10).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TextIO, cast

import narration
from narration.contracts.errors import NarrationError
from narration.platform import ProcessPlatform, get_platform

from .cli import (
    EXIT_FAILED,
    EXIT_UNAVAILABLE,
    EXIT_USAGE,
    PROGRAM,
    Admin,
    AdminError,
    Handler,
    Subparsers,
    names_the_module,
)

log = logging.getLogger("narration.admin")


@dataclass(frozen=True, slots=True)
class CommandGroup:
    """One group of commands: its name on the command line, the module that registers it, and its help."""

    name: str
    module: str
    help: str


COMMAND_GROUPS: Final[tuple[CommandGroup, ...]] = (
    CommandGroup(
        "install",
        "narration.admin.install",
        "download the pinned models (verified) and sync the worker venvs (section 17.8)",
    ),
    CommandGroup("engine", "narration.engine.admin", "pin, re-pin or bridge the engine profile (section 10.1)"),
    CommandGroup(
        "gc", "narration.admin.gc", "remove what retention no longer keeps; a dry run by default (section 15)"
    ),
    CommandGroup("verify", "narration.admin.verify", "re-hash the store's immutable files (section 15)"),
    CommandGroup("bench", "narration.bench.admin", "measure the cue aligner on the service's benchmark (section 11.2)"),
    CommandGroup("daemon", "narration.admin.daemon", "start, stop or show the daemon (section 4.1)"),
    CommandGroup("doctor", "narration.admin.doctor", "check this machine and the configuration, and say what to fix"),
    CommandGroup("render", "narration.admin.render", "render one text in a voice, from the terminal"),
)
"""The command groups, in the order ``--help`` lists them (section 7.1's order)."""

Register = Callable[[Subparsers], None]


@dataclass(frozen=True, slots=True)
class Loaded:
    """``load_group``'s answer: the group's ``register``, or why it is not available and the exit code its
    stand-in returns (``EXIT_UNAVAILABLE`` when the module is not in this build, ``EXIT_FAILED`` when it is
    there but broken)."""

    register: Register | None
    why: str = ""
    exit_code: int = EXIT_FAILED


def load_group(group: CommandGroup) -> Loaded:
    """Import the group's module. A module (or a package it is in) that is not installed is "not in this
    build"; one that is there but fails to import, for example on a missing dependency, is broken, and the
    message names what is missing."""
    absent = Loaded(None, f"it is not in this build ({group.module} is not installed)", EXIT_UNAVAILABLE)
    try:
        spec = importlib.util.find_spec(group.module)
    except ModuleNotFoundError as exc:  # importing a parent package failed on a missing module
        if names_the_module(exc, group.module):
            return absent
        return Loaded(None, f"it could not be loaded (a module it needs is missing: {exc.name})")
    except Exception as exc:  # a parent package failed to import
        return Loaded(None, f"it could not be loaded ({type(exc).__name__}: {exc})")
    if spec is None:
        return absent
    try:
        module = importlib.import_module(group.module)
    except ModuleNotFoundError as exc:
        log.debug("could not import %s", group.module, exc_info=True)
        if names_the_module(exc, group.module):
            return absent
        return Loaded(None, f"it could not be loaded (a module it needs is missing: {exc.name})")
    except Exception as exc:
        log.debug("could not import %s", group.module, exc_info=True)
        return Loaded(None, f"it could not be loaded ({type(exc).__name__}: {exc})")
    register: object = getattr(module, "register", None)
    if not callable(register):
        return Loaded(None, f"{group.module} has no register(subparsers)")
    return Loaded(cast("Register", register))


def _unavailable(group: CommandGroup, why: str, exit_code: int) -> Handler:
    def handler(admin: Admin, args: argparse.Namespace) -> int:
        admin.warn(f"{PROGRAM} {group.name} is not available: {why}.")
        return exit_code

    return handler


def _add_unavailable(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],  # pyright: ignore[reportPrivateUsage]
    group: CommandGroup,
    why: str,
    exit_code: int = EXIT_FAILED,
) -> None:
    """Add the group's stand-in, first dropping whatever parser a failed ``register`` left under its name."""
    subparsers.choices.pop(group.name, None)
    listed: list[Any] = getattr(subparsers, "_choices_actions", [])
    listed[:] = [action for action in listed if getattr(action, "dest", None) != group.name]
    parser = subparsers.add_parser(group.name, help=f"{group.help} [not available: {why}]", add_help=False)
    parser.add_argument("rest", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    parser.set_defaults(handler=_unavailable(group, why, exit_code))


def build_parser(groups: Sequence[CommandGroup] = COMMAND_GROUPS) -> argparse.ArgumentParser:
    """The whole command line: the global options and every group (a stand-in for each one that is not
    available)."""
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Operator commands for narration-mcp: for the machine, not for any use of it (section 7.1).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="the configuration file (default: $NARRATION_CONFIG, else narration.toml in the service's folder)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log what the commands do (to stderr)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {narration.__version__}")
    subparsers = parser.add_subparsers(title="commands", metavar="<command>")
    for group in groups:
        loaded = load_group(group)
        register = loaded.register
        if register is None:
            _add_unavailable(subparsers, group, loaded.why, loaded.exit_code)
            continue
        try:
            register(subparsers)
        except Exception as exc:
            log.debug("%s.register failed", group.module, exc_info=True)
            _add_unavailable(subparsers, group, f"its commands could not be set up ({type(exc).__name__}: {exc})")
            continue
        if group.name not in subparsers.choices:
            _add_unavailable(subparsers, group, f"{group.module}.register added no {group.name!r} command")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    out: TextIO | None = None,
    err: TextIO | None = None,
    platform: Callable[[], ProcessPlatform] = get_platform,
    groups: Sequence[CommandGroup] = COMMAND_GROUPS,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Run one operator command and return its exit code (``narration.admin.cli``). ``out``, ``err``,
    ``platform``, ``groups`` and ``environ`` (for ``NARRATION_CONFIG``) are for tests."""
    parser = build_parser(groups)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse's usage errors, --help and --version
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    handler: Handler | None = getattr(args, "handler", None)
    if handler is None:
        parser.print_help(file=sys.stderr if err is None else err)
        return EXIT_USAGE
    _log_to_stderr(verbose=bool(args.verbose), err=err)
    admin = Admin(config_path=args.config, out=out, err=err, platform=platform, environ=environ)
    try:
        return handler(admin, args)
    except AdminError as exc:
        admin.warn(f"{PROGRAM}: {exc.message}")
        return exc.exit_code
    except NarrationError as exc:
        admin.warn(f"{PROGRAM}: {exc.message} ({exc.code}). {exc.hint}")
        return EXIT_FAILED
    except KeyboardInterrupt:
        admin.warn(f"{PROGRAM}: interrupted.")
        return 130
    finally:
        admin.close()


def _log_to_stderr(*, verbose: bool, err: TextIO | None) -> None:
    """Warnings (and with ``-v``, what the commands do) go to stderr; stdout is the commands' output. Does
    nothing when logging is already set up (``basicConfig``'s rule), as under a test runner."""
    logging.basicConfig(
        stream=sys.stderr if err is None else err,
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


if __name__ == "__main__":
    sys.exit(main())
