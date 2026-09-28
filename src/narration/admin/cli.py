"""What every ``narration-admin`` command group is built on (design section 7.1; plan.md WP37).

**A command group** is a module with one function::

    def register(subparsers: Subparsers) -> None

It adds the group's parser, named as ``narration.admin.__main__.COMMAND_GROUPS`` names the group
(``subparsers.add_parser("engine", help=...)``). On every command it can run, it sets
``set_defaults(handler=<function>)``. A handler is::

    def handler(admin: Admin, args: argparse.Namespace) -> int

It gets the ``Admin`` context and the parsed arguments, and returns the exit code. The context holds the
configuration (loaded on first use), the platform, the store (opened on first use) and the output streams.

**Errors.** A problem the operator can fix is raised as ``AdminError``, with a message that says what to do
next. ``main`` prints it to stderr and exits with the code it carries (``EXIT_FAILED`` by default). A
``NarrationError`` is printed with its code and hint.

**Exit codes:** ``EXIT_OK`` 0; ``EXIT_FAILED`` 1 (the command ran and could not do it, or found a problem);
``EXIT_USAGE`` 2 (bad arguments, or no configuration that loads); ``EXIT_UNAVAILABLE`` 3 (the command is not
in this build).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final, Protocol, TextIO

from narration.config import Config, find_config, load_config
from narration.contracts.errors import ConfigError
from narration.platform import ProcessPlatform, get_platform
from narration.store import NarrationStore
from narration.store.layout import DB_NAME

PROGRAM: Final = "narration-admin"
EXIT_OK: Final = 0
EXIT_FAILED: Final = 1
EXIT_USAGE: Final = 2
EXIT_UNAVAILABLE: Final = 3


class Subparsers(Protocol):
    """The object ``register`` gets: what ``ArgumentParser.add_subparsers`` returns."""

    def add_parser(self, name: str, **kwargs: Any) -> argparse.ArgumentParser:
        """Add the parser of the command ``name``."""
        ...


class AdminError(Exception):
    """A problem the operator can fix. ``message`` says what went wrong and what to do next."""

    def __init__(self, message: str, *, exit_code: int = EXIT_FAILED) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code


class Admin:
    """What a command's handler gets.

    ``config_path`` is the ``--config`` the operator gave, or None; the file used is ``find_config``'s, the
    one rule ``narration-mcp`` uses too (``--config``, else ``NARRATION_CONFIG`` in ``environ``, else
    ``narration.toml`` in the service's folder). The configuration is loaded, and the store opened, on first
    use, so a command that needs neither (``doctor`` reporting a missing file) still runs. ``platform`` makes
    the ``Platform`` (tests pass a stand-in). ``inp`` is where ``ask`` reads the operator's answer (stdin by
    default). Call ``close`` when done.
    """

    def __init__(
        self,
        *,
        config_path: Path | None,
        out: TextIO | None = None,
        err: TextIO | None = None,
        inp: TextIO | None = None,
        platform: Callable[[], ProcessPlatform] = get_platform,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.given_config = config_path
        self.environ = environ
        self.out: TextIO = sys.stdout if out is None else out
        self.err: TextIO = sys.stderr if err is None else err
        self.inp: TextIO | None = sys.stdin if inp is None else inp
        self._make_platform = platform
        self._platform: ProcessPlatform | None = None
        self._config: Config | None = None
        self._store: NarrationStore | None = None

    def config_path(self) -> Path:
        """The configuration file this command uses (``find_config``); ``AdminError`` (``EXIT_USAGE``) saying
        what to do when there is none."""
        try:
            return find_config(self.given_config, environ=self.environ)
        except ConfigError as exc:
            raise AdminError(str(exc), exit_code=EXIT_USAGE) from exc

    def config(self) -> Config:
        """The configuration; ``AdminError`` (``EXIT_USAGE``) when there is none or it does not load."""
        if self._config is None:
            path = self.config_path()
            try:
                self._config = load_config(path)
            except ConfigError as exc:
                raise AdminError(
                    f"{exc}. Fix it (narration.example.toml documents every key), then run the command again.",
                    exit_code=EXIT_USAGE,
                ) from exc
        return self._config

    def platform(self) -> ProcessPlatform:
        """This OS's ``Platform`` (made once)."""
        if self._platform is None:
            self._platform = self._make_platform()
        return self._platform

    def store_exists(self) -> bool:
        """Whether the configured store has a database yet (a daemon or a command has used it)."""
        return (self.config().server.store_root / DB_NAME).is_file()

    def store(self) -> NarrationStore:
        """The store at ``[server] store_root``, opened once (and created if it is not there yet)."""
        if self._store is None:
            self._store = NarrationStore.from_config(self.config(), self.platform())
        return self._store

    def say(self, text: str = "") -> None:
        """Print a line of the command's output."""
        print(text, file=self.out, flush=True)

    def warn(self, text: str) -> None:
        """Print a line to stderr."""
        print(text, file=self.err, flush=True)

    def ask(self, question: str) -> str | None:
        """Print ``question`` and read one line of the operator's answer, without its line ending. None when there
        is no answer to read: stdin is closed, missing (``pythonw``) or at its end (a pipe with nothing more, or
        Ctrl+Z / Ctrl+D)."""
        print(question, end="", file=self.out, flush=True)
        line = ""
        if self.inp is not None:
            try:
                line = self.inp.readline()
            except (OSError, ValueError):  # ValueError: the stream is closed
                line = ""
        if not line:
            self.say()  # end the question's line
            return None
        return line.rstrip("\r\n")

    def close(self) -> None:
        """Close the store, if it was opened."""
        if self._store is not None:
            self._store.close()
            self._store = None


def names_the_module(exc: ModuleNotFoundError, module: str) -> bool:
    """Whether ``exc`` says ``module`` itself (or a package it is in) is not installed, as opposed to a
    dependency that module or package imports: the first means "not in this build", the second a broken
    build."""
    missing = exc.name
    return missing is not None and (module == missing or module.startswith(missing + "."))


Handler = Callable[[Admin, argparse.Namespace], int]
"""A command's handler (see the module docstring)."""


__all__ = [
    "EXIT_FAILED",
    "EXIT_OK",
    "EXIT_UNAVAILABLE",
    "EXIT_USAGE",
    "PROGRAM",
    "Admin",
    "AdminError",
    "Handler",
    "Subparsers",
    "names_the_module",
]
