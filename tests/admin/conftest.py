"""Fixtures for the operator CLI's tests: a configuration under pytest's tmp_path, the platform stand-in, and
``admin``, which runs ``narration-admin`` in this process and captures what it prints.

The checkout's own ``narration.toml`` (if the owner has one) and a ``NARRATION_CONFIG`` in this process's
environment are never used: ``service_root`` is replaced for every test here, and ``admin`` passes its own
environment.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

import narration.config
from narration.admin.__main__ import COMMAND_GROUPS, CommandGroup, main
from narration.platform.testing import StandInPlatform


@dataclass(frozen=True)
class Ran:
    """What one ``narration-admin`` run returned and printed."""

    code: int
    out: str
    err: str


AdminRun = Callable[..., Ran]


def write_config(folder: Path, extra: str = "") -> Path:
    """``<folder>/narration.toml`` with the store and models under ``folder``, plus ``extra`` TOML."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "narration.toml"
    path.write_text(f'[server]\nstore_root = "store"\nmodels_root = "models"\n{extra}', encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def no_checkout_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(narration.config, "service_root", lambda: None)


@pytest.fixture
def platform() -> StandInPlatform:
    return StandInPlatform()


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    return write_config(tmp_path / "service")


@pytest.fixture
def admin(platform: StandInPlatform) -> AdminRun:
    def run(
        *argv: str, groups: Sequence[CommandGroup] = COMMAND_GROUPS, environ: Mapping[str, str] | None = None
    ) -> Ran:
        out, err = io.StringIO(), io.StringIO()
        code = main(list(argv), out=out, err=err, platform=lambda: platform, groups=groups, environ=environ or {})
        return Ran(code, out.getvalue(), err.getvalue())

    return run
