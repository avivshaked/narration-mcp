"""The ``narration-mcp`` console script (design sections 5, 16): it finds its configuration by the service's
one rule (``--config``, else ``NARRATION_CONFIG``, else ``narration.toml`` in the service's folder), and
without a usable one exits at once, saying what to do. No server is started here."""

from __future__ import annotations

from pathlib import Path

import pytest

from narration.config import Config
from narration.mcp import __main__ as console


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[Config]:
    """Stand in for ``serve``: record the configuration it would serve."""
    seen: list[Config] = []

    def fake_serve(config: Config, *, log_level: str = "INFO") -> int:
        seen.append(config)
        return 0

    monkeypatch.setattr(console, "serve", fake_serve)
    monkeypatch.delenv("NARRATION_CONFIG", raising=False)
    return seen


def write_config(path: Path, body: str | None = None) -> Path:
    store = (path.parent / "store").as_posix()
    models = (path.parent / "models").as_posix()
    text = body if body is not None else f"[server]\nstore_root = '{store}'\nmodels_root = '{models}'\n"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_missing_config_exits_with_what_to_do_s16(
    tmp_path: Path, served: list[Config], capsys: pytest.CaptureFixture[str]
) -> None:
    code = console.main(["--config", str(tmp_path / "absent.toml")])
    assert code == console.EXIT_USAGE
    err = capsys.readouterr().err
    assert "narration.example.toml" in err and "--config" in err
    assert served == []


def test_a_config_that_does_not_load_exits_naming_the_problem_s16(
    tmp_path: Path, served: list[Config], capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_config(tmp_path / "narration.toml", "[server]\nstore_root = 3\n")
    assert console.main(["--config", str(path)]) == console.EXIT_USAGE
    assert "narration-mcp:" in capsys.readouterr().err
    assert served == []


def test_the_config_named_by_config_is_served_s16(tmp_path: Path, served: list[Config]) -> None:
    path = write_config(tmp_path / "narration.toml")
    assert console.main(["--config", str(path)]) == 0
    (config,) = served
    assert config.path is not None and config.path.resolve() == path.resolve()
    assert config.server.store_root == tmp_path / "store"


def test_the_config_named_by_the_environment_is_served_s16(
    tmp_path: Path, served: list[Config], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_config(tmp_path / "service.toml")
    monkeypatch.setenv("NARRATION_CONFIG", str(path))
    assert console.main([]) == 0
    assert served[0].path is not None and served[0].path.resolve() == path.resolve()


def test_an_explicit_config_wins_over_the_environment_s16(
    tmp_path: Path, served: list[Config], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NARRATION_CONFIG", str(tmp_path / "absent.toml"))
    path = write_config(tmp_path / "narration.toml")
    assert console.main(["--config", str(path)]) == 0
    assert len(served) == 1
