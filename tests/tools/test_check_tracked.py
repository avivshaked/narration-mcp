"""The tracked-files check refuses local paths and binary data (AGENTS.md hard rules 2 and 4; plan.md WP00)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


check = load_tool("check_tracked")
PATTERNS = [(name, __import__("re").compile(rx)) for name, rx in check.GENERIC_PATTERNS]


def test_the_tracked_tree_passes() -> None:
    assert check.main([]) == 0


@pytest.mark.parametrize(
    "text",
    [
        "store_root = 'C:" + "\\Users\\someone\\narration'",
        "cache at /home/" + "someone/.cache/x",
        "see /Users/" + "someone/Documents/x",
        "D:" + "\\Projects\\thing",
        "C:" + "\\Users\\x\\App" + "Data\\Local\\Temp",
    ],
)
def test_a_local_path_is_refused(text: str) -> None:
    assert check.scan_text("f.py", text, PATTERNS)


@pytest.mark.parametrize("text", ["<store_root>\\takes\\8c", "https://example.org/home/page", "~/.cache is fine"])
def test_placeholders_and_urls_pass(text: str) -> None:
    assert not check.scan_text("f.md", text, PATTERNS)


@pytest.mark.parametrize("name", ["take.wav", "model.safetensors", "w.bin", "db.sqlite", "a.npy"])
def test_binary_data_is_refused_by_extension(name: str) -> None:
    assert check.scan_file(name, b"RIFF", PATTERNS)


def test_a_non_text_file_is_refused() -> None:
    assert check.scan_file("data.dat", b"\x00\x01\x02", PATTERNS)


def test_a_planted_wav_in_the_index_is_refused(tmp_path: Path) -> None:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "ok.txt").write_text("fine\n", encoding="utf-8")
    (repo / "take.wav").write_bytes(b"RIFF....WAVE")
    subprocess.run(["git", "-C", str(repo), "add", "-f", "ok.txt", "take.wav"], check=True)
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_tracked.py"), "--staged"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "take.wav" in result.stderr
