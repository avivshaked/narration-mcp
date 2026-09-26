"""The private-text check refuses passages, names and numbers copied from a private corpus (AGENTS.md hard
rule 1). Every text here is invented for the test; the corpus is written to a temporary folder."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "check_private.py"

PRIVATE = (
    "At the far end of the valley the quiet lanterns of Orravel burned through the winter, "
    "and by spring 4,812 of them had gone dark.\n"
    "Everyone agreed that the long road home was worth it.\n"
)
PUBLISHED = "Everyone agreed that the long road home was worth it.\n"


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_private", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_private"] = module
    spec.loader.exec_module(module)
    return module


check = load_tool()


def git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.org",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.org",
    }
    return subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository whose ``main`` publishes one sentence of the private text, and a private folder."""
    private = tmp_path / "private"
    private.mkdir()
    (private / "script.txt").write_text(PRIVATE, encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    (repo / "design.md").write_text("# Design\n\n" + PUBLISHED, encoding="utf-8")
    git(repo, "add", "design.md")
    git(repo, "commit", "-q", "-m", "base")
    (repo / ".dev").mkdir()
    (repo / ".dev" / "private-text.txt").write_text(f"# the private corpus\n{private}\n", encoding="utf-8")
    (repo / ".dev" / "private-terms.txt").write_text("Orravel\n", encoding="utf-8")
    return repo


def run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "NARRATION_PRIVATE_TEXT"}
    return subprocess.run(
        [sys.executable, str(TOOL), *args], cwd=repo, capture_output=True, text=True, check=False, env=env
    )


def test_passages_need_three_real_words_so_numbers_and_keys_are_not_passages() -> None:
    assert list(check.passages(["start", "1", "2", "end", "1"])) == []
    assert list(check.passages(["the", "quiet", "lanterns", "of", "orravel"])) == ["the quiet lanterns of orravel"]


def test_words_fold_the_curly_apostrophe() -> None:
    assert check.words("Orravel’s LANTERNS") == ["orravel's", "lanterns"]


def test_with_no_private_folder_configured_it_does_nothing(repo: Path) -> None:
    (repo / ".dev" / "private-text.txt").unlink()
    (repo / "notes.md").write_text("the quiet lanterns of Orravel burned through the winter\n", encoding="utf-8")
    git(repo, "add", "notes.md")
    result = run(repo, "--staged")
    assert result.returncode == 0 and result.stderr == ""


def test_a_copied_passage_is_refused(repo: Path) -> None:
    (repo / "test_x.py").write_text(
        'TEXT = "the lanterns burned through the winter, and by spring"\n', encoding="utf-8"
    )
    git(repo, "add", "test_x.py")
    result = run(repo, "--staged")
    assert result.returncode == 1
    assert "passage" in result.stderr and "burned through the winter and" in result.stderr


def test_a_passage_the_base_already_publishes_is_not_reported_again(repo: Path) -> None:
    (repo / "readme.md").write_text("As the design says: " + PUBLISHED, encoding="utf-8")
    git(repo, "add", "readme.md")
    assert run(repo, "--staged").returncode == 0


def test_a_private_term_is_refused_even_alone(repo: Path) -> None:
    (repo / "test_y.py").write_text('HINT = "Orravel"\n', encoding="utf-8")
    git(repo, "add", "test_y.py")
    result = run(repo, "--staged")
    assert result.returncode == 1 and "term: 'orravel'" in result.stderr


def test_a_distinctive_number_of_the_private_text_is_refused(repo: Path) -> None:
    (repo / "plan.md").write_text("A paragraph ending in 4,812 moved the cue.\n", encoding="utf-8")
    git(repo, "add", "plan.md")
    result = run(repo, "--staged")
    assert result.returncode == 1 and "number: '4,812'" in result.stderr


def test_a_commit_message_is_checked(repo: Path, tmp_path: Path) -> None:
    msg = tmp_path / "msg.txt"
    msg.write_text("fix: the quiet lanterns of orravel\n", encoding="utf-8")
    result = run(repo, "--msg-file", str(msg))
    assert result.returncode == 1


def test_every_commit_of_a_push_is_checked_not_only_the_last(repo: Path) -> None:
    # Text added by one commit and removed by the next is still published by the push.
    git(repo, "checkout", "-q", "-b", "wp")
    (repo / "a.md").write_text("by spring many of them had gone dark\n", encoding="utf-8")
    git(repo, "add", "a.md")
    git(repo, "commit", "-q", "-m", "add")
    (repo / "a.md").write_text("nothing here\n", encoding="utf-8")
    git(repo, "commit", "-q", "-am", "remove")
    result = run(repo, "--commits", "main..wp", "--base", "main")
    assert result.returncode == 1 and "added in" in result.stderr
    assert "spring many of them had" not in (repo / "a.md").read_text(encoding="utf-8")


def test_a_clean_push_passes(repo: Path) -> None:
    git(repo, "checkout", "-q", "-b", "clean")
    (repo / "b.md").write_text("An invented sentence about a harbour and its tides.\n", encoding="utf-8")
    git(repo, "add", "b.md")
    git(repo, "commit", "-q", "-m", "docs: an invented sentence")
    assert run(repo, "--commits", "main..clean", "--base", "main").returncode == 0


def test_an_allowed_passage_is_not_reported(repo: Path) -> None:
    (repo / ".dev" / "private-allow.txt").write_text("burned through the winter and\n", encoding="utf-8")
    (repo / "c.md").write_text("it burned through the winter, and that was all\n", encoding="utf-8")
    git(repo, "add", "c.md")
    assert run(repo, "--staged").returncode == 0


def test_a_missing_private_folder_is_an_error_not_a_silent_pass(repo: Path, tmp_path: Path) -> None:
    (repo / ".dev" / "private-text.txt").write_text(str(tmp_path / "gone") + "\n", encoding="utf-8")
    result = run(repo, "--staged")
    assert result.returncode != 0 and "does not exist" in result.stderr
