"""Refuse local paths and binary data in what git tracks (AGENTS.md, hard rules 2 and 4).

Checks every tracked file by default, or only the staged files (``--staged``, for the pre-commit hook),
or a commit message (``--msg-file``, for the commit-msg hook), or the messages of a range of commits
(``--commits A..B``, for CI and before a push).

Two kinds of finding:

* **binary data**: a file with an audio, weight, checkpoint or database extension, a file over the size
  limit, or a file that is not text;
* **local paths**: text that looks like a path on someone's machine, such as a Windows user profile,
  ``AppData``, a Unix home directory, or a drive-letter ``Projects`` folder. A developer can add
  machine-specific patterns (their user name, their drive layout) to the gitignored file
  ``.dev/local-path-patterns.txt``, one regular expression per line.

Standard library only, Python 3.10 or later, so it runs in hooks and CI without a virtual environment.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

FORBIDDEN_SUFFIXES = frozenset(
    {
        ".wav",
        ".flac",
        ".mp3",
        ".ogg",
        ".m4a",
        ".aac",
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".ckpt",
        ".onnx",
        ".gguf",
        ".npz",
        ".npy",
        ".sqlite",
        ".sqlite3",
        ".db",
    }
)
MAX_BYTES = 5 * 1024 * 1024
BINARY_ALLOWED_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg"})

GENERIC_PATTERNS: tuple[tuple[str, str], ...] = (
    ("Windows user profile", r"\b[A-Za-z]:[\\/]+(?:Users|Documents and Settings)[\\/]+[^\\/\s\"'`]+"),
    ("AppData folder", r"[\\/]AppData[\\/]"),
    ("macOS home directory", r"(?<![\w.])/Users/[^/\s\"'`]+/"),
    ("Unix home directory", r"(?<![\w.])/home/[^/\s\"'`]+/"),
    ("drive-letter Projects folder", r"\b[A-Za-z]:[\\/]+Projects[\\/]"),
)
LOCAL_PATTERNS_FILE = Path(".dev") / "local-path-patterns.txt"
SELF = Path("tools") / "check_tracked.py"


@dataclass(frozen=True)
class Finding:
    where: str
    what: str

    def __str__(self) -> str:
        return f"{self.where}: {self.what}"


def git(*args: str, root: Path) -> bytes:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout


def repo_root() -> Path:
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True)
    return Path(out.stdout.strip())


def load_patterns(root: Path) -> list[tuple[str, re.Pattern[str]]]:
    patterns = [(name, re.compile(rx)) for name, rx in GENERIC_PATTERNS]
    local = root / LOCAL_PATTERNS_FILE
    if local.is_file():
        for n, line in enumerate(local.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append((f"local pattern {LOCAL_PATTERNS_FILE.as_posix()}:{n}", re.compile(line)))
    return patterns


def scan_text(where: str, text: str, patterns: list[tuple[str, re.Pattern[str]]]) -> list[Finding]:
    findings = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for name, rx in patterns:
            if rx.search(line):
                findings.append(Finding(f"{where}:{lineno}", f"looks like a local path ({name})"))
    return findings


def scan_file(rel: str, data: bytes, patterns: list[tuple[str, re.Pattern[str]]]) -> list[Finding]:
    path = Path(rel)
    suffix = path.suffix.lower()
    if suffix in FORBIDDEN_SUFFIXES:
        return [Finding(rel, f"'{suffix}' files are never committed (audio, weights, databases)")]
    if len(data) > MAX_BYTES:
        return [Finding(rel, f"larger than {MAX_BYTES // (1024 * 1024)} MB")]
    if suffix in BINARY_ALLOWED_SUFFIXES and suffix != ".svg":
        return []
    if b"\0" in data[:8192]:
        return [Finding(rel, "not a text file")]
    if path == SELF:
        return []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return [Finding(rel, "not valid UTF-8")]
    return scan_text(rel, text, patterns)


def tracked_files(root: Path, staged: bool) -> list[tuple[str, bytes]]:
    if staged:
        names = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z", root=root)
        return [(n, git("show", f":{n}", root=root)) for n in names.decode("utf-8").split("\0") if n]
    names = git("ls-files", "-z", root=root).decode("utf-8").split("\0")
    return [(n, (root / n).read_bytes()) for n in names if n and (root / n).is_file()]


def commit_messages(root: Path, rev_range: str) -> list[tuple[str, str]]:
    out = git("log", "--format=%H%x00%B%x01", rev_range, root=root).decode("utf-8")
    messages = []
    for record in out.split("\x01"):
        record = record.strip("\n")
        if record:
            sha, _, body = record.partition("\0")
            messages.append((f"commit {sha[:12]}", body))
    return messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--staged", action="store_true", help="check only staged files (pre-commit)")
    group.add_argument("--msg-file", type=Path, help="check one commit message file (commit-msg)")
    group.add_argument(
        "--commits", metavar="RANGE", help="check the messages of a commit range, e.g. origin/main..HEAD"
    )
    args = parser.parse_args(argv)

    root = repo_root()
    patterns = load_patterns(root)
    findings: list[Finding] = []
    if args.msg_file:
        findings += scan_text("commit message", args.msg_file.read_text(encoding="utf-8"), patterns)
    elif args.commits:
        for where, body in commit_messages(root, args.commits):
            findings += scan_text(where, body, patterns)
    else:
        for rel, data in tracked_files(root, staged=args.staged):
            findings += scan_file(rel, data, patterns)

    for finding in findings:
        print(finding, file=sys.stderr)
    if findings:
        print(
            f"\n{len(findings)} problem(s). Local paths belong in AGENTS.local.md; audio and weights never go in git.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
