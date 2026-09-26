"""Refuse private text in what git tracks: passages, names or numbers copied from a private corpus.

The repository is public, but a developer may work beside private material: text that is theirs or someone
else's and must never be published (here, the private bake-off's scripts and transcripts). This check reads
that material at check time, compares it with what git would publish, and prints findings only to the
developer's own console. Nothing private is ever written into the repository.

Configure it locally, in gitignored files:

* ``.dev/private-text.txt``: the private folders, one per line (or ``NARRATION_PRIVATE_TEXT``: folders
  separated by ``os.pathsep``). Every ``.txt``, ``.json``, ``.jsonl``, ``.md``, ``.csv``, ``.srt`` and ``.vtt``
  file under them is read.
* ``.dev/private-terms.txt`` (optional): distinctive words of the private text, such as invented names, one
  per line.
* ``.dev/private-allow.txt`` (optional): passages that may appear anyway, one per line, as five words.

Three kinds of finding, each reported only when it is **new**, that is absent from the base tree
(``--base``, by default ``origin/main``). Text the project already publishes, such as its design, is not
reported again.

* **passage**: five consecutive words shared with the private text;
* **term**: a word listed in ``private-terms.txt``;
* **number**: a number written with thousands separators (``12,345``) that the private text contains.

Modes: the whole tracked tree (default), ``--staged`` (pre-commit), ``--msg-file`` (commit-msg), or
``--commits A..B`` (pre-push: the files those commits change, and their messages).

With no private folder configured it prints nothing and exits 0, so CI and other machines are unaffected.
Standard library only, Python 3.10 or later.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

N = 5
"""Passage length in words. Four-word runs matched ordinary phrases ("at the end or") too often."""

CORPUS_SUFFIXES = frozenset({".txt", ".json", ".jsonl", ".md", ".csv", ".srt", ".vtt"})
CHECKED_SUFFIXES = frozenset(
    {".py", ".md", ".txt", ".json", ".jsonl", ".toml", ".csv", ".yml", ".yaml", ".cfg", ".ini", ".sh", ".ps1", ""}
)
MAX_FILE_BYTES = 20 << 20
_WORD = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")
_NUMBER = re.compile(r"(?<![\d,])\d{1,3}(?:,\d{3})+(?![\d,])")


@dataclass(frozen=True)
class Finding:
    where: str
    kind: str
    text: str

    def __str__(self) -> str:
        return f"{self.where}: {self.kind}: {self.text!r}"


def words(text: str) -> list[str]:
    """Lower-case words and digit runs; a curly apostrophe is folded to a straight one."""
    return [w.replace("’", "'") for w in _WORD.findall(text.lower())]


def passages(tokens: list[str]) -> Iterator[str]:
    """Every run of ``N`` words that holds at least three words of three letters or more, so that runs of
    numbers and JSON keys (``start 1 2 end 1``) are not passages."""
    for i in range(len(tokens) - N + 1):
        run = tokens[i : i + N]
        if sum(1 for w in run if len(w) >= 3 and w.isalpha()) >= 3:
            yield " ".join(run)


@dataclass
class Private:
    passages: set[str]
    terms: set[str]
    numbers: set[str]


def git(*args: str, root: Path) -> bytes:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout


def repo_root() -> Path:
    return Path(git("rev-parse", "--show-toplevel", root=Path.cwd()).decode("utf-8").strip())


def _lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def config_dir(root: Path) -> Path:
    """The main checkout's ``.dev`` folder, so every worktree of the repository shares one configuration."""
    common = git("rev-parse", "--path-format=absolute", "--git-common-dir", root=root).decode("utf-8").strip()
    return Path(common).parent / ".dev"


def private_folders(conf: Path) -> list[Path]:
    """The configured private folders: ``NARRATION_PRIVATE_TEXT``, else ``.dev/private-text.txt``."""
    env = os.environ.get("NARRATION_PRIVATE_TEXT", "")
    names = [p for p in env.split(os.pathsep) if p.strip()] or _lines(conf / "private-text.txt")
    return [Path(n.strip()) for n in names]


def _corpus_files(folders: Iterable[Path]) -> Iterator[Path]:
    for folder in folders:
        if not folder.is_dir():
            raise SystemExit(f"check_private: the private folder {folder} does not exist; fix .dev/private-text.txt")
        for path in folder.rglob("*"):
            if path.suffix.lower() in CORPUS_SUFFIXES and path.is_file() and path.stat().st_size <= MAX_FILE_BYTES:
                yield path


def load_private(conf: Path, folders: list[Path]) -> Private:
    grams: set[str] = set()
    numbers: set[str] = set()
    for path in _corpus_files(folders):
        text = path.read_text(encoding="utf-8", errors="replace")
        grams.update(passages(words(text)))
        numbers.update(_NUMBER.findall(text))
    grams.difference_update(" ".join(words(line)) for line in _lines(conf / "private-allow.txt"))
    terms = {t.lower().replace("’", "'") for t in _lines(conf / "private-terms.txt")}
    return Private(passages=grams, terms=terms, numbers=numbers)


@dataclass
class Baseline:
    passages: set[str]
    words: set[str]
    numbers: set[str]


def _is_text(name: str) -> bool:
    return Path(name).suffix.lower() in CHECKED_SUFFIXES


def baseline(root: Path, base: str | None) -> Baseline:
    """What the base tree already publishes. Empty when there is no base."""
    b = Baseline(set(), set(), set())
    if base is None:
        return b
    for text in _blob_texts(root, base):
        tokens = words(text)
        b.passages.update(passages(tokens))
        b.words.update(tokens)
        b.numbers.update(_NUMBER.findall(text))
    return b


def _blob_texts(root: Path, ref: str) -> Iterator[str]:
    """The text of every checked file in ``ref``'s tree, read with one ``git cat-file --batch``."""
    shas = []
    for entry in git("ls-tree", "-r", "-z", ref, root=root).decode("utf-8").split("\0"):
        meta, _, name = entry.partition("\t")
        parts = meta.split()
        if name and len(parts) == 3 and parts[1] == "blob" and _is_text(name):
            shas.append(parts[2])
    if not shas:
        return
    batch = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=root,
        input=("\n".join(shas) + "\n").encode(),
        capture_output=True,
        check=True,
    ).stdout
    pos = 0
    for _ in shas:
        header_end = batch.index(b"\n", pos)
        size = int(batch[pos:header_end].split()[2])
        yield batch[header_end + 1 : header_end + 1 + size].decode("utf-8", errors="replace")
        pos = header_end + 1 + size + 1


def scan_text(where: str, text: str, private: Private, base: Baseline) -> list[Finding]:
    """Findings in one text, each reported once per text."""
    findings: list[Finding] = []
    tokens = words(text)
    seen: set[str] = set()
    for gram in passages(tokens):
        if gram in private.passages and gram not in base.passages and gram not in seen:
            seen.add(gram)
            findings.append(Finding(where, "passage", gram))
    for term in sorted(private.terms & set(tokens) - base.words):
        findings.append(Finding(where, "term", term))
    for number in sorted(set(_NUMBER.findall(text)) & private.numbers - base.numbers):
        findings.append(Finding(where, "number", number))
    return findings


def _default_base(root: Path) -> str | None:
    for ref in ("origin/main", "main"):
        if subprocess.run(["git", "rev-parse", "--verify", "-q", ref], cwd=root, capture_output=True).returncode == 0:
            return ref
    return None


def commit_texts(root: Path, rev_range: str) -> list[tuple[str, str]]:
    """Each commit's message, and the lines each commit adds to each checked file.

    Every commit is checked, not only the range's final tree: a push publishes every commit, so text added
    by one commit and removed by a later one is still published.
    """
    out: list[tuple[str, str]] = []
    shas = git("rev-list", "--reverse", rev_range, root=root).decode("utf-8").split()
    for sha in shas:
        body = git("log", "-1", "--format=%B", sha, root=root).decode("utf-8", "replace")
        out.append((f"commit {sha[:12]} message", body))
        patch = git("show", "--format=", "--unified=0", "--no-color", "--no-renames", sha, root=root)
        name: str | None = None
        added: dict[str, list[str]] = {}
        for raw in patch.decode("utf-8", "replace").splitlines():
            if raw.startswith("+++ "):
                path = raw[4:]
                name = path[2:] if path.startswith("b/") else None
            elif raw.startswith("+") and name is not None and _is_text(name):
                added.setdefault(name, []).append(raw[1:])
        out += [(f"{name} (added in {sha[:12]})", "\n".join(lines)) for name, lines in added.items()]
    return out


def texts_to_check(root: Path, args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.msg_file:
        return [("commit message", args.msg_file.read_text(encoding="utf-8"))]
    if args.commits:
        return commit_texts(root, args.commits)
    if args.staged:
        names = git("diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR", root=root).decode("utf-8")
        return [
            (n, git("show", f":{n}", root=root).decode("utf-8", "replace"))
            for n in names.split("\0")
            if n and _is_text(n)
        ]
    names = git("ls-files", "-z", root=root).decode("utf-8").split("\0")
    return [
        (n, (root / n).read_text(encoding="utf-8", errors="replace"))
        for n in names
        if n and _is_text(n) and (root / n).is_file()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--staged", action="store_true", help="check only staged files (pre-commit)")
    group.add_argument("--msg-file", type=Path, help="check one commit message file (commit-msg)")
    group.add_argument("--commits", metavar="RANGE", help="check the files and messages of A..B (pre-push)")
    parser.add_argument("--base", help="the published tree to compare with (default: origin/main, else main)")
    args = parser.parse_args(argv)

    root = repo_root()
    conf = config_dir(root)
    folders = private_folders(conf)
    if not folders:
        return 0
    private = load_private(conf, folders)
    base = baseline(root, args.base or _default_base(root))
    findings: list[Finding] = []
    for where, text in texts_to_check(root, args):
        findings += scan_text(where, text, private, base)

    for finding in findings:
        print(finding, file=sys.stderr)
    if findings:
        print(
            f"\n{len(findings)} finding(s) of private text. Remove it, and rewrite any unpushed commit that carries"
            " it; a later commit does not unpublish an earlier one. A false positive can be listed in"
            " .dev/private-allow.txt.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
