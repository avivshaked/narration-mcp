"""``[voices] allow_sha256`` in the text of a configuration file: read it, and add a hash to it (DC-17).

``narration-admin voices allow`` changes the owner's own ``narration.toml``, so the edit keeps everything else
in the file as it was: every other line, every comment, the blank lines and the line endings. It is a text
edit, not a load and re-dump of the TOML (which would drop the comments):

- ``_Scanner`` walks the file as TOML does: table headers, keys (bare, quoted, dotted) and values (strings of
  all four kinds, arrays, inline tables and the other scalars), with the comments and newlines between them.
  It finds where ``[voices]`` is, and where ``allow_sha256``'s array opens, holds each item and closes. A ``[``
  or a ``#`` inside a string is therefore never taken for a header or a comment.
- ``add_hash`` inserts one line, ``"<sha256>",  # <note>``, as the array's last item, and adds a comma after
  the item before it when that item has none. A one-line array (``[]``, ``["..."]``) is opened onto lines, so
  that the note can follow the new item. A missing ``allow_sha256`` goes into the ``[voices]`` table, after
  the comment lines directly below its header; a missing ``[voices]`` table goes at the end of the file. New
  lines end as the line where they go in does (CRLF or LF), so a file with mixed endings keeps each part's.
- The edited text is then parsed again with ``tomllib``, and must equal the original with the one hash
  appended and nothing else changed. A file the scanner cannot follow, or an edit that fails that check,
  raises ``AllowlistEditError``, so a wrong edit is never returned, let alone written.

``allow_sha256`` written in a form this edit does not follow (inside an inline table, ``voices = {...}``) is
refused with ``AllowlistEditError`` too; the command then prints what to add by hand (``hand_edit``: the bare
value for an array written on one line, where a comment would swallow the rest of the line).
"""

from __future__ import annotations

import copy
import json
import re
import tomllib
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, cast

TABLE: Final = "voices"
KEY: Final = "allow_sha256"
PATH: Final = (TABLE, KEY)
ITEM_INDENT: Final = "  "
"""The indent of a new item, when the array has no item on a line of its own to copy the indent from."""

_SHA256: Final = re.compile(r"[0-9a-f]{64}")
_BARE_KEY: Final = re.compile(r"[A-Za-z0-9_-]+")
_DATE: Final = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_TIME_AFTER_SPACE: Final = re.compile(r" [0-9]{2}:")
_SCALAR_ENDS: Final = frozenset(" \t\r\n,]}#")

StatementKind = Literal["table", "array_table", "keyval"]
ValueKind = Literal["array", "inline_table", "other"]


class AllowlistEditError(ValueError):
    """The hash could not be added safely; the message says why. Nothing was changed."""


@dataclass(frozen=True, slots=True)
class Entry:
    """One allowed clip: its sha256, and the comment on its line (the note ``add_hash`` writes), if any."""

    sha256: str
    note: str | None


@dataclass(frozen=True, slots=True)
class _Array:
    open: int
    close: int
    items: tuple[tuple[int, int], ...]
    """Where each item's value starts and ends."""
    trailing_comma: bool
    """Whether a comma follows the last item."""


@dataclass(frozen=True, slots=True)
class _Statement:
    kind: StatementKind
    path: tuple[str, ...]
    """A header: the table's name. A key/value: the name of the table it is in, then the key's parts."""
    start: int
    line_end: int
    """Just past the newline that ends the statement's last line (or the end of the text)."""
    value: ValueKind | None
    array: _Array | None


class _Scanner:
    """Walks valid TOML (the caller has parsed it with ``tomllib`` first) and records its statements."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    # ---------------------------------------------------------------- statements
    def statements(self) -> list[_Statement]:
        found: list[_Statement] = []
        table: tuple[str, ...] = ()
        text = self.text
        while True:
            self._skip_blank()
            if self.pos >= len(text):
                return found
            start = self.pos
            kind: StatementKind
            value: ValueKind | None = None
            array: _Array | None = None
            if text.startswith("[[", start):
                self.pos += 2
                table = self._header_name("]]")
                kind, path = "array_table", table
            elif text.startswith("[", start):
                self.pos += 1
                table = self._header_name("]")
                kind, path = "table", table
            else:
                key = self._key()
                self._spaces()
                self._expect("=")
                self._spaces()
                value, array = self._value()
                kind, path = "keyval", table + key
            self._end_of_line()
            found.append(_Statement(kind, path, start, self.pos, value, array))

    def _header_name(self, close: str) -> tuple[str, ...]:
        self._spaces()
        name = self._key()
        self._spaces()
        self._expect(close)
        return name

    # ---------------------------------------------------------------- keys
    def _key(self) -> tuple[str, ...]:
        parts = [self._simple_key()]
        while True:
            self._spaces()
            if not self.text.startswith(".", self.pos):
                return tuple(parts)
            self.pos += 1
            self._spaces()
            parts.append(self._simple_key())

    def _simple_key(self) -> str:
        start = self.pos
        if self.text.startswith('"', start):
            self._basic_string()
            return _decode_key(self.text[start : self.pos])
        if self.text.startswith("'", start):
            self._literal_string()
            return self.text[start + 1 : self.pos - 1]
        match = _BARE_KEY.match(self.text, start)
        if match is None:
            raise self._error("a key")
        self.pos = match.end()
        return match.group()

    # ---------------------------------------------------------------- values
    def _value(self) -> tuple[ValueKind, _Array | None]:
        text, pos = self.text, self.pos
        if text.startswith('"""', pos):
            self._multiline('"""', escapes=True)
        elif text.startswith("'''", pos):
            self._multiline("'''", escapes=False)
        elif text.startswith('"', pos):
            self._basic_string()
        elif text.startswith("'", pos):
            self._literal_string()
        elif text.startswith("[", pos):
            return "array", self._array()
        elif text.startswith("{", pos):
            self._inline_table()
            return "inline_table", None
        else:
            self._scalar()
        return "other", None

    def _array(self) -> _Array:
        opened = self.pos
        self.pos += 1
        items: list[tuple[int, int]] = []
        trailing_comma = False
        while True:
            self._skip_blank()
            if self.text.startswith("]", self.pos):
                break
            start = self.pos
            self._value()
            items.append((start, self.pos))
            trailing_comma = False
            self._skip_blank()
            if self.text.startswith(",", self.pos):
                self.pos += 1
                trailing_comma = True
                continue
            if not self.text.startswith("]", self.pos):
                raise self._error("',' or ']' in an array")
            break
        closed = self.pos
        self.pos += 1
        return _Array(opened, closed, tuple(items), trailing_comma)

    def _inline_table(self) -> None:
        self.pos += 1
        self._spaces()
        if self.text.startswith("}", self.pos):
            self.pos += 1
            return
        while True:
            self._spaces()
            self._key()
            self._spaces()
            self._expect("=")
            self._spaces()
            self._value()
            self._spaces()
            if self.text.startswith(",", self.pos):
                self.pos += 1
                continue
            self._expect("}")
            return

    def _basic_string(self) -> None:
        text, pos = self.text, self.pos + 1
        while pos < len(text):
            char = text[pos]
            if char == "\\":
                pos += 2
            elif char == '"':
                self.pos = pos + 1
                return
            elif char in "\r\n":
                break
            else:
                pos += 1
        raise self._error("the end of a string")

    def _literal_string(self) -> None:
        end = self.text.find("'", self.pos + 1)
        if end == -1 or any(c in self.text[self.pos : end] for c in "\r\n"):
            raise self._error("the end of a string")
        self.pos = end + 1

    def _multiline(self, quotes: str, *, escapes: bool) -> None:
        text, pos = self.text, self.pos + 3
        mark = quotes[0]
        while pos < len(text):
            if escapes and text[pos] == "\\":
                pos += 2
                continue
            if text.startswith(quotes, pos):
                run = 3
                while run < 5 and text.startswith(mark, pos + run):  # one or two quotes may end the content
                    run += 1
                self.pos = pos + run
                return
            pos += 1
        raise self._error("the end of a multi-line string")

    def _scalar(self) -> None:
        text, start = self.text, self.pos
        pos = start
        while pos < len(text) and text[pos] not in _SCALAR_ENDS:
            pos += 1
        if _DATE.fullmatch(text, start, pos) and _TIME_AFTER_SPACE.match(text, pos):
            pos += 1  # a date and a time separated by a space
            while pos < len(text) and text[pos] not in _SCALAR_ENDS:
                pos += 1
        if pos == start:
            raise self._error("a value")
        self.pos = pos

    # ---------------------------------------------------------------- whitespace, comments, lines
    def _spaces(self) -> None:
        text, pos = self.text, self.pos
        while pos < len(text) and text[pos] in " \t":
            pos += 1
        self.pos = pos

    def _comment(self) -> None:
        if self.text.startswith("#", self.pos):
            end = self.text.find("\n", self.pos)
            self.pos = len(self.text) if end == -1 else end
            if self.text[self.pos - 1] == "\r":  # a CRLF's CR belongs to the newline, not to the comment
                self.pos -= 1

    def _newline(self) -> bool:
        for newline in ("\r\n", "\n"):
            if self.text.startswith(newline, self.pos):
                self.pos += len(newline)
                return True
        return False

    def _skip_blank(self) -> None:
        """Spaces, comments and newlines: what may come between two statements, or two items of an array."""
        while True:
            self._spaces()
            self._comment()
            if not self._newline():
                return

    def _end_of_line(self) -> None:
        self._spaces()
        self._comment()
        if self.pos < len(self.text) and not self._newline():
            raise self._error("the end of the line")

    def _expect(self, token: str) -> None:
        if not self.text.startswith(token, self.pos):
            raise self._error(repr(token))
        self.pos += len(token)

    def _error(self, wanted: str) -> AllowlistEditError:
        line = self.text.count("\n", 0, self.pos) + 1
        return AllowlistEditError(f"could not follow the file's TOML at line {line} (expected {wanted})")


def _decode_key(quoted: str) -> str:
    """A basic-string key's value, escapes and all (``tomllib`` decodes it)."""
    return next(iter(tomllib.loads(f"{quoted} = 0")))


# ==================================================================== reading
def entries(text: str) -> tuple[Entry, ...]:
    """The allowed hashes in a configuration file's text, in order, each with the comment on its line.

    The hashes are ``tomllib``'s. The comments are the scanner's, and are None where there is none (or where the
    scanner cannot follow the file). Raises ``AllowlistEditError`` when the text is not TOML, or
    ``allow_sha256`` is not a list of strings.
    """
    listed = _listed(_parse(text))
    notes: list[str | None] = [None] * len(listed)
    try:
        found = _find(_Scanner(text).statements(), PATH)
    except AllowlistEditError:
        found = None
    if found is not None and found.array is not None and len(found.array.items) == len(listed):
        notes = [_note_after(text, end) for _, end in found.array.items]
    return tuple(Entry(sha, note) for sha, note in zip(listed, notes, strict=True))


def _note_after(text: str, end: int) -> str | None:
    """The comment that follows an array item on its line (past its comma), or None."""
    pos = _past_spaces(text, end)
    if text.startswith(",", pos):
        pos = _past_spaces(text, pos + 1)
    if not text.startswith("#", pos):
        return None
    stop = text.find("\n", pos)
    note = text[pos + 1 : len(text) if stop == -1 else stop].strip()
    return note or None


def _past_spaces(text: str, pos: int) -> int:
    while pos < len(text) and text[pos] in " \t":
        pos += 1
    return pos


# ==================================================================== adding
def add_hash(text: str, sha256: str, note: str | None) -> str | None:
    """``text`` with ``sha256`` added to ``[voices] allow_sha256``, with ``note`` as the comment on its line; or
    None when the hash is listed already (compared without regard to case). The hash is written in lower case.

    Every other line, comment and line ending is kept (the module docstring). Raises ``AllowlistEditError``
    when ``sha256`` is not a sha256, the text is not TOML, ``allow_sha256`` is written in a form this edit does
    not follow, or the edited text would not parse to the original plus this one hash.
    """
    sha = sha256.strip().lower()
    if not _SHA256.fullmatch(sha):
        raise AllowlistEditError(f"{sha256!r} is not a sha256 (64 hex characters)")
    data = _parse(text)
    if any(listed.lower() == sha for listed in _listed(data)):
        return None
    statements = _Scanner(text).statements()
    item = f'"{sha}",' + (f"  # {comment_text(note)}" if note else "")
    found = _find(statements, PATH)
    if found is not None:
        if found.array is None:
            raise AllowlistEditError(f"{KEY} is not written as an array")
        edited = _append_item(text, found, found.array, item)
    else:
        header = _find(statements, (TABLE,), kind="table")
        if any(s.path[:1] == (TABLE,) and s is not header for s in statements):
            raise AllowlistEditError(f"[{TABLE}] is written in a form this command does not edit")
        if header is not None:
            edited = _add_key(text, statements, header, item)
        elif TABLE in data:
            raise AllowlistEditError(f"could not find where [{TABLE}] is defined")
        else:
            edited = _add_table(text, item)
    expected = copy.deepcopy(data)
    expected.setdefault(TABLE, {}).setdefault(KEY, []).append(sha)
    try:
        after = tomllib.loads(edited)
    except tomllib.TOMLDecodeError as exc:
        raise AllowlistEditError(f"the edit would not be valid TOML ({exc})") from exc
    if _canonical(after) != _canonical(expected):
        raise AllowlistEditError("the edit would change more than the one hash")
    return edited


@dataclass(frozen=True, slots=True)
class HandEdit:
    """What to add to the file by hand to allow a hash, when the command cannot (``hand_edit``)."""

    kind: Literal["line", "value", "key"]
    text: str


def hand_edit(text: str | None, sha256: str, note: str | None) -> HandEdit:
    """What an operator adds by hand to allow ``sha256``, fitted to how ``text`` writes ``allow_sha256``:

    - ``line``: a multi-line array. The text is a line of its own, ``"<sha256>",  # <note>``.
    - ``value``: ``allow_sha256`` fits on one line (a one-line array, or an inline ``voices = {...}`` table),
      or the file cannot be followed or read (``text`` None). The text is the bare ``"<sha256>"``, with no
      comment, which would swallow the rest of the line (an inline table must stay on one line).
    - ``key``: there is no ``allow_sha256``. The text is ``allow_sha256 = ["<sha256>"]``, for ``[voices]``.
    """
    value = f'"{sha256.strip().lower()}"'
    if text is None:
        return HandEdit("value", value)
    try:
        data = _parse(text)
        _listed(data)
    except AllowlistEditError:
        return HandEdit("value", value)
    table = data.get(TABLE)
    if not isinstance(table, dict) or KEY not in table:
        return HandEdit("key", f"{KEY} = [{value}]")
    try:
        found = _find(_Scanner(text).statements(), PATH)
    except AllowlistEditError:
        found = None
    if found is not None and found.array is not None and "\n" in text[found.array.open : found.array.close]:
        return HandEdit("line", f"{value},  # {comment_text(note)}" if note else f"{value},")
    return HandEdit("value", value)


def _append_item(text: str, statement: _Statement, array: _Array, item: str) -> str:
    """Add ``item`` as the array's last item, on a line of its own that ends as the ``]``'s line does."""
    close = array.close
    newline = _ending_at(text, close)
    close_line = text.rfind("\n", 0, close) + 1
    key_indent = _indent_before(text, statement.start)
    indent = _item_indent(text, array.items)
    if indent is None:
        indent = key_indent + ITEM_INDENT
    if close_line > array.open and text[close_line:close].strip(" \t") == "":
        at = cut = close_line  # ']' begins its line: the item goes on a line just above it
        insertion = f"{indent}{item}{newline}"
    else:  # ']' follows other text on its line: a one-line array, or the last item is on the bracket's line
        at = close
        while at > array.open + 1 and text[at - 1] in " \t":
            at -= 1
        cut, insertion = close, f"{newline}{indent}{item}{newline}{key_indent}"
    if array.items and not array.trailing_comma:
        comma = array.items[-1][1]
        return text[:comma] + "," + text[comma:at] + insertion + text[cut:]
    return text[:at] + insertion + text[cut:]


def _add_key(text: str, statements: Sequence[_Statement], header: _Statement, item: str) -> str:
    """Add ``allow_sha256 = [...]`` to the ``[voices]`` table: after its last key, or after the comment lines
    directly below its header (they describe the table). The new lines end as the header's line does."""
    newline = _ending_at(text, header.start)
    body: list[_Statement] = []
    for statement in statements[statements.index(header) + 1 :]:
        if statement.kind != "keyval":
            break
        body.append(statement)
    if body:
        at = body[-1].line_end
    else:
        at = header.line_end
        while at < len(text):
            stop = text.find("\n", at)
            line_end = len(text) if stop == -1 else stop + 1
            if not text[at:line_end].lstrip(" \t").startswith("#"):
                break
            at = line_end
    lead = newline if at == len(text) and text and not text.endswith("\n") else ""
    block = f"{KEY} = [{newline}{ITEM_INDENT}{item}{newline}]{newline}"
    return text[:at] + lead + block + text[at:]


def _add_table(text: str, item: str) -> str:
    """Add a ``[voices]`` table with ``allow_sha256`` at the end of the file, after a blank line. The new lines
    end as the file's last line does."""
    newline = _ending_at(text, len(text))
    lead = newline if text and not text.endswith("\n") else ""
    if (text + lead).strip() and not (text + lead).endswith(("\n\n", "\n\r\n")):
        lead += newline
    return f"{text}{lead}[{TABLE}]{newline}{KEY} = [{newline}{ITEM_INDENT}{item}{newline}]{newline}"


# ==================================================================== helpers
def comment_text(note: str) -> str:
    """``note`` as the text of a one-line TOML comment. A control character (a newline would end the comment
    and start TOML), a line separator or an unpaired surrogate becomes U+FFFD, and a tab a space."""
    out: list[str] = []
    for char in note:
        if char == "\t":
            out.append(" ")
        elif unicodedata.category(char) in {"Cc", "Cs", "Zl", "Zp"}:
            out.append("\ufffd")
        else:
            out.append(char)
    return "".join(out).strip()


def _parse(text: str) -> dict[str, Any]:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise AllowlistEditError(f"the file is not valid TOML ({exc})") from exc


def _listed(data: dict[str, Any]) -> list[str]:
    table = data.get(TABLE, {})
    if not isinstance(table, dict):
        raise AllowlistEditError(f"{TABLE} is not a table")
    listed = cast("dict[str, Any]", table).get(KEY, [])
    if not isinstance(listed, list):
        raise AllowlistEditError(f"{TABLE}.{KEY} is not a list")
    items = cast("list[Any]", listed)
    if not all(isinstance(v, str) for v in items):
        raise AllowlistEditError(f"{TABLE}.{KEY} is not a list of strings")
    return [str(v) for v in items]


def _find(
    statements: Sequence[_Statement], path: tuple[str, ...], *, kind: StatementKind = "keyval"
) -> _Statement | None:
    for statement in statements:
        if statement.kind == kind and statement.path == path:
            return statement
    return None


def _ending_at(text: str, pos: int) -> str:
    """The line ending (CRLF or LF) of the line that holds ``pos``. For a last line that has none, the ending of
    the line before it; LF for a text with no line break at all."""
    stop = text.find("\n", pos)
    if stop == -1:
        stop = text.rfind("\n", 0, pos)
    if stop == -1:
        return "\n"
    return "\r\n" if stop > 0 and text[stop - 1] == "\r" else "\n"


def _indent_before(text: str, pos: int) -> str:
    """The spaces and tabs from the start of ``pos``'s line to ``pos``; "" when other text is there too."""
    lead = text[text.rfind("\n", 0, pos) + 1 : pos]
    return lead if lead.strip(" \t") == "" else ""


def _item_indent(text: str, items: Sequence[tuple[int, int]]) -> str | None:
    """The indent of the first item that begins a line of its own, or None."""
    for start, _ in items:
        lead = text[text.rfind("\n", 0, start) + 1 : start]
        if lead.strip(" \t") == "":
            return lead
    return None


def _canonical(data: Any) -> str:
    """Parsed TOML in a form that compares equal exactly when the values do (NaN equals NaN; 1 is not 1.0)."""
    return json.dumps(data, sort_keys=True, ensure_ascii=False, default=repr)


__all__ = [
    "ITEM_INDENT",
    "KEY",
    "PATH",
    "TABLE",
    "AllowlistEditError",
    "Entry",
    "HandEdit",
    "add_hash",
    "comment_text",
    "entries",
    "hand_edit",
]
