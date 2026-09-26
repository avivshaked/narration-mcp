"""Windows path rules as pure string logic (design sections 17.2 and 17.3; WP19).

Everything here works on any OS: it uses ``ntpath``, never ``os.path``, and touches no file. That keeps the
rules testable on Linux CI too. The file-system half of the checks (drive types, links, reparse points,
regular files) lives in ``narration.platform._windows``.

A path that breaks a rule raises ``NarrationError(PATH_NOT_ALLOWED)``. Its ``details`` hold the ``path`` as
given and the ``rule`` it broke, one of ``PathRule``.
"""

from __future__ import annotations

import ntpath
from collections.abc import Sequence
from typing import Final, Literal

from narration.contracts import codes
from narration.contracts.errors import NarrationError

PathRule = Literal[
    "empty",
    "not_absolute",
    "network",
    "device",
    "reserved_name",
    "invalid_name",
    "no_such_drive",
    "not_found",
    "unreadable",
    "not_regular_file",
    "link_loop",
    "reparse_point",
    "outside_root",
]
"""Why a path was refused; ``details["rule"]`` of the ``PATH_NOT_ALLOWED`` error."""

RESERVED_NAMES: Final[frozenset[str]] = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"{port}{digit}" for port in ("COM", "LPT") for digit in "0123456789\u00b9\u00b2\u00b3"}
)
"""Windows device names. A file name whose stem is one of these opens the device, whatever its extension.

The list errs on the side of caution, as CPython's ``ntpath.isreserved`` (3.13) does: ``COM0``/``LPT0`` and the
superscript-digit forms (``COM¹``) are reserved on some Windows versions and not others.
"""

_INVALID_CHARS: Final[frozenset[str]] = frozenset('<>:"|?*') | frozenset(chr(c) for c in range(32))

STORE_PATH_HINT: Final = (
    "The service writes only inside its store and never through a link. Ask the operator to check the store "
    "folder for links, junctions or reserved names (narration-admin doctor)."
)
"""The hint of a refused store path (section 17.2): the store's own layout is at fault, not the caller."""


def path_error(rule: PathRule, message: str, path: str, *, hint: str | None = None) -> NarrationError:
    """A ``PATH_NOT_ALLOWED`` error for ``path``, naming the rule it broke."""
    return NarrationError(codes.PATH_NOT_ALLOWED, message, hint=hint, details={"path": path, "rule": rule})


def is_reserved_name(name: str) -> bool:
    """Whether one path component names a Windows device (``CON``, ``nul.txt``, ``COM1.wav``, ``lpt¹``, ``aux .x``).

    The stem before the first dot (and before any ``:stream``), with trailing spaces dropped, is compared
    case-insensitively with ``RESERVED_NAMES``.
    """
    stem = name.split(".", 1)[0].split(":", 1)[0].rstrip(" ")
    return stem.upper() in RESERVED_NAMES


def name_problem(name: str) -> tuple[PathRule, str] | None:
    """What is wrong with one path component, or None.

    Refused: an empty name, ``.`` and ``..``, a colon (an alternate data stream), the characters
    ``< > " | ? *`` and control characters, a trailing dot or space (Windows silently drops them, so the name
    would alias another), and reserved device names.
    """
    if name in ("", ".", ".."):
        return "invalid_name", f"the path has an empty, '.' or '..' component ({name!r})"
    if ":" in name:
        return "invalid_name", f"{name!r} contains a colon; alternate data streams are refused"
    bad = sorted({c for c in name if c in _INVALID_CHARS})
    if bad:
        return "invalid_name", f"{name!r} contains characters Windows does not allow in a name: {bad!r}"
    if name.endswith((".", " ")):
        return "invalid_name", f"{name!r} ends with a dot or a space, which Windows drops"
    if is_reserved_name(name):
        return "reserved_name", f"{name!r} is a reserved Windows device name"
    return None


def strip_verbatim(path: str) -> str:
    r"""Turn the verbatim form Windows stores in links back into an ordinary path.

    ``\\?\C:\x`` and ``\??\C:\x`` become ``C:\x``; ``\\?\UNC\server\share`` becomes ``\\server\share``.
    Anything else (a volume GUID path, for one) is returned unchanged, and ``parse_absolute`` then refuses it.
    """
    for prefix in ("\\\\?\\", "\\??\\"):
        if path.startswith(prefix):
            rest = path[len(prefix) :]
            if rest[:4].upper() == "UNC\\":
                return "\\\\" + rest[4:]
            if len(rest) >= 3 and rest[0].isascii() and rest[0].isalpha() and rest[1:3] == ":\\":
                return rest
    return path


def parse_absolute(path: str, *, what: str = "the path") -> tuple[str, tuple[str, ...]]:
    r"""Check a caller's path as text (section 17.3) and split its normal form into drive and names.

    Returns ``("C:", ("folder", "clip.wav"))``. ``/`` counts as ``\``, and ``.`` and ``..`` are collapsed as
    Windows itself does before it opens anything. Refused, before any file is touched:

    * an empty path or one with a NUL character;
    * a device or namespace path (``\\.\``, ``\\?\``, ``\??\``);
    * a network path (``\\server\share``, ``\\?\UNC\``);
    * a relative, rooted (``\x``) or drive-relative (``C:x``) path;
    * a component ``name_problem`` refuses, such as a reserved device name.

    ``what`` names the path in messages (for example "the link target of ...").
    """
    if not path or "\x00" in path:
        raise path_error("empty", f"{what} is empty or contains a NUL character", path)
    p = path.replace("/", "\\")
    if p[:8].upper() in ("\\\\?\\UNC\\", "\\??\\UNC\\"):
        raise path_error("network", f"{what} is a network path; only local drives are read", path)
    if p.startswith(("\\\\?\\", "\\\\.\\", "\\??\\")):
        raise path_error("device", f"{what} is a device or namespace path (\\\\?\\ or \\\\.\\)", path)
    if p.startswith("\\\\"):
        raise path_error("network", f"{what} is a network path (\\\\server\\share); only local drives are read", path)
    drive, rest = ntpath.splitdrive(p)
    if len(drive) != 2 or not (drive[0].isascii() and drive[0].isalpha()) or not rest.startswith("\\"):
        raise path_error(
            "not_absolute", f"{what} is not absolute: give a full path starting with a drive (C:\\...)", path
        )
    names = tuple(n for n in ntpath.normpath(p)[2:].split("\\") if n)
    for name in names:
        problem = name_problem(name)
        if problem is not None:
            raise path_error(problem[0], problem[1], path)
    return drive.upper(), names


def split_under_root(path: str, root: str) -> tuple[str, ...]:
    r"""The names of ``path`` below ``root`` (section 17.2), checked as text; ``()`` when ``path`` is ``root``.

    Both are normalised as ``parse_absolute`` does and compared name by name, case-insensitively, so
    ``D:\store2`` is not inside ``D:\store``. Refused: a relative ``path``, one outside ``root`` (``..``
    included), and a name below the root that ``name_problem`` refuses. The root itself is the operator's
    choice and is not checked here.
    """
    p = path.replace("/", "\\")
    drive, rest = ntpath.splitdrive(p)
    if not drive or not rest.startswith("\\"):
        raise path_error("not_absolute", "a store path must be absolute", path, hint=STORE_PATH_HINT)
    below = relative_names(p, root)
    if below is None:
        raise path_error("outside_root", "the path is outside the store root", path, hint=STORE_PATH_HINT)
    for name in below:
        problem = name_problem(name)
        if problem is not None:
            raise path_error(problem[0], problem[1], path, hint=STORE_PATH_HINT)
    return below


def relative_names(path: str, root: str) -> tuple[str, ...] | None:
    """The normalised names of ``path`` below ``root``, or None if it is not ``root`` or inside it."""
    p_drive, p_names = _split(path)
    r_drive, r_names = _split(root)
    if ntpath.normcase(p_drive) != ntpath.normcase(r_drive) or len(p_names) < len(r_names):
        return None
    if any(ntpath.normcase(a) != ntpath.normcase(b) for a, b in zip(p_names, r_names, strict=False)):
        return None
    return p_names[len(r_names) :]


def _split(path: str) -> tuple[str, tuple[str, ...]]:
    drive, rest = ntpath.splitdrive(ntpath.normpath(path.replace("/", "\\")))
    return drive, tuple(n for n in rest.split("\\") if n)


def join(drive: str, names: Sequence[str]) -> str:
    """``drive`` + ``\\`` + the names, joined with ``\\``."""
    return drive + "\\" + "\\".join(names)
