"""``narration-admin voices allow <clip.wav> [--yes]`` and ``voices list [--json]`` (DC-17; design sections 16, 17.4).

The service clones a clip only if it designed the clip (its provenance list) or the owner allowed the clip's
sha256 in ``[voices] allow_sha256`` (section 17.4). ``allow`` adds a clip designed elsewhere to that list, so
the owner need not hash it and edit ``narration.toml`` by hand:

1. The configuration is the one every command uses (``find_config``), and it must load before it is changed.
2. The clip is read through the platform's path check (section 17.3): a local file, not a network or device
   path. The path is made absolute first (``os.path.abspath``, see ``read_clip``), so a relative path is
   taken from the operator's working folder. The file is read once, at most 20 MB (``MAX_CLIP_BYTES``), and
   the bytes that are hashed are the bytes checked to be a WAV with audio in it.
3. Its path, length and sha256 are printed, and the operator is asked to confirm that it is synthetic, not a
   recording of a real person. Only ``yes``, typed in full, goes on. Any other answer, or no answer at all
   (stdin closed, or at its end), changes nothing and exits 1. A person must confirm: ``--yes`` confirms
   without asking, only for the operator's own scripts.
4. The hash is added in lower case, with a comment naming the clip's file (``narration.admin.allowlist``: a
   text edit that keeps every other line, comment and line ending). The edited file must load with every
   other setting unchanged. It is written to a temporary name and renamed over the file (``os.replace``).
   The file is read again just before each rename attempt, and nothing is written if it changed while the
   command ran. A short window remains between that read and the rename: there is no lock between writers.
   The file keeps its POSIX mode bits, but not its owner, group or ACL (``_replace``).

A hash already listed changes nothing. A clip longer than ``[limits] max_clip_seconds`` is warned about: the
service would refuse to clone it as the limit stands.

**What reads the list, and when.** Each process reads the configuration once, when it starts:

- ``narration-mcp`` (one per MCP client) checks the list when ``measure_voice``, ``submit_job`` or
  ``audition_pronunciation`` names a clip (``narration.backend``);
- the daemon checks it again when it opens a ``measure`` or ``generate`` job, before a worker sees the clip
  (``narration.jobs.engine``, ``narration.measure.handler``).

So both must be restarted to read a new hash; ``allow`` ends by saying so (``restart_advice``).

``list`` prints the list, each hash with the comment on its line.

**Never an MCP tool** (DC-17): the allowlist is the synthetic-voices gate, and a tool would let any caller
allow a recording of a real person. Only the operator, at this machine, changes it.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import io
import json
import os
import secrets
import shutil
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import soundfile

from narration.backend.clips import WAV_FORMATS
from narration.config import Config, parse_config
from narration.contracts.errors import ConfigError, NarrationError
from narration.contracts.interfaces import Platform
from narration.jobs.voice import MAX_CLIP_BYTES
from narration.store.files import REPLACE_ATTEMPTS, fsync_dir

from . import allowlist
from .cli import EXIT_OK, PROGRAM, Admin, AdminError, Subparsers

CONFIRMATIONS: Final = frozenset({"yes"})
"""The answers that confirm a clip is synthetic (compared without regard to case or surrounding spaces). Only
the whole word: section 17.4's gate wants an explicit confirmation, and the question says "Type yes"."""
SECTION: Final = f"[{allowlist.TABLE}] {allowlist.KEY}"


@dataclass(frozen=True, slots=True)
class Clip:
    """A clip as ``allow`` read it: the file (links resolved), its sha256 and its audio."""

    path: Path
    sha256: str
    seconds: float
    samplerate: int
    channels: int


def register(subparsers: Subparsers) -> None:
    """Add ``voices allow`` and ``voices list`` (``narration.admin.cli``)."""
    parser = subparsers.add_parser(
        "voices",
        help="allow a synthetic clip designed elsewhere, by its sha256, or list the allowed ones (section 17.4)",
        description=(
            "The service clones only synthetic voices: a clip it designed, or one whose sha256 is in "
            f"{SECTION} (section 17.4). These commands add a clip to that list, or show it."
        ),
    )
    commands = parser.add_subparsers(title="voices commands", metavar="<voices command>", required=True)

    allow = commands.add_parser(
        "allow",
        help=f"add a synthetic clip's sha256 to {SECTION}, after you confirm it is synthetic",
        description=(
            f"Read the clip (a WAV on a local drive), print its path, length and sha256, ask you to confirm it is "
            f"synthetic and not a recording of a real person, then add its sha256 to {SECTION} in the "
            "configuration file, with a comment naming the clip. Every other line and comment of the file is "
            "kept. narration-mcp and the daemon read the list when they start: restart them afterwards. A person "
            "must confirm the clip; an agent or MCP client must not allow one."
        ),
    )
    allow.add_argument("clip", help="the clip: a WAV file of a synthetic voice")
    allow.add_argument(
        "--yes",
        action="store_true",
        help="confirm, without being asked, that the clip is synthetic: only for the operator's own scripts, once "
        "the operator knows the clip is synthetic",
    )
    allow.set_defaults(handler=allow_clip)

    listing = commands.add_parser(
        "list",
        help=f"print {SECTION}",
        description=f"Print each sha256 in {SECTION}, with the comment on its line. Changes nothing.",
    )
    listing.add_argument("--json", action="store_true", help="print the list as JSON")
    listing.set_defaults(handler=list_clips)


# ---------------------------------------------------------------- allow
def allow_clip(admin: Admin, args: argparse.Namespace) -> int:
    """``voices allow``: see the module docstring."""
    path = admin.config_path()
    config = admin.config()
    clip = read_clip(admin.platform(), args.clip)
    channels = "channel" if clip.channels == 1 else "channels"
    admin.say(f"clip:    {clip.path}")
    admin.say(f"length:  {clip.seconds:.2f} s ({clip.samplerate} Hz, {clip.channels} {channels})")
    admin.say(f"sha256:  {clip.sha256}")
    admin.say(f"config:  {path}")
    limit = config.limits.max_clip_seconds
    if clip.seconds > limit:
        admin.warn(
            f"warning: the clip is {clip.seconds:.1f} s long and [limits] max_clip_seconds is {limit}, so the "
            "service will refuse to clone it (UNSUPPORTED_AUDIO) unless that limit is raised."
        )
    if _listed(config, clip.sha256):
        _say_already_listed(admin)
        return EXIT_OK
    by_hand = _by_hand(path, clip)
    if not os.access(path, os.W_OK):
        raise AdminError(f"{path} is read-only, so nothing changed. Make it writable and run this again. {by_hand}")
    if not args.yes:
        confirm_synthetic(admin)
    if not add_to_file(path, clip, by_hand=by_hand):
        _say_already_listed(admin)  # it was added since the configuration was read
        return EXIT_OK
    admin.say(f"\nAdded it to {SECTION} in {path}.")
    admin.say()
    admin.say(restart_advice(path, autostart=config.daemon.autostart))
    return EXIT_OK


def read_clip(platform: Platform, given: str) -> Clip:
    """The clip at ``given``, read through section 17.3's path check; ``AdminError`` when it is not a local
    file, cannot be read, is over 20 MB, or is not a WAV with audio in it.

    The path is first made absolute with ``os.path.abspath`` (after ``~`` is expanded). A relative path is thus
    taken from the operator's working folder; the daemon has no such folder, which is why a request's path
    must be absolute. On Windows this also normalises the path the Windows way before the check sees it: a
    drive-relative ``C:clip.wav`` or a rooted ``\\clip.wav`` becomes absolute, and a trailing dot or space on a
    name is dropped. Windows opens the same file either way, and the hash and the note name the file the
    check resolved.
    """
    absolute = os.path.abspath(os.path.expanduser(given))
    try:
        source = platform.check_readable_path(absolute)
    except NarrationError as exc:
        raise AdminError(f"{given}: {exc.message} ({exc.code}), so nothing changed. {exc.hint}") from exc
    try:
        with open(source, "rb") as handle:
            data = handle.read(MAX_CLIP_BYTES + 1)
    except OSError as exc:
        raise AdminError(f"cannot read {source} ({exc.strerror or type(exc).__name__}), so nothing changed.") from exc
    if len(data) > MAX_CLIP_BYTES:
        raise AdminError(
            f"{source} is larger than 20 MB, and the service clones a clip of at most 20 MB (section 17.3), so "
            "it would never use this one. Nothing changed."
        )
    try:
        info = soundfile.info(io.BytesIO(data))
    except (soundfile.SoundFileError, RuntimeError, OSError, ValueError) as exc:
        raise AdminError(
            f"{source} is not an audio file the service reads (a voice clip is a WAV), so nothing changed."
        ) from exc
    if info.format not in WAV_FORMATS:
        raise AdminError(f"{source} is a {info.format} file, not a WAV (a voice clip is a WAV), so nothing changed.")
    if info.frames <= 0 or info.samplerate <= 0:
        raise AdminError(f"{source} is a WAV with no audio in it, so nothing changed.")
    return Clip(
        path=source,
        sha256=hashlib.sha256(data).hexdigest(),
        seconds=info.frames / info.samplerate,
        samplerate=int(info.samplerate),
        channels=int(info.channels),
    )


def confirm_synthetic(admin: Admin) -> None:
    """Ask the operator to confirm the clip is synthetic; ``AdminError`` (exit 1) unless the answer is yes."""
    admin.say()
    admin.say(
        "The service clones only synthetic voices (section 17.4). Allow this clip only if a speech synthesiser "
        "made it. Never allow a recording of a real person, whoever it is."
    )
    answer = admin.ask("Is this clip synthetic? Type yes to allow it: ")
    if answer is None:
        raise AdminError(
            "no answer (standard input is closed or at its end), so nothing changed. A person must confirm that "
            "the clip is synthetic: the operator runs this command in a terminal and types yes. An agent or an MCP "
            "client must not allow a clip; --yes is only for the operator's own scripts."
        )
    given = answer.strip()
    if given.lower() not in CONFIRMATIONS:
        said = f"the answer was {given!r}" if given else "the answer was empty"
        raise AdminError(
            f"not confirmed ({said}, not yes), so nothing changed. Run it again and answer yes once you know "
            "the clip is synthetic."
        )


def add_to_file(path: Path, clip: Clip, *, by_hand: str) -> bool:
    """Add ``clip``'s sha256 to the configuration file at ``path`` (the module docstring, step 4). False when it
    is listed already, and the file is left as it was. ``AdminError`` when it cannot be added safely; the
    file is then unchanged, and the message ends with ``by_hand``."""
    try:
        before = path.read_bytes()
    except OSError as exc:
        raise AdminError(f"cannot read {path} ({exc.strerror or type(exc).__name__}), so nothing changed.") from exc
    try:
        text = before.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AdminError(f"{path} is not UTF-8 text, so nothing changed. {by_hand}") from exc
    try:
        edited = allowlist.add_hash(text, clip.sha256, str(clip.path))
    except allowlist.AllowlistEditError as exc:
        raise AdminError(f"{path}: {exc}, so nothing changed. {by_hand}") from exc
    if edited is None:
        return False
    _check_same_settings(path, text, edited, clip.sha256, by_hand=by_hand)
    _replace(path, before, edited.encode("utf-8"), by_hand=by_hand)
    return True


def restart_advice(path: Path, *, autostart: bool) -> str:
    """What to restart so the new hash is read (the module docstring: what reads the list, and when).

    The daemon comes first. A front-end restarted while the old daemon still runs would admit a job for the
    clip, and the old daemon, checking with its old list, would then fail it (``VOICE_NOT_SYNTHETIC``)."""
    start_again = (
        "the next job starts it again ([daemon] autostart)"
        if autostart
        else f"then start it with `{PROGRAM} daemon start` ([daemon] autostart is off)"
    )
    return "\n".join(
        (
            "narration-mcp and the daemon read the list only when they start, so the ones running now do not",
            "know the new hash yet. Restart both, in this order:",
            "  1. the daemon, which checks the list again before it measures a voice or clones a clip: run",
            f"     `{PROGRAM} daemon stop` (it finishes the segment in flight; queued jobs wait); {start_again}.",
            f"     If no daemon runs (`{PROGRAM} daemon status`), there is nothing to stop.",
            "  2. narration-mcp, the server each MCP client starts, which checks the list when measure_voice,",
            "     submit_job or audition_pronunciation names a clip: restart or reconnect it in every client",
            "     that runs it, or start a new session there.",
            f"A server or daemon started with another configuration file than {path} does not read this one.",
        )
    )


def _listed(config: Config, sha256: str) -> bool:
    return any(listed.lower() == sha256.lower() for listed in config.voices.allow_sha256)


def _say_already_listed(admin: Admin) -> None:
    admin.say(f"\nIt is already in {SECTION}, so nothing changed.")
    admin.say(
        "narration-mcp and the daemon read the list when they start: one started before the hash was added "
        "does not know it until it is restarted."
    )


def _by_hand(path: Path, clip: Clip) -> str:
    """What to add by hand, fitted to how the file writes ``allow_sha256`` (``allowlist.hand_edit``)."""
    try:
        text: str | None = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        text = None
    hint = allowlist.hand_edit(text, clip.sha256, str(clip.path))
    if hint.kind == "line":
        return f"To allow the clip by hand, add this line to {SECTION} in {path}:\n    {hint.text}"
    if hint.kind == "key":
        return f"To allow the clip by hand, add this line to the [{allowlist.TABLE}] table in {path}:\n    {hint.text}"
    return (
        f"To allow the clip by hand, add this value inside the brackets of {SECTION} in {path}, after a comma if "
        f"the list is not empty, with no comment after it on that line:\n    {hint.text}"
    )


def _as_loaded(path: Path, text: str) -> Config:
    """``text`` as ``load_config`` would load it from ``path`` (it reads with universal newlines)."""
    data = tomllib.loads(text.replace("\r\n", "\n").replace("\r", "\n"))
    return parse_config(data, base=path.resolve().parent, path=path.resolve())


def _check_same_settings(path: Path, text: str, edited: str, sha256: str, *, by_hand: str) -> None:
    """The edited file must load, with the hash appended and every other setting as it was."""
    try:
        old, new = _as_loaded(path, text), _as_loaded(path, edited)
    except (tomllib.TOMLDecodeError, ConfigError) as exc:
        raise AdminError(f"{path}: the edited file would not load ({exc}), so nothing changed. {by_hand}") from exc
    voices = dataclasses.replace(old.voices, allow_sha256=(*old.voices.allow_sha256, sha256))
    # repr, not ==: a NaN setting is equal to itself in its repr only.
    if repr(new) != repr(dataclasses.replace(old, voices=voices)):
        raise AdminError(f"{path}: the edit would change other settings, so nothing changed. {by_hand}")


def _replace(path: Path, before: bytes, data: bytes, *, by_hand: str) -> None:
    """Write ``data`` to a temporary name beside ``path``, then rename it over ``path`` (``_replace_unchanged``:
    not if the file no longer holds ``before``).

    What carries over: the POSIX mode bits (``shutil.copymode``; on Windows only the read-only flag). What
    does not: the owner and group, an ACL, and other attributes. The renamed file is a new file, so it has
    those of a new file in the folder. A hard link to the old file keeps the old content.
    """
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        with open(tmp, "xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        shutil.copymode(path, tmp)
        _replace_unchanged(tmp, path, before)
    except OSError as exc:
        raise AdminError(
            f"cannot write {path} ({exc.strerror or type(exc).__name__}), so nothing changed. {by_hand}"
        ) from exc
    finally:
        with contextlib.suppress(OSError):  # gone after the rename; a leftover is only a stray temporary file
            tmp.unlink(missing_ok=True)
    fsync_dir(path.parent)


def _replace_unchanged(tmp: Path, path: Path, before: bytes) -> None:
    """``os.replace(tmp, path)``, but only while ``path`` still holds ``before``.

    Before every attempt the file is read again. If it changed (an editor saved it, another run added a hash)
    or is gone, this raises the "changed while this command ran" ``AdminError`` and writes nothing. Windows
    refuses a rename, or a read, while another program has the file open for a moment (an editor saving, a
    virus scan). Both are then retried for about a second in all, as the store does
    (``narration.store.files``), and each retry reads the file again first.

    A short window remains between the last read and the rename. A change made in it is overwritten, since
    there is no lock between writers. Two ``voices allow`` runs at the same moment can therefore lose one hash.
    """
    refused: PermissionError | None = None
    for attempt in range(REPLACE_ATTEMPTS):
        if attempt:
            time.sleep(0.005 * attempt)
        try:
            current: bytes | None = path.read_bytes()
        except FileNotFoundError:
            current = None
        except PermissionError as exc:  # it is being replaced this moment: read it again
            refused = exc
            continue
        if current != before:
            raise AdminError(f"{path} changed while this command ran, so nothing changed. Run it again.")
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:  # another program has it open: read it again, then try again
            refused = exc
    assert refused is not None
    raise refused


# ---------------------------------------------------------------- list
def list_clips(admin: Admin, args: argparse.Namespace) -> int:
    """``voices list``: each allowed sha256, with the comment on its line."""
    path = admin.config_path()
    admin.config()  # it must load
    try:
        found = allowlist.entries(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, allowlist.AllowlistEditError) as exc:
        raise AdminError(f"cannot read {SECTION} from {path} ({exc}).") from exc
    if args.json:
        listed = [{"sha256": entry.sha256, "note": entry.note} for entry in found]
        admin.say(json.dumps({"config": str(path), "allow_sha256": listed}, ensure_ascii=False, indent=2))
        return EXIT_OK
    if not found:
        admin.say(f"{SECTION} in {path} is empty: no clip designed elsewhere is allowed.")
        admin.say(f"Clips this service designed need no entry. Add a clip with `{PROGRAM} voices allow <clip.wav>`.")
        return EXIT_OK
    clips = "clip" if len(found) == 1 else "clips"
    admin.say(f"{SECTION} in {path}: {len(found)} {clips}")
    for entry in found:
        admin.say(f"  {entry.sha256}  {entry.note}" if entry.note else f"  {entry.sha256}")
    admin.say("narration-mcp and the daemon read this list when they start.")
    return EXIT_OK
