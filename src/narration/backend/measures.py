"""A voice's measurement as the front-end looks it up (design sections 3.2, 7.6, 10.2): a narrow seam over
``narration.measure`` (WP33), which owns the rules.

- ``voice_hash``: the voice's hash, as the job engine computes it (``voice_hash_of``);
- ``require``: the measurement a generation is judged against, or ``VOICE_NOT_MEASURED`` (``field: voice``,
  with the hint to run ``measure_voice`` first), by the job engine's rule (``require_measured``);
- ``current``: ``measure_voice``'s answer at once: a stored measurement only when its key is the one a new
  measurement would have (the same voice, engine profile, corpus and ladder settings;
  ``current_measurement``); else None, and a ``measure`` job is queued.

``StoreMeasurements`` is the service's; tests pass their own ``Measurements``.

**A transcript that is not the one measured** (``measured_nearby``). The voice's hash covers its transcript
character for character (NFC only; section 10.2), so a transcript retyped, read from a file with a trailing
newline, or given curly quotes for straight ones is another voice, and is not measured. The measurement does not
keep the transcript it verified, and the store finds measurements only by ``voice_hash``, so the service cannot
look up "this clip, any transcript". It tries the spellings such a slip makes instead (``transcript_variants``:
the edges trimmed, the whitespace collapsed, a trailing newline added, quotes, dashes and ellipses made plain or
typographic) and, when one of them is measured for this clip, says which rewrites give the measured transcript
and where the transcript sent first differs from it. Caught: whitespace the transcript sent has and the measured
one has not, typographic quotes, dashes and ellipses sent for plain ones, straight quotes sent for typographic
ones, and a trailing newline the measured one had. Not caught: other whitespace the measured one had, and a
typographic dash or ellipsis it had where the one sent has a plain one. No key changes: the hash is computed as
always, once per spelling tried.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Protocol

from narration.config import Config
from narration.contracts import codes
from narration.contracts.errors import MaterialError, NarrationError
from narration.contracts.interfaces import Store
from narration.contracts.models import EngineProfile, MaterialSet, MeasurementRecord
from narration.jobs.plan import VoiceSpec
from narration.measure import current_measurement, load_corpus, require_measured, voice_hash_of


class Measurements(Protocol):
    """How the front-end asks about a voice's measurement (see the module docstring)."""

    def voice_hash(self, voice: VoiceSpec) -> str:
        """The voice's hash (section 10.2)."""
        ...

    def require(self, voice_hash: str, profile: EngineProfile) -> MeasurementRecord:
        """The voice's measurement under ``profile``; ``VOICE_NOT_MEASURED`` when there is none."""
        ...

    def current(self, voice_hash: str, profile: EngineProfile) -> MeasurementRecord | None:
        """The voice's measurement when it is current, else None (``measure_voice`` then queues a job)."""
        ...


class StoreMeasurements:
    """``Measurements`` over the store, by ``narration.measure``'s rules. The calibration corpus (for
    ``current``'s key) is read once, from the service's ``material/``."""

    def __init__(self, store: Store, config: Config) -> None:
        self.store = store
        self.config = config
        self._corpus: MaterialSet | None = None

    def voice_hash(self, voice: VoiceSpec) -> str:
        return voice_hash_of(clip_sha256=voice.sha256, transcript=voice.transcript, config=self.config)

    def require(self, voice_hash: str, profile: EngineProfile) -> MeasurementRecord:
        return require_measured(self.store, voice_hash=voice_hash, engine_profile_id=profile.engine_profile_id)

    def current(self, voice_hash: str, profile: EngineProfile) -> MeasurementRecord | None:
        return current_measurement(
            self.store, voice_hash=voice_hash, profile=profile, corpus=self._corpus_set(), config=self.config
        )

    def _corpus_set(self) -> MaterialSet:
        """The calibration corpus ``[measurement] corpus``; ``BACKEND_NOT_INSTALLED`` when it cannot be read."""
        if self._corpus is None:
            set_id = self.config.measurement.corpus
            try:
                self._corpus = load_corpus(set_id)
            except MaterialError as exc:
                raise NarrationError(
                    codes.BACKEND_NOT_INSTALLED,
                    f"the calibration corpus {set_id} cannot be read: {exc}",
                    hint="Reinstall the service: its material folder is missing or changed.",
                    details={"corpus": set_id},
                ) from exc
        return self._corpus


# ======================================================================== a transcript not the one measured

TO_PLAIN: Final = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u2026": "...",
    }
)
"""Typographic quotes, primes, dashes and the ellipsis, as plain ASCII."""
CONTROL_NAMES: Final[dict[str, str]] = {"\n": "LINE FEED", "\r": "CARRIAGE RETURN", "\t": "CHARACTER TABULATION"}
"""Names for the control characters a transcript read from a file may carry (``unicodedata`` has none)."""
WHITESPACE: Final = "whitespace"
PUNCTUATION: Final = "punctuation"
REWRITES: Final[dict[str, str]] = {
    "trim_edges": "remove the whitespace at the start and end of your transcript",
    "collapse_whitespace": (
        "write every run of whitespace in it (spaces, tabs, newlines) as one space, with none at the start or end"
    ),
    "add_trailing_newline": "end it with one newline",
    "plain_punctuation": "write its typographic quotes, dashes and ellipses as plain ' \" - and ...",
    "typographic_quotes": "write its straight quotes as typographic ones: \u2019 for ', and \u201c and \u201d for \"",
}
"""The rewrites ``transcript_variants`` tries, by name, and each as a step for the caller to take."""
KIND_OF_REWRITE: Final[dict[str, str]] = {
    "trim_edges": WHITESPACE,
    "collapse_whitespace": WHITESPACE,
    "add_trailing_newline": WHITESPACE,
    "plain_punctuation": PUNCTUATION,
    "typographic_quotes": PUNCTUATION,
}


def _typographic(text: str) -> str:
    """Straight quotes as typographic ones: every ``'`` as a right single quote (the apostrophe), and ``"``
    opening and closing in turn."""
    out: list[str] = []
    opening = True
    for char in text.replace("'", "\u2019"):
        if char == '"':
            out.append("\u201c" if opening else "\u201d")
            opening = not opening
        else:
            out.append(char)
    return "".join(out)


def transcript_variants(transcript: str) -> list[tuple[str, tuple[str, ...]]]:
    """The other spellings of ``transcript`` a slip in sending it makes, fewest changes first: its edges trimmed,
    its whitespace collapsed, a trailing newline added, its quotes, dashes and ellipses plain or typographic, and
    each whitespace rewrite with each punctuation one. Pairs of (spelling, the ``REWRITES`` that give it, in
    order); never the transcript itself, an empty one, or one twice (the first, fewest rewrites, is kept)."""

    def trim(text: str) -> str:
        return text.strip()

    def collapse(text: str) -> str:
        return " ".join(text.split())

    def plain(text: str) -> str:
        return text.translate(TO_PLAIN)

    spaces: tuple[tuple[str, Callable[[str], str]], ...] = (("trim_edges", trim), ("collapse_whitespace", collapse))
    marks: tuple[tuple[str, Callable[[str], str]], ...] = (
        ("plain_punctuation", plain),
        ("typographic_quotes", _typographic),
    )
    candidates: list[tuple[str, tuple[str, ...]]] = [(fix(transcript), (name,)) for name, fix in spaces]
    candidates.append((transcript + "\n", ("add_trailing_newline",)))
    candidates += [(fix(transcript), (name,)) for name, fix in marks]
    candidates += [
        (mark(space(transcript)), (space_name, mark_name)) for space_name, space in spaces for mark_name, mark in marks
    ]
    seen = {transcript, ""}
    out: list[tuple[str, tuple[str, ...]]] = []
    for text, rewrites in candidates:
        if text not in seen:
            seen.add(text)
            out.append((text, rewrites))
    return out


def differs_in(rewrites: tuple[str, ...]) -> str:
    """What the rewrites change: ``whitespace``, ``punctuation``, or ``whitespace and punctuation``."""
    kinds = [kind for kind in (WHITESPACE, PUNCTUATION) if any(KIND_OF_REWRITE[r] == kind for r in rewrites)]
    return " and ".join(kinds)


def first_difference(sent: str, other: str) -> int:
    """The index of the first character at which ``sent`` and ``other`` differ; the shorter one's length when
    one begins with the other."""
    for index, (a, b) in enumerate(zip(sent, other, strict=False)):
        if a != b:
            return index
    return min(len(sent), len(other))


def describe_char(text: str, index: int) -> str:
    """The character of ``text`` at ``index`` by code point and name (``U+2019 RIGHT SINGLE QUOTATION MARK``), or
    ``the end of the text``: enough to find it, without quoting the text."""
    if index >= len(text):
        return "the end of the text"
    char = text[index]
    name = CONTROL_NAMES.get(char) or unicodedata.name(char, "")
    return f"U+{ord(char):04X} {name}".rstrip()


@dataclass(frozen=True, slots=True, kw_only=True)
class NearbyMeasurement:
    """A measurement of the same clip under a transcript that differs from the one sent only in whitespace or
    punctuation (``measured_nearby``): its voice hash, the rewrites of the transcript sent that give the measured
    one (``REWRITES``' names, in order) and what they change, and where the transcript sent first differs from the
    measured one (a character index into the transcript sent, and each side's character there)."""

    voice_hash: str
    rewrites: tuple[str, ...]
    differs_in: str
    first_difference: int
    sent: str
    measured: str
    sent_chars: int
    measured_chars: int

    def as_json(self) -> dict[str, object]:
        """``VOICE_NOT_MEASURED``'s ``details.transcript_mismatch``."""
        return {
            "measured_voice_hash": self.voice_hash,
            "rewrites": list(self.rewrites),
            "differs_in": self.differs_in,
            "first_difference": self.first_difference,
            "sent": self.sent,
            "measured": self.measured,
            "sent_chars": self.sent_chars,
            "measured_chars": self.measured_chars,
        }


def measured_nearby(measurements: Measurements, voice: VoiceSpec, profile: EngineProfile) -> NearbyMeasurement | None:
    """The first of ``transcript_variants`` of the voice's transcript that is measured for this clip under
    ``profile``, by the same rule as ``require``; None when none is (see the module docstring)."""
    for text, rewrites in transcript_variants(voice.transcript):
        other = measurements.voice_hash(VoiceSpec(path=voice.path, sha256=voice.sha256, transcript=text))
        try:
            measurements.require(other, profile)
        except NarrationError as exc:
            if exc.code != codes.VOICE_NOT_MEASURED:
                raise
            continue
        index = first_difference(voice.transcript, text)
        return NearbyMeasurement(
            voice_hash=other,
            rewrites=rewrites,
            differs_in=differs_in(rewrites),
            first_difference=index,
            sent=describe_char(voice.transcript, index),
            measured=describe_char(text, index),
            sent_chars=len(voice.transcript),
            measured_chars=len(text),
        )
    return None


MEASURE_HINT: Final = (
    "Measure the clip first with measure_voice (a heavy GPU job; check get_server_status first). If this clip "
    "was measured before, send the transcript it was measured with instead, exactly as then: the voice is the "
    "clip and its transcript character for character, so any other character, space or quote is another voice."
)
"""``VOICE_NOT_MEASURED``'s hint when no spelling of the transcript is measured for the clip."""


def not_measured(
    exc: NarrationError, voice: VoiceSpec, voice_hash: str, profile: EngineProfile, nearby: NearbyMeasurement | None
) -> NarrationError:
    """``VOICE_NOT_MEASURED`` as ``submit_job`` answers it, from the error ``require`` raised: the voice in
    ``details``, and either where its transcript differs from the one this clip is measured under (``nearby``;
    ``field: voice.transcript``: send that one, do not measure again) or the hint to measure it, or to send the
    measured transcript exactly if the clip was measured before. Neither transcript is quoted: the details
    point to the character."""
    details: dict[str, object] = {
        **(exc.details or {}),
        "voice_hash": voice_hash,
        "clip_sha256": voice.sha256,
        "engine_profile_id": profile.engine_profile_id,
    }
    if nearby is None:
        return NarrationError(
            codes.VOICE_NOT_MEASURED, exc.message, field=exc.field, hint=MEASURE_HINT, details=details
        )
    at = nearby.first_difference
    steps = "; then ".join(REWRITES[r] for r in nearby.rewrites)
    return NarrationError(
        codes.VOICE_NOT_MEASURED,
        f"{exc.message}; this clip is measured under a transcript that differs from the one sent only in "
        f"{nearby.differs_in}, first at character {at}",
        field="voice.transcript",
        hint=(
            f"Send the transcript as this clip was measured with it: {steps}. The first difference is at "
            f"character {at}, counting from 0: yours has {nearby.sent}, the measured one {nearby.measured}. It is "
            "the same clip; do not measure it again, since a measurement under this transcript would make it a "
            "second voice, with a cache of its own."
        ),
        details={**details, "transcript_mismatch": nearby.as_json()},
    )


__all__ = [
    "MEASURE_HINT",
    "REWRITES",
    "Measurements",
    "NearbyMeasurement",
    "StoreMeasurements",
    "describe_char",
    "differs_in",
    "first_difference",
    "measured_nearby",
    "not_measured",
    "transcript_variants",
]
