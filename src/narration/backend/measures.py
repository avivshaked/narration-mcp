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
whitespace at the edges or inside, typographic or plain quotes, dashes and ellipses) and, when one of them is
measured for this clip, says where the transcript sent first differs from it. No key changes: the hash is
computed as always, once per spelling tried.
"""

from __future__ import annotations

import unicodedata
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


def transcript_variants(transcript: str) -> list[tuple[str, str]]:
    """The other spellings of ``transcript`` a slip in sending it makes, fewest changes first: its edges
    trimmed, its whitespace collapsed, its quotes, dashes and ellipses plain or typographic, and each whitespace
    form with each punctuation form. Pairs of (spelling, what differs: ``whitespace``, ``punctuation`` or both);
    never the transcript itself, an empty one, or one twice."""
    spaces = ((transcript.strip(), WHITESPACE), (" ".join(transcript.split()), WHITESPACE))
    marks = ((transcript.translate(TO_PLAIN), PUNCTUATION), (_typographic(transcript), PUNCTUATION))
    both = [
        (mark(text), f"{WHITESPACE} and {PUNCTUATION}")
        for text, _ in spaces
        for mark in (lambda t: t.translate(TO_PLAIN), _typographic)
    ]
    seen = {transcript, ""}
    out: list[tuple[str, str]] = []
    for text, differs in (*spaces, *marks, *both):
        if text not in seen:
            seen.add(text)
            out.append((text, differs))
    return out


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
    punctuation (``measured_nearby``): its voice hash, what differs, and where the transcript sent first differs
    from the measured one (a character index into the transcript sent, and each side's character there)."""

    voice_hash: str
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
    for text, differs_in in transcript_variants(voice.transcript):
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
            differs_in=differs_in,
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
    return NarrationError(
        codes.VOICE_NOT_MEASURED,
        f"{exc.message}; this clip is measured under a transcript that differs from the one sent only in "
        f"{nearby.differs_in}, first at character {at}",
        field="voice.transcript",
        hint=(
            f"Send the transcript exactly as it was measured: at character {at} (counting from 0) yours has "
            f"{nearby.sent}, the measured one {nearby.measured}. It is the same clip; do not measure it again, "
            "since a measurement under this transcript would make it a second voice, with a cache of its own."
        ),
        details={**details, "transcript_mismatch": nearby.as_json()},
    )


__all__ = [
    "MEASURE_HINT",
    "Measurements",
    "NearbyMeasurement",
    "StoreMeasurements",
    "describe_char",
    "first_difference",
    "measured_nearby",
    "not_measured",
    "transcript_variants",
]
