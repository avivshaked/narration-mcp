"""``Scorer``: the ``QaScorer`` of design section 11.1 (plan.md WP14), plain Python.

A take's verdict depends only on that take and the request's inputs for it (``QaInputs``), never on other
takes: ``score`` reads nothing but its argument and the scorer's fixed settings, and keeps no state between
calls. Suggestions, the consistency report and ``listen_first`` are computed afterwards from scored takes and
change no verdict.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from narration.config import MeasurementConfig
from narration.contracts.interfaces import QaInputs, ScoredTake
from narration.contracts.models import (
    Anchor,
    Consistency,
    Flag,
    ListenFirstItem,
    MeasurementRecord,
    Pace,
    QaMetrics,
    QaResult,
    QaThresholds,
    SegmentResult,
    SimilarityBaseline,
    Suggestion,
)

from . import consistency as _consistency
from . import listen_first as _listen_first
from . import report as _report
from . import suggest as _suggest
from ._flags import verdict_of
from .checks import alignment_flags, pace_check, signal_flags, speaker_check
from .normaliser import NumberReader
from .profile import DEFAULT_PROFILE, QaProfile
from .textmatch import match_text

__all__ = ["Scorer", "VoiceFacts", "voice_facts"]


@dataclass(frozen=True, slots=True)
class VoiceFacts:
    """The voice's measured facts one take is judged against: anchor, similarity baseline and pace curve."""

    anchor: Anchor | None
    similarity: SimilarityBaseline | None
    pace: Pace | None


def voice_facts(inputs: QaInputs) -> VoiceFacts:
    """Where the anchor, baseline and pace come from (``QaInputs``' docstring).

    A finished measurement supplies all three. Work without one (the measurement's own ladder takes, an
    audition) passes what exists in ``anchor``, ``similarity`` and ``pace``; what is missing is not checked.
    """
    m = inputs.measurement
    if m is not None:
        return VoiceFacts(m.anchor, m.similarity, m.pace)
    return VoiceFacts(inputs.anchor, inputs.similarity, inputs.pace)


class Scorer:
    """QA logic for takes (``narration.contracts.interfaces.QaScorer``).

    ``config`` supplies the speaker margins and the pace-tolerance floor (``[measurement]`` in design section
    16); ``profile`` the fixed thresholds of ``default.v3``. Both are fixed for the scorer's lifetime.
    """

    def __init__(self, config: MeasurementConfig | None = None, profile: QaProfile = DEFAULT_PROFILE) -> None:
        self._config = config if config is not None else MeasurementConfig()
        self._profile = profile
        self._reader = NumberReader()

    @property
    def profile_version(self) -> str:
        """``names.QA_PROFILE``, the version of the thresholds in force."""
        return self._profile.name

    @property
    def number_reader(self) -> str:
        """``names.NUMBER_READER``, the version of the exact-span reading."""
        return self._reader.version

    def score(self, inputs: QaInputs) -> QaResult:
        """One take's QA: signal, text match with the word-count rule, exact spans, terms, insertions, speaker,
        pace and the aligner's cue flags, then the verdict (section 11.1, thresholds ``default.v3``).

        The result holds nothing the analysis key does not cover: no flag carries a segment id, and each exact
        result is placed by its ``words`` (its ``start``/``end`` are the span's code points in the cue's spoken
        text, which the job assembler replaces with the request's own). Raises ``QaUnavailable`` when a check
        the inputs call for cannot run (``narration.qa.errors``).
        """
        segment = inputs.segment
        facts = voice_facts(inputs)

        signal = signal_flags(inputs.signal, inputs.hit_token_cap, self._profile)
        text = match_text(segment, inputs.hints, inputs.asr_text, inputs.voice_transcript, self._reader, self._profile)
        speaker = speaker_check(inputs.embedding, facts.anchor, facts.similarity, self._config)
        pace = pace_check(segment, inputs.signal, facts.pace, self._config, self._profile)
        cues = alignment_flags(inputs.alignment, segment)

        flags: tuple[Flag, ...] = (*signal, *text.flags, *speaker.flags, *pace.flags, *cues)
        return QaResult(
            verdict=verdict_of(flags),
            transcript=inputs.asr_text,
            exact=text.exact,
            terms=text.terms,
            metrics=QaMetrics(
                wer_raw=text.wer_raw,
                wer_adj=text.wer_adj,
                word_errors=text.word_errors,
                exact_ok=text.exact_ok,
                spk_sim_anchor=speaker.similarity,
                spoken_wpm=pace.spoken_wpm,
                expected_spoken_wpm=pace.expected_wpm,
                head_insertion_words=text.head_count,
                end_insertion_words=text.end_count,
                longest_silence_s=inputs.signal.longest_internal_silence_s,
                spoken_cps=pace.spoken_cps,
                clipping_fraction=inputs.signal.raw_clipping_fraction,
                articulation_cps=pace.articulation_cps,
                expected_articulation_cps=pace.expected_cps,
                pause_s=pace.pause_s,
            ),
            thresholds=QaThresholds(spk_warn=speaker.warn_below, spk_fail=speaker.fail_below, pace_tol=pace.tol),
            flags=flags,
        )

    def suggest(self, takes: Sequence[ScoredTake]) -> tuple[str | None, Suggestion | None]:
        """Section 8's tiers, lowest attempt within the first tier that has a take."""
        return _suggest.suggest(takes)

    def consistency(
        self, suggested: Sequence[tuple[str, tuple[float, ...]]], measurement: MeasurementRecord | None
    ) -> tuple[Consistency, tuple[Flag, ...]]:
        """Similarity of each suggested take to their centroid; outliers get ``SPK_OUTLIER`` (info).

        The baseline is the measurement's ``consistency_p5`` minus ``sim_warn_margin`` (the design's 0.01).
        """
        return _consistency.consistency(suggested, measurement, self._config.sim_warn_margin)

    def listen_first(self, segments: Sequence[SegmentResult]) -> tuple[ListenFirstItem, ...]:
        """Section 11.1's listen-first order over the job's segments."""
        return _listen_first.listen_first(segments)

    def report_md(self, results: Mapping[str, Any]) -> str:
        """``report.md`` from the assembled ``get_results`` object."""
        return _report.report_md(results)

    def report_json(self, results: Mapping[str, Any]) -> dict[str, Any]:
        """The same report as data (``names.REPORT_SCHEMA``)."""
        return _report.report_json(results)
