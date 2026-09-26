"""The suggested take of a segment (design section 8, V2). Advice only: the caller chooses.

Tiers, the first that has a take wins, lowest attempt within it:

1. verdict ``pass`` (a pass has every cue placed);
2. verdict ``warn``, every cue placed;
3. verdict ``warn``, some cue unplaced (``CUE_UNALIGNED``);
4. verdict ``fail`` (the job's outcome is then ``needs_attention``): every cue placed first, then the fewest
   fail flags.

The suggestion depends only on the takes it is given, and changes no verdict.
"""

from __future__ import annotations

from collections.abc import Sequence

from narration.contracts import codes
from narration.contracts.interfaces import ScoredTake
from narration.contracts.models import Suggestion
from narration.contracts.names import SuggestionTier

__all__ = ["cues_placed", "suggest", "tier_of"]

_UNPLACED_CODES = frozenset({codes.CUE_UNALIGNED, codes.ALIGNMENT_ERROR})

_REASONS: dict[SuggestionTier, str] = {
    1: "lowest passing attempt",
    2: "no take passed; lowest warned attempt with every cue placed",
    3: "no take passed or warned with every cue placed; lowest warned attempt, with a cue left unplaced",
    4: "every take failed QA; the one with every cue placed and the fewest fail flags, lowest attempt first",
}


def cues_placed(take: ScoredTake) -> bool:
    """Whether every cue of the take has times: no null time and no ``CUE_UNALIGNED``/``ALIGNMENT_ERROR`` flag."""
    if any(flag.code in _UNPLACED_CODES for flag in take.flags):
        return False
    return all(cue.start_s is not None and cue.end_s is not None for cue in take.cues)


def tier_of(take: ScoredTake) -> SuggestionTier:
    """The take's tier (1–4) under section 8's rule."""
    if take.verdict == "fail":
        return 4
    placed = cues_placed(take)
    if take.verdict == "pass" and placed:
        return 1
    return 2 if placed else 3


def _fail_flags(take: ScoredTake) -> int:
    return sum(1 for flag in take.flags if flag.severity in ("fail", "error"))


def suggest(takes: Sequence[ScoredTake]) -> tuple[str | None, Suggestion | None]:
    """The suggested take id and why, or (None, None) when there is no take."""
    if not takes:
        return None, None
    tier: SuggestionTier = min((tier_of(t) for t in takes), key=int)
    candidates = [t for t in takes if tier_of(t) == tier]
    if tier == 4:
        best = min(candidates, key=lambda t: (not cues_placed(t), _fail_flags(t), t.attempt, t.take_id))
    else:
        best = min(candidates, key=lambda t: (t.attempt, t.take_id))
    return best.take_id, Suggestion(tier=tier, reason=_REASONS[tier])
