"""What the transcript says against what was to be spoken (design sections 11.1 steps 4–7 and 11.3).

Pure functions over a segment's spoken text, the hints used in it, the voice's transcript and Whisper's
transcript; no model, no audio. ``match_text`` returns every number and flag the text checks produce.

Both sides are read by the number reader (``names.NUMBER_READER``, phrase by phrase) with every apostrophe
removed first, so "guide's" and "guides" (which no listener can tell apart) read alike.

* **``wer_raw``**: Whisper's normaliser alone on both sides, then the word error rate. For comparison only.
* **``wer_adj``**, the gate: each occurrence of a hinted term collapsed to one token on both sides, then the
  word error rate with the word-count rule. On the transcript side, the term's **window** is what was heard
  in its place: the words between the aligned words on either side of the term that best match it. Both
  the term and the window are read by the reader and compared on their letters and digits, so "Seven
  Sisters" matches "7 sisters". Every word of a window, inside it or at its edges, must raise its fuzzy
  ratio against the term (taking the word out must lower it). An edge word must raise it by at least
  ``_MIN_GAIN`` unless the window already matches the term without it. So a word that does not sound like
  part of the term (the next word, a filler, a hallucinated "thank you", a word inserted inside a name) is
  never taken in, and no such word can lift a window to a match.

  - A term **found** there (fuzzy ratio at least 0.75, or equal to one of its ``asr_aliases``) becomes the
    term's own token and costs nothing.
  - A term **not found** has its best-matching window collapsed to one token paired with the term's:
    exactly one substitution, never more. Other words in the gap stay as they are and count as errors of
    their own. The window never reaches past the aligned words on either side. At the segment's start it
    must end at the first aligned word after the term, and at the end begin right after the last aligned
    word before it, so a head or end insertion is never taken for the term.
  - A term with **nothing heard** in its place costs one deletion.

  An alias is recognised only inside its term's window; the same words elsewhere are ordinary words.
* **Exact spans** (section 11.3): the spoken text is read in pieces cut at the span edges, so each span has
  its own normalised words. The transcript is aligned to it word by word, and what it has at the span's
  words, plus anything inserted between them, is what was heard there. A term inside a span heard as one of
  its aliases counts as the term; a merely similar spelling does not.
* **Terms**: one result per occurrence of a hinted term, where the text pipeline applied it
  (``hints_applied``), or, when a cue has no record, where section 9.1's rule finds its canonical form.
* **Insertions**: transcript words before the first word aligned to the spoken text (head) and after the
  last one (end). Head words that match the voice's transcript are reference bleed.

The results hold nothing a request adds beyond the analysis key (``QaScorer.score``): flags carry no
segment id, and each exact result is placed by its word range, with ``start``/``end`` the span's code points
in the cue's spoken text, which the job assembler replaces with the request's own offsets.

The edit alignment is jiwer's word-level Levenshtein alignment, as the bake-off used.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import jiwer
from rapidfuzz import fuzz

from narration.contracts import codes
from narration.contracts.models import ExactResult, Flag, Hint, SegmentText, TermResult
from narration.text import canonical_form, words
from narration.text.canonical import is_punctuation
from narration.text.hints import PreparedHint, apply_hints, matching_order

from ._flags import make_flag
from .normaliser import NumberReader, letters_and_digits, letters_only, remove_apostrophes
from .profile import QaProfile

__all__ = [
    "Chunk",
    "SpanOccurrence",
    "TermOccurrence",
    "TextMatch",
    "align",
    "find_spans",
    "find_terms",
    "match_text",
]

_EXTEND = 3
"""How many transcript words beyond its own a term's window may take in (a name heard as several words)."""

_MIN_GAIN = 0.05
"""ASSUME (to be measured in WP40): how much a word at the edge of a term's window must raise the window's
fuzzy ratio against the term, unless the window already matches the term without it (then any gain will do,
as it will for a word inside a window). A word that does not sound like part of the term (a filler, the next
word, an insertion) is never taken in."""

Reader = Callable[[str], list[str]]


# ======================================================================== locating terms and spans


@dataclass(frozen=True, slots=True)
class TermOccurrence:
    """One occurrence of a hinted term in a cue.

    ``term`` is the canonical term; ``hint`` the request's hint for it (its ``asr_aliases``). ``start``/``end``
    are code points in the segment's spoken text (the join); ``end`` includes a possessive ``'s`` when there is
    one (``possessive``), which the term check sets aside.
    """

    cue: int
    term: str
    hint: Hint
    start: int
    end: int
    possessive: bool


@dataclass(frozen=True, slots=True)
class SpanOccurrence:
    """One exact span (section 11.3): its cue and its word range.

    ``start``/``end`` are the code points of its words in the cue's spoken text, punctuation around them left
    out (what ``ExactResult`` reports). ``char_start``/``char_end`` are its whole tokens in the segment's spoken
    text, punctuation included, which is what is read: "38%" is one word whose core is "38", and the "%" is
    part of the number."""

    cue: int
    words: tuple[int, int]
    start: int
    end: int
    char_start: int
    char_end: int


def _stands_alone_after(text: str, j: int) -> bool:
    return j == len(text) or text[j] == " " or is_punctuation(text[j])


def find_terms(segment: SegmentText, hints: Sequence[Hint]) -> tuple[TermOccurrence, ...]:
    """Every occurrence of every hinted term, in order.

    A cue's occurrences are those the text pipeline recorded (``CueText.hints_applied``). A cue with no record
    is searched by the pipeline's own rule (section 9.1: case-sensitive, longest first, whole words) for each
    hint's canonical term. A term with neither letters nor digits is ignored. Raises ValueError when a
    recorded offset does not hold its term, which means the segment was not built by the text pipeline.
    """
    by_term: dict[str, Hint] = {}
    for hint in hints:
        term = canonical_form(hint.term)
        if letters_and_digits(term):
            by_term.setdefault(term, hint)
    ordered = matching_order([PreparedHint(i, term, None) for i, term in enumerate(by_term)])
    found: list[TermOccurrence] = []
    for cue in segment.cues:
        text, base = cue.spoken, cue.spoken_span[0]
        applied = cue.hints_applied or apply_hints(text, ordered).applied
        for app in applied:
            hint = by_term.get(app.term)
            if hint is None:
                continue
            i, j = app.offset, app.offset + len(app.term)
            if text[i:j] != app.term:
                raise ValueError(
                    f"segment {segment.segment_id!r} cue {cue.index}: hints_applied puts {app.term!r} at {i}, "
                    f"where the spoken text has {text[i:j]!r}"
                )
            possessive = text[j : j + 2] in ("'s", "’s") and _stands_alone_after(text, j + 2)
            end = j + 2 if possessive else j
            found.append(
                TermOccurrence(
                    cue=cue.index, term=app.term, hint=hint, start=base + i, end=base + end, possessive=possessive
                )
            )
    return tuple(sorted(found, key=lambda t: t.start))


def find_spans(segment: SegmentText) -> tuple[SpanOccurrence, ...]:
    """Every exact span of the segment, from each cue's word ranges (``ExactWords.words``), with words counted
    as ``narration.text.words`` counts them.

    Raises ValueError when a word range does not fit its cue's spoken text, which means the segment was not
    built by the text pipeline's rules.
    """
    spans: list[SpanOccurrence] = []
    for cue in segment.cues:
        cue_words = words(cue.spoken)
        base = cue.spoken_span[0]
        for exact in cue.exact:
            first, stop = exact.words
            if not 0 <= first < stop <= len(cue_words):
                raise ValueError(
                    f"segment {segment.segment_id!r} cue {cue.index}: exact span words {exact.words} do not fit "
                    f"the cue's {len(cue_words)} words"
                )
            spans.append(
                SpanOccurrence(
                    cue=cue.index,
                    words=(first, stop),
                    start=cue_words[first].core_start,
                    end=cue_words[stop - 1].core_end,
                    char_start=base + cue_words[first].start,
                    char_end=base + cue_words[stop - 1].end,
                )
            )
    return tuple(spans)


# ======================================================================== tokens and alignment


@dataclass(frozen=True, slots=True)
class Reference:
    """The spoken text read in pieces cut at the given items' edges: ``ranges[k]`` is item k's token range."""

    tokens: tuple[str, ...]
    ranges: tuple[tuple[int, int], ...]


def build_reference(text: str, items: Sequence[tuple[int, int]], read: Reader) -> Reference:
    """Read ``text`` piece by piece, cutting at every item edge, so each item's words are its own.

    Reading in pieces keeps a number inside an item from merging with words outside it, and tells exactly
    which normalised words belong to the item.
    """
    cuts = sorted({0, len(text), *(a for a, _ in items), *(b for _, b in items)})
    tokens: list[str] = []
    at: dict[int, int] = {}
    for a, b in pairwise(cuts):
        at[a] = len(tokens)
        tokens.extend(read(text[a:b]))
    at[cuts[-1]] = len(tokens)
    return Reference(tokens=tuple(tokens), ranges=tuple((at[a], at[b]) for a, b in items))


@dataclass(frozen=True, slots=True)
class Chunk:
    """One block of the word alignment: ``kind`` is equal, substitute, delete or insert; ranges are [start, end)."""

    kind: str
    ref: tuple[int, int]
    hyp: tuple[int, int]


def align(ref: Sequence[str], hyp: Sequence[str]) -> tuple[Chunk, ...]:
    """The word-level edit alignment of ``hyp`` to ``ref`` (jiwer's Levenshtein alignment).

    Tokens must contain no whitespace. Either side may be empty.
    """
    if not ref and not hyp:
        return ()
    if not ref:
        return (Chunk("insert", (0, 0), (0, len(hyp))),)
    if not hyp:
        return (Chunk("delete", (0, len(ref)), (0, 0)),)
    out = jiwer.process_words(" ".join(ref), " ".join(hyp))
    return tuple(
        Chunk(c.type, (c.ref_start_idx, c.ref_end_idx), (c.hyp_start_idx, c.hyp_end_idx)) for c in out.alignments[0]
    )


def _ref_to_hyp(chunks: Sequence[Chunk]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for c in chunks:
        if c.kind in ("equal", "substitute"):
            for k in range(min(c.ref[1] - c.ref[0], c.hyp[1] - c.hyp[0])):
                mapping[c.ref[0] + k] = c.hyp[0] + k
    return mapping


def _errors(chunks: Sequence[Chunk]) -> int:
    total = 0
    for c in chunks:
        if c.kind in ("substitute", "delete"):
            total += c.ref[1] - c.ref[0]
        elif c.kind == "insert":
            total += c.hyp[1] - c.hyp[0]
    return total


def _rate(chunks: Sequence[Chunk], n_ref: int) -> float | None:
    return _errors(chunks) / n_ref if n_ref else None


# ======================================================================== where a term was heard


@dataclass(frozen=True, slots=True)
class _Heard:
    window: tuple[int, int] | None  # [start, end) in the transcript's tokens; None when nothing was heard
    ratio: float
    ok: bool
    alias: bool  # found because the window reads as one of the term's aliases
    text: str | None
    extras: int = 0  # words inside the window that are not part of the term: one word error each


_NOTHING = _Heard(window=None, ratio=0.0, ok=False, alias=False, text=None)


@dataclass(frozen=True, slots=True)
class _Judged:
    ratio: float  # against the term, without the extras
    alias: bool
    extras: int


@dataclass(frozen=True, slots=True)
class _TermKeys:
    term: str  # the term read by the reader, letters and digits only
    aliases: frozenset[str]
    width: int  # the most transcript words a window may have


def _term_keys(occ: TermOccurrence, n_words: int, read: Reader) -> _TermKeys:
    aliases = [read(a) for a in occ.hint.asr_aliases]
    keys = frozenset(filter(None, (letters_and_digits("".join(a)) for a in aliases)))
    longest = max((len(a) for a in aliases), default=0)
    return _TermKeys(letters_and_digits("".join(read(occ.term))), keys, max(n_words, longest) + _EXTEND)


def _term_heard(
    occ: TermOccurrence,
    ref_range: tuple[int, int],
    ref2hyp: dict[int, int],
    n_ref: int,
    hyp: Sequence[str],
    floor: int,
    ceiling: int,
    keys: _TermKeys,
    profile: QaProfile,
) -> _Heard:
    """Where the term was heard: the best found window if there is one, else the best window allowed for a term
    not found, else nothing (see the module docstring).

    A word inside a window that does not raise its match with the term (taking it out raises the ratio) is an
    **extra**: the window keeps it, so the name's other pieces stay together, but each extra costs one word
    error of its own ("oss of vine" is the name plus one inserted "of"). The window is judged on its other
    words. A word at either edge must raise the match, by at least ``_MIN_GAIN`` unless the window without it
    already matches the term. So no word is hidden that does not sound like part of the term, and none can lift
    a window to a match ("bell moran" is 0.737; "thank" would lift it to 0.75).

    Between two aligned words, a window may lie anywhere in the gap. A found window over the words the
    alignment put on the term is preferred; for a term not found the best match wins, and covering those words
    only breaks a tie. At the segment's start or end, where the alignment's choice of those words is
    arbitrary, the best match wins; and a term not found must end at the first aligned word after it (at the
    start) or begin right after the last aligned word before it (at the end), so a head or end insertion is
    never taken for the term.
    """
    a, b = ref_range
    core = sorted(ref2hyp[k] for k in range(a, b) if k in ref2hyp)
    before = next((ref2hyp[k] for k in range(a - 1, -1, -1) if k in ref2hyp), None)
    after = next((ref2hyp[k] for k in range(b, n_ref) if k in ref2hyp), None)
    lo = max(before + 1 if before is not None else 0, floor)
    hi = min(after if after is not None else len(hyp), ceiling)
    c0, c1 = (core[0], core[-1] + 1) if core else (lo, lo)
    inside = before is not None and after is not None
    ratios: dict[str, tuple[float, bool]] = {}
    judged: dict[tuple[int, int], _Judged | None] = {}

    def match(words: Sequence[str]) -> tuple[float, bool]:
        """The words' best ratio against the term, and whether they read as one of the term's aliases."""
        heard = letters_and_digits("".join(words))
        if heard not in ratios:
            variants = {heard}
            if occ.possessive and heard.endswith("s") and len(heard) > 1:
                variants.add(heard[:-1])
            ratio = max(fuzz.ratio(keys.term, v) / 100.0 for v in variants)
            ratios[heard] = (ratio, bool(variants & keys.aliases))
        return ratios[heard]

    def judge(w: tuple[int, int]) -> _Judged | None:
        """The window's ratio (without its extras), whether it reads as an alias, and how many extras it has;
        None when an edge word does not belong in it."""
        if w in judged:
            return judged[w]
        s, e = w
        ratio, alias = match(hyp[s:e])
        result: _Judged | None = _Judged(ratio, alias, 0)
        if not alias and e - s > 1:
            extras = {i for i in range(s + 1, e - 1) if ratio - match([*hyp[s:i], *hyp[i + 1 : e]])[0] <= 0}
            kept = [hyp[i] for i in range(s, e) if i not in extras]
            ratio, alias = match(kept)
            result = _Judged(ratio, alias, len(extras))
            for without in (kept[1:], kept[:-1]):
                rest, rest_alias = match(without)
                gain = ratio - rest
                # An edge word added to a window that already matches without it may gain anything; one that
                # would lift a window to a match, or extend one that does not match, must gain _MIN_GAIN.
                need = 0.0 if rest_alias or rest >= profile.term_min_ratio else _MIN_GAIN
                if gain <= 0 or gain < need:
                    result = None
                    break
        judged[w] = result
        return result

    def windows(left: int, right: int) -> list[tuple[tuple[int, int], _Judged]]:
        spans = [(s, e) for s in range(left, right) for e in range(s + 1, min(right, s + keys.width) + 1)]
        return [(w, j) for w in spans if (j := judge(w)) is not None]

    def covers(w: tuple[int, int]) -> bool:
        return inside and bool(core) and w[0] <= c0 and c1 <= w[1]

    def shape(w: tuple[int, int], j: _Judged) -> tuple[int, int, int]:
        return (-j.extras, -(w[1] - w[0]), -w[0])  # then the fewest extras, the fewest words, the earliest

    def heard(w: tuple[int, int], j: _Judged, ok: bool) -> _Heard:
        return _Heard(
            window=w, ratio=j.ratio, ok=ok, alias=ok and j.alias, text=" ".join(hyp[w[0] : w[1]]), extras=j.extras
        )

    # Found: a split name often lands a word or two into the head or the tail, so at an edge the search
    # reaches past the words the alignment put on the term.
    found_lo = lo if before is not None else max(lo, c0 - _EXTEND)
    found_hi = hi if after is not None else min(hi, c1 + _EXTEND)
    hits = [(w, j) for w, j in windows(found_lo, found_hi) if j.alias or j.ratio >= profile.term_min_ratio]
    if hits:
        w, j = max(hits, key=lambda x: (covers(x[0]), x[1].ratio, *shape(*x)))
        return heard(w, j, ok=True)

    # Not found: the best match; covering the aligned words only breaks a tie.
    candidates = windows(lo, hi)
    if before is None and after is not None:
        candidates = [(w, j) for w, j in candidates if w[1] == hi]
    elif after is None and before is not None:
        candidates = [(w, j) for w, j in candidates if w[0] == lo]
    if not candidates:
        return _NOTHING
    w, j = max(candidates, key=lambda x: (x[1].ratio, covers(x[0]), *shape(*x)))
    return heard(w, j, ok=False)


def _placeholder(k: int) -> str:
    # U+27E6/U+27E7 are symbols the reader deletes, so no transcript word can equal a placeholder.
    return f"\u27e6t{k}\u27e7"


# ======================================================================== the checks


@dataclass(frozen=True, slots=True)
class TextMatch:
    """Everything the text checks found for one take."""

    wer_raw: float | None
    wer_adj: float | None
    word_errors: int
    ref_words: int
    exact: tuple[ExactResult, ...]
    terms: tuple[TermResult, ...]
    head_count: int
    end_count: int
    head_words: tuple[str, ...]
    end_words: tuple[str, ...]
    bleed: bool
    bleed_ratio: float | None
    flags: tuple[Flag, ...] = field(default=())

    @property
    def exact_ok(self) -> bool:
        return all(e.match == "same" for e in self.exact)


def _reader(reader: NumberReader) -> Reader:
    return lambda s: reader.read(remove_apostrophes(s)).split()


def _bleed(head_words: Sequence[str], voice_letters: str, profile: QaProfile) -> float | None:
    """The best fuzzy ratio of the head's letters against as many letters at the end, or the start, of the
    voice's transcript; None when the head is too short to tell bleed from chance (``QaProfile``)."""
    head = letters_only("".join(head_words))
    if not voice_letters or not head:
        return None
    if len(head_words) < profile.bleed_min_words and len(head) < profile.bleed_min_letters:
        return None
    n = len(head)
    return max(fuzz.ratio(head, voice_letters[-n:]), fuzz.ratio(head, voice_letters[:n])) / 100.0


def _word_error_flags(wer_adj: float, word_errors: int, n_ref: int, profile: QaProfile) -> list[Flag]:
    details = {
        "wer_adj": round(wer_adj, 6),
        "word_errors": word_errors,
        "ref_words": n_ref,
        "warn_above": profile.wer_warn_above,
        "warn_min_errors": profile.wer_warn_min_errors,
        "fail_above": profile.wer_fail_above,
        "fail_min_errors": profile.wer_fail_min_errors,
    }
    if wer_adj > profile.wer_fail_above and word_errors >= profile.wer_fail_min_errors:
        return [
            make_flag(
                codes.WER_HIGH,
                "fail",
                f"wer_adj {wer_adj:.3f} with {word_errors} word errors in {n_ref} words "
                f"(fail above {profile.wer_fail_above} with at least {profile.wer_fail_min_errors})",
                details=details,
            )
        ]
    if wer_adj > profile.wer_warn_above and word_errors >= profile.wer_warn_min_errors:
        return [
            make_flag(
                codes.WER_HIGH,
                "warn",
                f"wer_adj {wer_adj:.3f} with {word_errors} word error(s) in {n_ref} words "
                f"(warn above {profile.wer_warn_above})",
                details=details,
            )
        ]
    return []


def match_text(
    segment: SegmentText,
    hints: Sequence[Hint],
    transcript: str,
    voice_transcript: str,
    reader: NumberReader,
    profile: QaProfile,
) -> TextMatch:
    """Run the text checks of sections 11.1 (steps 4–7) and 11.3 on one take's transcript."""
    spoken = segment.spoken_text
    read = _reader(reader)
    flags: list[Flag] = []

    # ---- wer_raw: Whisper's normaliser alone, for comparison with the bake-off.
    raw_ref = reader.whisper(spoken).split()
    raw_hyp = reader.whisper(transcript).split()
    wer_raw = _rate(align(raw_ref, raw_hyp), len(raw_ref))

    # ---- where each term was heard
    terms = find_terms(segment, hints)
    ref = build_reference(spoken, [(t.start, t.end) for t in terms], read)
    hyp = tuple(read(transcript))
    ref2hyp = _ref_to_hyp(align(ref.tokens, hyp))
    live = [(t, r) for t, r in zip(terms, ref.ranges, strict=True) if r[0] < r[1]]

    heard: list[_Heard] = []
    floor = 0
    for k, (occ, rng) in enumerate(live):
        nxt = [ref2hyp[i] for _, n_rng in live[k + 1 :] for i in range(*n_rng) if i in ref2hyp][:1]
        h = _term_heard(
            occ,
            rng,
            ref2hyp,
            len(ref.tokens),
            hyp,
            floor,
            nxt[0] if nxt else len(hyp),
            _term_keys(occ, rng[1] - rng[0], read),
            profile,
        )
        heard.append(h)
        if h.window:
            floor = h.window[1]

    # ---- wer_adj: each term collapsed to one token on both sides
    ref2: list[str] = []
    at = 0
    for k, (_, (a, b)) in enumerate(live):
        ref2.extend(ref.tokens[at:a])
        ref2.append(_placeholder(k))
        at = b
    ref2.extend(ref.tokens[at:])

    # A term's window takes the term's own token on both sides, found or not, so the alignment always pairs
    # them (it would otherwise be free to pair the term with a word of an insertion). A term not found then
    # adds its one substitution below.
    windows = sorted((h.window, k, h) for k, h in enumerate(heard) if h.window)
    hyp2: list[str] = []
    shown2: list[str] = []
    at = 0
    for (s, e), k, h in windows:
        hyp2.extend(hyp[at:s])
        shown2.extend(hyp[at:s])
        hyp2.append(_placeholder(k))
        shown2.append(h.text or "")
        at = e
    hyp2.extend(hyp[at:])
    shown2.extend(hyp[at:])

    chunks2 = align(ref2, hyp2)
    paired = {ref2[i] for c in chunks2 if c.kind == "equal" for i in range(*c.ref)}
    not_found = sum(1 for _, k, h in windows if not h.ok and _placeholder(k) in paired)
    word_errors = _errors(chunks2) + not_found + sum(h.extras for _, _, h in windows)
    wer_adj = word_errors / len(ref2) if ref2 else None
    if wer_adj is not None:
        flags.extend(_word_error_flags(wer_adj, word_errors, len(ref2), profile))

    # ---- terms
    term_results = tuple(
        TermResult(term=occ.term, cue=occ.cue, heard=h.text, ok=h.ok) for (occ, _), h in zip(live, heard, strict=True)
    )
    for (occ, _), h in zip(live, heard, strict=True):
        if not h.ok:
            flags.append(
                make_flag(
                    codes.TERM_UNVERIFIED,
                    "warn",
                    f"cue {occ.cue}: the transcript has {h.text!r} where {occ.term!r} was to be spoken; "
                    "listen to it, or add the transcript's spelling as an asr_alias if it sounds right"
                    if h.text
                    else f"cue {occ.cue}: the transcript has nothing where {occ.term!r} was to be spoken; listen to it",
                    cue=occ.cue,
                    details={
                        "term": occ.term,
                        "heard": h.text,
                        "ratio": round(h.ratio, 4),
                        "min_ratio": profile.term_min_ratio,
                    },
                )
            )

    # ---- insertions, from the collapsed alignment
    aligned = [c for c in chunks2 if c.kind in ("equal", "substitute")]
    if aligned:
        n_head = aligned[0].hyp[0]
        n_end = len(hyp2) - aligned[-1].hyp[1]
        head_words = tuple(shown2[:n_head])
        end_words = tuple(shown2[len(hyp2) - n_end :])
    else:
        head_words, end_words, n_head, n_end = (), (), 0, 0
    bleed_ratio = _bleed(head_words, letters_only("".join(read(voice_transcript))), profile) if n_head else None
    bleed = bleed_ratio is not None and bleed_ratio >= profile.bleed_min_ratio
    if n_head >= profile.insertion_warn_words:
        severity = "fail" if bleed or n_head >= profile.insertion_fail_words else "warn"
        what = "reference bleed: they match the voice's transcript" if bleed else "words before the first cue"
        flags.append(
            make_flag(
                codes.HEAD_INSERTION,
                severity,
                f"{n_head} extra word(s) at the head, {' '.join(head_words)!r} ({what})",
                cue=0,
                details={
                    "words": n_head,
                    "text": " ".join(head_words),
                    "bleed": bleed,
                    "bleed_ratio": None if bleed_ratio is None else round(bleed_ratio, 4),
                    "warn_at": profile.insertion_warn_words,
                    "fail_at": profile.insertion_fail_words,
                },
            )
        )
    if n_end >= profile.insertion_warn_words:
        severity = "fail" if n_end >= profile.insertion_fail_words else "warn"
        flags.append(
            make_flag(
                codes.END_INSERTION,
                severity,
                f"{n_end} extra word(s) after the last cue, {' '.join(end_words)!r}",
                cue=segment.cues[-1].index if segment.cues else None,
                details={
                    "words": n_end,
                    "text": " ".join(end_words),
                    "warn_at": profile.insertion_warn_words,
                    "fail_at": profile.insertion_fail_words,
                },
            )
        )

    # ---- exact spans (section 11.3): the reference cut at the span edges; a term heard as an alias is the term
    exact_results: list[ExactResult] = []
    spans = find_spans(segment)
    if spans:
        ex_ref = build_reference(spoken, [(s.char_start, s.char_end) for s in spans], read)
        ex_hyp: list[str] = []
        at = 0
        for (s, e), k, h in windows:
            if h.alias:
                a, b = live[k][1]
                ex_hyp.extend(hyp[at:s])
                ex_hyp.extend(ref.tokens[a:b])
                at = e
        ex_hyp.extend(hyp[at:])
        ex_map = _ref_to_hyp(align(ex_ref.tokens, ex_hyp))
        for span, (a, b) in zip(spans, ex_ref.ranges, strict=True):
            expected = " ".join(ex_ref.tokens[a:b])
            positions = [ex_map[k] for k in range(a, b) if k in ex_map]
            got: tuple[str, ...] | None = tuple(ex_hyp[min(positions) : max(positions) + 1]) if positions else None
            if a == b:
                match, heard_text = "same", ""
            elif got is None:
                match, heard_text = "missing", None
            else:
                match, heard_text = ("same" if got == ex_ref.tokens[a:b] else "different"), " ".join(got)
            result = ExactResult(
                cue=span.cue,
                start=span.start,
                end=span.end,
                words=span.words,
                expected=expected,
                heard=heard_text,
                match=match,
            )
            exact_results.append(result)
            if result.match != "same":
                flags.append(
                    make_flag(
                        codes.EXACT_SPAN_MISMATCH,
                        "fail",
                        f"cue {span.cue}, words {span.words[0]}–{span.words[1] - 1}: expected {result.expected!r}, "
                        "heard " + (repr(result.heard) if result.heard is not None else "nothing"),
                        cue=span.cue,
                        details={
                            "cue": span.cue,
                            "words": list(span.words),
                            "expected": result.expected,
                            "heard": result.heard,
                            "match": result.match,
                        },
                    )
                )

    return TextMatch(
        wer_raw=wer_raw,
        wer_adj=wer_adj,
        word_errors=word_errors,
        ref_words=len(ref2),
        exact=tuple(exact_results),
        terms=term_results,
        head_count=n_head,
        end_count=n_end,
        head_words=head_words,
        end_words=end_words,
        bleed=bleed,
        bleed_ratio=bleed_ratio,
        flags=tuple(flags),
    )
