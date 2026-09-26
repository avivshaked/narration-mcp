"""The service's own material under ``material/`` (plan.md WP18).

Design sections: 3.2 (calibration corpus and length ladder), 3.5 (positive-only voice descriptions), 7.2
(canonical form, join, spoken length, exact spans), 9.1 and 9.3 (text checks and their fixtures), 10.1 (the
canary), 11.2 (the alignment benchmark), 15 (material ships versioned and hashed), 16 (defaults), and 20
(the Phase 3 QA fixtures and the Phase 4 demo script).

The text rules in the "Local text rules" section below are a small stand-in for the text pipeline and the
lint (WP10: ``narration.text`` and ``narration.lint``), which are not built yet. Once they are, these tests
should call them instead, so that the service's own material is judged by the same code as a caller's text.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
import tomllib
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
MATERIAL = ROOT / "material"

CALIBRATION = ("calibration", "narration-en.v1")
ALIGNMENT = ("alignment", "alignment-en.v1")
CANARY = ("canary", "canary.v1")
DEMO = ("demo", "demo-en.v1")
TEXT_FIXTURES = ("fixtures", "text-v1")
QA_FIXTURES = ("fixtures", "qa-faults-v1")
EXPECTED_SETS = {CALIBRATION, ALIGNMENT, CANARY, DEMO, TEXT_FIXTURES, QA_FIXTURES}

SEGMENT_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")  # the Id fragment, design 7.2
MAX_CUE_CHARS = 600  # the Segment fragment's cue maxLength, design 7.2
MAX_DESIGN_TEXT_CHARS = 400  # design_voice's design_text limit, design 7.6
SENTENCE_END = re.compile(r"[.!?…][\"”’')]*$")

# The flag codes of design section 14. Copied here until WP01 publishes them in narration.contracts; then
# take them from there (AGENTS.md section 6: names are never retyped).
FLAG_CODES = frozenset(
    {
        "WRITTEN_FORM_TOKEN",
        "TERM_SPLIT_ACROSS_CUES",
        "SEGMENT_TOO_LONG",
        "WER_HIGH",
        "EXACT_SPAN_MISMATCH",
        "TERM_UNVERIFIED",
        "SPK_SIM_LOW",
        "SPK_OUTLIER",
        "PACE_FAST",
        "PACE_SLOW",
        "HEAD_INSERTION",
        "END_INSERTION",
        "SILENCE_LONG",
        "CLIPPING",
        "TOKEN_CAP_HIT",
        "CUE_UNALIGNED",
        "CUE_LOW_CONFIDENCE",
        "CUE_ALIGNMENT_DISAGREE",
        "CUE_BOUNDARY_NO_PAUSE",
        "ALIGNMENT_ERROR",
        "FIT_TIGHT",
        "OVER_SCENE",
        "CANARY_MISMATCH",
        "LOUDNESS_UNDER_TARGET",
        "GAIN_HIGH",
        "RETAKEN",
    }
)
SEVERITIES = frozenset({"info", "warn", "fail", "error"})


# ============================================================================ loading


def set_dir(key: tuple[str, str]) -> Path:
    kind, set_id = key
    return MATERIAL / kind / set_id


def read_json(path: Path) -> Any:
    return json.loads(path.read_bytes().decode("utf-8"))


@functools.cache
def load(key: tuple[str, str], name: str) -> Any:
    """A material file, or an empty dict if it is missing or broken (the manifest tests report that)."""
    try:
        return read_json(set_dir(key) / name)
    except (OSError, ValueError):
        return {}


@functools.cache
def design_config() -> dict[str, Any]:
    """The TOML configuration block of design section 16, parsed."""
    text = (ROOT / "docs" / "design.md").read_text(encoding="utf-8")
    section = text[text.index("## 16. Configuration") :]
    start = section.index("```toml") + len("```toml")
    return tomllib.loads(section[start : section.index("```", start)])


def calibration() -> list[dict[str, Any]]:
    return load(CALIBRATION, "paragraphs.json").get("calibration", [])


def ladder() -> list[dict[str, Any]]:
    return load(CALIBRATION, "paragraphs.json").get("ladder", [])


def alignment() -> list[dict[str, Any]]:
    return load(ALIGNMENT, "paragraphs.json").get("paragraphs", [])


def demo() -> list[dict[str, Any]]:
    return load(DEMO, "paragraphs.json").get("paragraphs", [])


def qa_specs() -> list[dict[str, Any]]:
    return load(QA_FIXTURES, "specs.json").get("specs", [])


def text_cases() -> list[dict[str, Any]]:
    return load(TEXT_FIXTURES, "cases.json").get("cases", [])


def lint_cases() -> list[dict[str, Any]]:
    return load(TEXT_FIXTURES, "lint.json").get("cases", [])


def spoken_segments() -> list[tuple[str, dict[str, Any]]]:
    """Every segment in a spoken-form set, as (id, segment); the text fixtures are the only exception."""
    out = [(f"narration-en.v1/{p['segment_id']}", p) for p in calibration() + ladder()]
    out += [(f"alignment-en.v1/{p['segment_id']}", p) for p in alignment()]
    out += [(f"demo-en.v1/{p['segment_id']}", p) for p in demo()]
    out += [(f"qa-faults-v1/{s['name']}", s["segment"]) for s in qa_specs()]
    return out


def manifests() -> list[Path]:
    return sorted(MATERIAL.glob("*/*/manifest.json"))


def material_files() -> list[Path]:
    return sorted(p for p in MATERIAL.glob("*/*/*") if p.is_file())


def ids(items: list[tuple[str, Any]]) -> list[str]:
    return [name for name, _ in items]


# ============================================================================ local text rules
# A stand-in for WP10's narration.text and narration.lint, only as strict as this corpus needs.

PASSING_PUNCTUATION = frozenset(".,;:!?'’‘\"“”()-–—…")  # design 9.1: voiced as prosody
LONE_LETTERS_ALLOWED = frozenset({"a", "A", "I"})
LONE_UNIT_LETTERS = frozenset({"m", "s", "g"})
MARKUP = ("[", "]", "<|", "|>")

# The generic unit list of design 9.1, approximated: SI symbols with their prefixes, plus common
# abbreviations. WP10 owns the real list. Ordinary English words that are also unit symbols are left out,
# as the design says; only the ones this corpus uses are excluded here, so this list is the stricter one.
_SI_PREFIXES = ("", "Q", "R", "Y", "Z", "E", "P", "T", "G", "M", "k", "h", "da", "d", "c", "m", "µ", "μ", "u")
_SI_PREFIXES += ("n", "p", "f", "a", "z", "y", "r", "q")
_SI_UNITS = ("m", "g", "s", "A", "K", "mol", "cd", "Hz", "N", "Pa", "J", "W", "C", "V", "F", "Ω", "S", "Wb")
_SI_UNITS += ("T", "H", "lm", "lx", "Bq", "Gy", "Sv", "kat", "L", "l", "t", "eV", "Wh", "bar")
_COMMON_UNITS = ("km", "kg", "mph", "kph", "ft", "yd", "mi", "lb", "lbs", "oz", "st", "gal", "pt", "qt", "hr")
_COMMON_UNITS += ("hrs", "min", "mins", "sec", "secs", "dB", "rpm", "psi", "atm", "mmHg", "kcal", "cal", "ha")
_COMMON_UNITS += ("sq", "mpg", "fps", "tsp", "tbsp", "Ah", "mAh", "kWh", "ml", "mL", "cc", "in")
_ENGLISH_WORDS = frozenset({"in", "am", "as", "at"})
UNIT_SYMBOLS = frozenset(
    tok
    for tok in {p + u for p in _SI_PREFIXES for u in _SI_UNITS} | set(_COMMON_UNITS)
    if len(tok) >= 2 and tok not in _ENGLISH_WORDS
)

_ONES = ("zero", "nought", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
_TEENS = ("eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
_TENS = ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
CARDINALS = frozenset(_ONES + _TEENS + _TENS + ("hundred", "thousand", "million"))
SPAN_NUMBER_WORDS = CARDINALS | {"and", "point"}

NEGATION_WORDS = frozenset({"not", "no", "never", "without", "avoid", "don't", "dont", "nor", "neither", "none"})
_LINT_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
_LINT_NON = re.compile(r"(?<![^\W\d_])non-", re.IGNORECASE)


def canonical(text: str) -> str:
    """Canonical form of a cue (design 7.2): NFC, every run of whitespace one space, trimmed."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def join(cues: list[dict[str, Any]]) -> str:
    """The join (design 7.2): the cues' canonical texts joined by one space."""
    return " ".join(canonical(c["text"]) for c in cues)


def is_punctuation(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P")


@dataclass(frozen=True)
class Word:
    """A whitespace-separated token, and its core: the token with leading and trailing punctuation set aside."""

    start: int
    end: int
    core_start: int
    core_end: int


def words(text: str) -> list[Word]:
    """The words of a text (design 7.2), with code-point offsets into it."""
    out = []
    for m in re.finditer(r"\S+", text):
        core_start, core_end = m.start(), m.end()
        while core_start < core_end and is_punctuation(text[core_start]):
            core_start += 1
        while core_end > core_start and is_punctuation(text[core_end - 1]):
            core_end -= 1
        out.append(Word(m.start(), m.end(), core_start, core_end))
    return out


@dataclass(frozen=True)
class Finding:
    """A text-check finding (design 9.1): kind, token and its offset into the canonical cue."""

    kind: str
    token: str
    offset: int

    @property
    def severity(self) -> str:
        return "info" if self.kind == "letter" else "warn"


def text_findings(cue_text: str) -> list[Finding]:
    """The four text checks of design 9.1 on one cue: digit, symbol, unit_like, letter (at most one per token)."""
    text = canonical(cue_text)
    found = []
    for w in words(text):
        token, core = text[w.start : w.end], text[w.core_start : w.core_end]
        if any(unicodedata.category(ch).startswith("N") for ch in token):
            found.append(Finding("digit", token, w.start))
        elif any(not (ch.isalpha() or ch in PASSING_PUNCTUATION) for ch in token):
            found.append(Finding("symbol", token, w.start))
        elif core in UNIT_SYMBOLS or core in LONE_UNIT_LETTERS:
            found.append(Finding("unit_like", core, w.core_start))
        else:
            # A lone letter, including one standing alone between hyphens or dashes ("A-frame"), to be safe.
            pos = w.core_start
            for piece in re.split(r"[-–—]", core):
                if len(piece) == 1 and piece.isalpha() and piece not in LONE_LETTERS_ALLOWED:
                    found.append(Finding("letter", piece, pos))
                pos += len(piece) + 1
    return found


def refused(cue_text: str) -> list[tuple[str, int]]:
    """What design 9.1 step 1 refuses in a cue as sent: markup, and control characters that are not whitespace."""
    out = [(token, m.start()) for token in MARKUP for m in re.finditer(re.escape(token), cue_text)]
    out += [(ch, i) for i, ch in enumerate(cue_text) if unicodedata.category(ch) == "Cc" and not ch.isspace()]
    return sorted(out, key=lambda item: item[1])


def covered(cue_text: str, start: int, end: int) -> list[int]:
    """Indices of the words whose core lies inside [start, end)."""
    return [
        i
        for i, w in enumerate(words(cue_text))
        if start <= w.core_start and w.core_end <= end and w.core_end > w.core_start
    ]


def span_error(cue_text: str, spans: list[dict[str, Any]], *, strict: bool) -> str | None:
    """Why a cue's exact spans are invalid (design 7.2), or None. Offsets are into the cue as sent.

    A span lies inside the cue, covers at least one word, and starts and ends on word edges. A span may end
    before or after trailing punctuation (and, as assumed here, start before or after leading punctuation).
    Spans do not overlap. ``strict`` (for the service's own material) asks for exactly the words' cores.
    """
    ws = words(cue_text)
    starts = {w.core_start for w in ws} | (set() if strict else {w.start for w in ws})
    ends = {w.core_end for w in ws} | (set() if strict else {w.end for w in ws})
    previous_end = 0
    for span in sorted(spans, key=lambda s: s["start"]):
        start, end = span["start"], span["end"]
        if not 0 <= start < end <= len(cue_text):
            return f"span {start}-{end} is empty or outside the cue"
        if start not in starts or end not in ends:
            return f"span {start}-{end} cuts a word"
        if not covered(cue_text, start, end):
            return f"span {start}-{end} covers no word"
        if start < previous_end:
            return f"span {start}-{end} overlaps the one before"
        previous_end = end
    return None


def span_words(cue_text: str, start: int, end: int) -> list[str]:
    ws = words(cue_text)
    return [cue_text[ws[i].core_start : ws[i].core_end] for i in covered(cue_text, start, end)]


def number_parts(word: str) -> list[str]:
    return word.lower().split("-")


def negations(description: str) -> list[tuple[str, int]]:
    """Triggers of the positive-only lint (design 3.5), as (trigger, offset).

    A plain word list, whole-word and case-insensitive: not, no, never, without, avoid, don't / dont, the
    other n't forms, nor, neither, none; and the prefix non- (with its hyphen).
    """
    found = []
    for m in _LINT_WORD.finditer(description):
        word = m.group(0).lower().replace("’", "'")
        if word in NEGATION_WORDS or word.endswith("n't"):
            found.append((m.group(0), m.start()))
    found += [(m.group(0), m.start()) for m in _LINT_NON.finditer(description)]
    return sorted(found, key=lambda item: item[1])


def resolve(obj: Any, path: str) -> Any:
    """Follow a path such as ``expect.cues[0].warnings`` into a JSON value (raises if it does not exist)."""
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        obj = obj[int(part[1:-1])] if part.startswith("[") else obj[part]
    return obj


def test_local_text_rules_catch_what_the_corpus_must_avoid_s9_1() -> None:
    """The local stand-in catches each kind of written form, so a clean result on the corpus means something."""
    assert [f.kind for f in text_findings("It is 40 metres.")] == ["digit"]
    assert [f.kind for f in text_findings("Ten % more & less.")] == ["symbol", "symbol"]
    assert [f.kind for f in text_findings("Ten km, four kg and two m.")] == ["unit_like", "unit_like", "unit_like"]
    assert [f.kind for f in text_findings("Plan B and an A-frame.")] == ["letter"]
    assert text_findings("A cat and I am in a boat, as it was at dawn.") == []
    assert refused("[laughs] <|x|> \u0007") == [("[", 0), ("]", 7), ("<|", 9), ("|>", 12), ("\u0007", 15)]
    assert span_error("They counted forty-two.", [{"start": 13, "end": 18}], strict=True) == "span 13-18 cuts a word"


# ============================================================================ manifests and files (section 15)


def test_every_set_is_present_s15() -> None:
    found = {(p.parent.parent.name, p.parent.name) for p in manifests()}
    assert found == EXPECTED_SETS


@pytest.mark.parametrize("manifest", manifests(), ids=lambda p: p.parent.name)
def test_manifest_describes_its_set_s15(manifest: Path) -> None:
    m = read_json(manifest)
    assert m["set"] == manifest.parent.name
    assert m["kind"] == manifest.parent.parent.name
    assert isinstance(m["version"], int) and m["version"] >= 1
    # Draft until the owner has listened to sample renders (gate H1, plan.md section 5).
    assert m["status"] in {"draft", "frozen"}
    assert m["language"] == "en"
    assert m["description"].strip()
    assert m["licence"].startswith("PolyForm Noncommercial 1.0.0")
    assert m["files"], "a set has at least one file"


@pytest.mark.parametrize("manifest", manifests(), ids=lambda p: p.parent.name)
def test_manifest_hashes_match_the_files_s15(manifest: Path) -> None:
    m = read_json(manifest)
    listed = {entry["path"] for entry in m["files"]}
    present = {p.name for p in manifest.parent.iterdir() if p.is_file() and p.name != "manifest.json"}
    assert listed == present, "every file of a set is listed in its manifest, and every listed file exists"
    for entry in m["files"]:
        data = (manifest.parent / entry["path"]).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        assert entry["sha256"] == actual, f"{entry['path']}: the manifest says {entry['sha256']}, the file is {actual}"
        assert entry["bytes"] == len(data), f"{entry['path']}: the manifest says {entry['bytes']} bytes"


@pytest.mark.parametrize("path", material_files(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_material_files_are_utf8_json_with_lf_line_ends_s15(path: Path) -> None:
    """The hashes are over exact bytes, so the bytes must not depend on the platform (.gitattributes: eol=lf)."""
    data = path.read_bytes()
    assert path.suffix == ".json"
    assert not data.startswith(b"\xef\xbb\xbf"), "no byte-order mark"
    assert b"\r" not in data, "LF line ends only"
    assert data.endswith(b"\n")
    obj = json.loads(data.decode("utf-8"))
    assert obj["set"] == path.parent.name


# ============================================================================ spoken form (sections 7.2, 9.1)


@pytest.mark.parametrize(("name", "segment"), spoken_segments(), ids=ids(spoken_segments()))
def test_spoken_form_segment_is_well_formed_s7_2(name: str, segment: dict[str, Any]) -> None:
    assert SEGMENT_ID.match(segment["segment_id"]), "segment ids follow the Id fragment"
    cues = segment["cues"]
    limits = design_config()["limits"]
    assert 1 <= len(cues) <= limits["max_cues_per_segment"]
    assert len(join(cues)) <= limits["max_chars_per_segment"]
    for cue in cues:
        assert set(cue) <= {"text", "exact"}, "cues are sent as they are: no extra fields"
        assert 1 <= len(cue["text"]) <= MAX_CUE_CHARS
        assert cue["text"] == canonical(cue["text"]), "the service's own cues are already in canonical form"
        assert refused(cue["text"]) == []
    if "spoken_chars" in segment:
        assert segment["spoken_chars"] == len(join(cues)), "spoken_chars is the join's length in code points"


@pytest.mark.parametrize(("name", "segment"), spoken_segments(), ids=ids(spoken_segments()))
def test_spoken_form_raises_no_text_finding_s9_1(name: str, segment: dict[str, Any]) -> None:
    """No digit, no symbol outside the passing punctuation, no unit abbreviation, no lone letter but a, A, I."""
    for index, cue in enumerate(segment["cues"]):
        assert text_findings(cue["text"]) == [], f"cue {index}: {cue['text']!r}"


@pytest.mark.parametrize(("name", "segment"), spoken_segments(), ids=ids(spoken_segments()))
def test_exact_spans_are_whole_words_in_range_without_overlap_s7_2(name: str, segment: dict[str, Any]) -> None:
    for index, cue in enumerate(segment["cues"]):
        spans = cue.get("exact", [])
        assert len(spans) <= 20
        assert span_error(cue["text"], spans, strict=True) is None, f"cue {index}: {cue['text']!r}"


# ============================================================================ calibration and ladder (section 3.2)


def test_corpus_is_the_one_the_design_configures_s3_2() -> None:
    assert design_config()["measurement"]["corpus"] == CALIBRATION[1]


def test_ladder_rungs_are_the_designs_s3_2() -> None:
    targets = [p["target_spoken_chars"] for p in ladder()]
    assert targets == design_config()["measurement"]["length_ladder_spoken_chars"]


@pytest.mark.parametrize("paragraph", ladder(), ids=lambda p: p["segment_id"])
def test_ladder_paragraph_hits_its_target_within_five_per_cent_s3_2(paragraph: dict[str, Any]) -> None:
    target = paragraph["target_spoken_chars"]
    spoken = len(join(paragraph["cues"]))
    assert abs(spoken - target) <= 0.05 * target, f"{spoken} spoken characters for a target of {target}"


def test_calibration_has_three_paragraphs_of_150_to_300_spoken_characters_s3_2() -> None:
    lengths = [len(join(p["cues"])) for p in calibration()]
    assert len(lengths) == 3
    assert all(150 <= n <= 300 for n in lengths), lengths


@pytest.mark.parametrize("paragraph", calibration() + ladder(), ids=lambda p: p["segment_id"])
def test_calibration_number_words_are_marked_exact_s3_2(paragraph: dict[str, Any]) -> None:
    """Spans hold only number words, and every number word but "one" (so often not a count) is in a span."""
    for cue in paragraph["cues"]:
        text = cue["text"]
        inside: set[int] = set()
        for span in cue.get("exact", []):
            for word in span_words(text, span["start"], span["end"]):
                assert all(part in SPAN_NUMBER_WORDS for part in number_parts(word)), f"{word!r} in a span"
            inside.update(covered(text, span["start"], span["end"]))
        for i, w in enumerate(words(text)):
            parts = number_parts(text[w.core_start : w.core_end])
            if any(part in CARDINALS - {"one"} for part in parts):
                assert i in inside, f"number word {text[w.core_start : w.core_end]!r} is not marked exact"


def test_most_calibration_paragraphs_have_number_words_s3_2() -> None:
    paragraphs = calibration() + ladder()
    with_numbers = [p for p in paragraphs if any(c.get("exact") for c in p["cues"])]
    assert len(with_numbers) * 3 >= len(paragraphs) * 2


SETS_WITH_NAMES = [CALIBRATION, ALIGNMENT, DEMO]


@pytest.mark.parametrize("key", SETS_WITH_NAMES, ids=lambda k: k[1])
def test_each_set_has_invented_names_that_occur_in_it_s3_2(key: tuple[str, str]) -> None:
    content = load(key, "paragraphs.json")
    paragraphs = content.get("calibration", []) + content.get("ladder", []) + content.get("paragraphs", [])
    text = " ".join(join(p["cues"]) for p in paragraphs)
    assert content["invented_names"], "at least one invented name per set"
    for name in content["invented_names"]:
        assert re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text), f"{name!r} does not occur"


# ============================================================================ alignment benchmark (section 11.2)


def test_alignment_benchmark_is_the_configured_one_of_about_twelve_paragraphs_s11_2() -> None:
    assert design_config()["alignment"]["benchmark"] == ALIGNMENT[1]
    paragraphs = alignment()
    assert 10 <= len(paragraphs) <= 14
    cues = sum(len(p["cues"]) for p in paragraphs)
    assert 25 <= cues <= 40, "about 60 marks per render: a start and an end for each cue"


@pytest.mark.parametrize("paragraph", alignment(), ids=lambda p: p["segment_id"])
def test_alignment_boundaries_are_marked_by_the_pause_rule_s11_2(paragraph: dict[str, Any]) -> None:
    """A boundary after a sentence expects a pause; one inside a sentence, with no punctuation, does not."""
    cues = paragraph["cues"]
    boundaries = paragraph["boundaries"]
    assert [b["after_cue"] for b in boundaries] == list(range(len(cues) - 1))
    for boundary in boundaries:
        text = cues[boundary["after_cue"]]["text"]
        ends_sentence = bool(SENTENCE_END.search(text))
        assert boundary["pause"] is ends_sentence, text
        if not ends_sentence:
            assert text[-1].isalpha(), f"a no-pause cue ends in a letter, not in punctuation: {text!r}"
    assert SENTENCE_END.search(cues[-1]["text"]), "the paragraph ends a sentence"


def test_alignment_has_both_boundary_kinds_and_short_and_long_cues_s11_2() -> None:
    pauses = [b["pause"] for p in alignment() for b in p["boundaries"]]
    assert pauses.count(True) >= 6 and pauses.count(False) >= 6
    lengths = [len(words(c["text"])) for p in alignment() for c in p["cues"]]
    assert min(lengths) <= 2 and max(lengths) >= 20
    text = " ".join(join(p["cues"]) for p in alignment())
    assert sum(1 for w in re.findall(r"[\w-]+", text) if set(number_parts(w)) & CARDINALS) >= 5


def test_alignment_letters_fold_to_the_aligners_alphabet_s11_2() -> None:
    """The aligner's alphabet is A to Z with accents dropped, and the apostrophe (design 11.2 step 1)."""
    for p in alignment():
        for cue in p["cues"]:
            folded = "".join(ch for ch in unicodedata.normalize("NFD", cue["text"]) if unicodedata.category(ch) != "Mn")
            assert all("A" <= ch.upper() <= "Z" for ch in folded if ch.isalpha()), cue["text"]


# ============================================================================ demo script (section 20, Phase 4)


def test_demo_has_about_twenty_paragraphs_s20() -> None:
    assert 18 <= len(demo()) <= 22


@pytest.mark.parametrize("paragraph", demo(), ids=lambda p: p["segment_id"])
def test_demo_paragraph_has_one_to_eight_cues_of_50_to_450_characters_s20(paragraph: dict[str, Any]) -> None:
    assert 1 <= len(paragraph["cues"]) <= 8
    assert 50 <= len(join(paragraph["cues"])) <= 450


def test_demo_hints_name_terms_that_occur_as_whole_words_s9_1() -> None:
    content = load(DEMO, "paragraphs.json")
    hints = content["hints"]
    assert len(hints) >= 2
    text = " ".join(join(p["cues"]) for p in demo())
    for hint in hints:
        assert set(hint) <= {"term", "respell", "align_as", "asr_aliases"}, "the Hint fragment of design 7.2"
        assert hint["term"] and hint["respell"]
        # whole words: bounded by the start or end, a space, or punctuation (the apostrophe included)
        assert re.search(rf"(?<![^\W_]){re.escape(hint['term'])}(?![^\W_])", text), hint["term"]
    assert sum(len(c.get("exact", [])) for p in demo() for c in p["cues"]) >= 2, "a few exact spans"


# ============================================================================ canary (sections 3.5, 10.1, 16)


def test_canary_description_is_positive_only_s3_5() -> None:
    description = load(CANARY, "canary.json")["voice"]["description"]
    assert negations(description) == []
    assert 0 < len(description) <= design_config()["limits"]["max_description_chars"]
    assert description == unicodedata.normalize("NFC", description)


def test_canary_design_text_is_the_designs_default_s16() -> None:
    design_text = load(CANARY, "canary.json")["voice"]["design_text"]
    assert design_text == design_config()["voice_design"]["design_text"]
    assert len(design_text) <= MAX_DESIGN_TEXT_CHARS
    assert text_findings(design_text) == []


def test_canary_gate_text_and_seeds_are_fixed_s10_1() -> None:
    canary = load(CANARY, "canary.json")
    gate_text = canary["gate"]["text"]
    assert gate_text == canonical(gate_text)
    assert text_findings(gate_text) == [] and refused(gate_text) == []
    assert 1 <= len(re.findall(r"[.!?…](?=\s|$)", gate_text)) <= 2, "one or two sentences"
    assert len(gate_text) <= MAX_DESIGN_TEXT_CHARS
    for seed in (canary["voice"]["seed"], canary["gate"]["seed"]):
        assert isinstance(seed, int) and 0 <= seed <= 0x7FFFFFFF, "a seed as design 10.3 derives them"


@pytest.mark.parametrize("case", lint_cases(), ids=lambda c: c["name"])
def test_local_negation_check_agrees_with_the_lint_fixtures_s3_5(case: dict[str, Any]) -> None:
    """The canary check above uses the local word list; it must find what the lint fixtures expect."""
    expected = [(f["trigger"], f["offset"]) for f in case["expect"]["findings"]]
    assert negations(case["description"]) == expected


# ============================================================================ text fixtures (section 9.3)

REQUIRED_TEXT_CASES = {
    "canonical_form_collapses_whitespace_runs",
    "canonical_form_applies_nfc",
    "join_is_cues_joined_by_one_space",
    "text_differing_from_join_is_refused",
    "hint_matches_whole_words_only",
    "hint_matches_case_sensitively",
    "hint_longest_term_first",
    "hint_possessive_replaces_the_term_only",
    "hint_never_applied_across_a_cue_boundary",
    "warn_digit",
    "warn_symbols",
    "warn_unit_like",
    "warn_unit_like_lone_lower_case_m_s_g",
    "no_warning_for_a_A_I_in_am",
    "lone_capitals_are_letter_info",
    "passing_punctuation_raises_no_warning",
    "exact_span_offsets_are_code_points_beyond_the_bmp",
    "exact_span_word_range_survives_canonical_form",
    "exact_span_that_cuts_a_hyphenated_word_is_refused",
    "markup_square_brackets_are_refused",
    "markup_special_token_delimiters_are_refused",
    "control_character_nul_is_refused",
}


def test_text_fixtures_cover_design_9_3_s9_3() -> None:
    names = [c["name"] for c in text_cases()]
    assert len(names) == len(set(names)), "case names are unique"
    assert set(names) >= REQUIRED_TEXT_CASES
    lint_names = [c["name"] for c in lint_cases()]
    assert len(lint_names) == len(set(lint_names))


@pytest.mark.parametrize("case", text_cases(), ids=lambda c: c["name"])
def test_text_fixture_is_well_formed_s9_3(case: dict[str, Any]) -> None:
    assert {"name", "design", "about", "input", "expect"} <= set(case)
    assert re.fullmatch(r"[A-Za-z0-9_]+", case["name"])
    assert SEGMENT_ID.match(case["input"]["segment"]["segment_id"])
    expect = case["expect"]
    assert expect["outcome"] in {"ok", "refused"}
    if expect["outcome"] == "refused":
        assert expect["error"]["code"] in {"INVALID_ARGUMENT", "TEXT_REFUSED"}
    for path in case.get("unsure", []):
        resolve(case, path)
    for cue_expect in expect.get("cues", []):
        for warning in cue_expect.get("warnings", []):
            assert warning["kind"] in {"digit", "symbol", "unit_like", "letter"}
            assert warning["severity"] == ("info" if warning["kind"] == "letter" else "warn")


@pytest.mark.parametrize("case", text_cases(), ids=lambda c: c["name"])
def test_text_fixture_agrees_with_the_local_rules_s9_3(case: dict[str, Any]) -> None:
    """A cross-check of the fixtures' offsets and texts against the local rules, where the design fixes them."""
    unsure = set(case.get("unsure", []))
    segment, expect = case["input"]["segment"], case["expect"]
    cues = segment.get("cues") or [{"text": segment["text"]}]
    spoken = [canonical(c["text"]) for c in cues]
    joined = " ".join(spoken)

    if expect["outcome"] == "refused":
        code = expect["error"]["code"]
        if code == "TEXT_REFUSED" and case["input"].get("strict_text"):
            assert any(text_findings(c["text"]) for c in cues)
        elif code == "TEXT_REFUSED":
            assert any(refused(c["text"]) for c in cues)
            offenders = {item for c in cues for item in refused(c["text"])}
            for offender in expect.get("offenders", []):
                assert (offender["text"], offender["offset"]) in offenders
        elif "text" in segment and "cues" in segment:
            assert canonical(segment["text"]) != joined
        else:
            assert any(span_error(c["text"], c.get("exact", []), strict=False) for c in cues)
        return

    assert all(refused(c["text"]) == [] for c in cues)
    for c in cues:
        assert span_error(c["text"], c.get("exact", []), strict=False) is None
    if "text" in segment and "cues" in segment:
        assert canonical(segment["text"]) == joined
    if "spoken" in expect:
        assert expect["spoken"] == joined
    if "spoken_chars" in expect:
        assert expect["spoken_chars"] == len(joined)
    for i, cue_expect in enumerate(expect.get("cues", [])):
        text = cues[i]["text"]
        if "spoken" in cue_expect:
            assert cue_expect["spoken"] == spoken[i]
        if "span" in cue_expect:
            start, end = cue_expect["span"]
            assert joined[start:end] == spoken[i]
        if "warnings" in cue_expect and f"expect.cues[{i}].warnings" not in unsure:
            got = text_findings(text)
            assert [(f.kind, f.severity) for f in got] == [(w["kind"], w["severity"]) for w in cue_expect["warnings"]]
            for finding, warning in zip(got, cue_expect["warnings"], strict=True):
                assert finding.token == warning.get("token", finding.token)
                assert finding.offset == warning.get("offset", finding.offset)
        for applied in cue_expect.get("hints_applied", []):
            assert spoken[i][applied["offset"] :].startswith(applied["term"])
        for span, span_expect in zip(cues[i].get("exact", []), cue_expect.get("exact", []), strict=True):
            if "words" in span_expect:
                assert span_words(text, span["start"], span["end"]) == span_expect["words"]
            if "word_range" in span_expect:
                chosen = covered(text, span["start"], span["end"])
                assert [chosen[0], chosen[-1] + 1] == span_expect["word_range"]


# ============================================================================ QA fault specs (section 20, Phase 3)

REQUIRED_QA_SPECS = {
    "fault_exact_span_different_number",
    "fault_head_insertion_three_words",
    "fault_token_cap_hit",
    "fault_unplaceable_cue",
    "nonfault_per_cent_heard_as_percent",
    "nonfault_nought_heard_as_zero",
    "nonfault_one_slip_in_a_three_word_title",
}


def test_qa_specs_cover_the_phase_3_exit_criteria_s20() -> None:
    names = [s["name"] for s in qa_specs()]
    assert len(names) == len(set(names))
    assert set(names) >= REQUIRED_QA_SPECS


@pytest.mark.parametrize("spec", qa_specs(), ids=lambda s: s["name"])
def test_qa_spec_is_well_formed_s20(spec: dict[str, Any]) -> None:
    assert spec["kind"] in {"fault", "non-fault"}
    assert spec["asr"]["text"].strip()
    render_text = spec["render"]["text"]
    assert text_findings(render_text) == [] and refused(render_text) == [], "what is rendered is spoken form too"
    expect = spec["expect"]
    for flag in expect.get("flags", []):
        assert flag["code"] in FLAG_CODES and flag["severity"] in SEVERITIES
        if "cue" in flag:
            assert 0 <= flag["cue"] < len(spec["segment"]["cues"])
    for code in expect.get("absent", []):
        assert code in FLAG_CODES
    for item in expect.get("absent_severity", []):
        assert item["code"] in FLAG_CODES and item["severity"] in SEVERITIES
    spans = [s for c in spec["segment"]["cues"] for s in c.get("exact", [])]
    assert len(expect.get("exact", [])) <= len(spans)
    for exact in expect.get("exact", []):
        assert exact["match"] in {"same", "different", "missing"}
    if spec["kind"] == "fault":
        assert expect["flags"], "a planted fault names the flag that catches it"
    else:
        assert expect["verdict"] in {"pass", "warn"}
        assert all(f["severity"] != "fail" for f in expect.get("flags", []))
    for path in spec.get("unsure", []):
        resolve(spec, path)
