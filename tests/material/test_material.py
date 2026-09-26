"""The service's own material under ``material/`` (plan.md WP18).

Design sections: 3.2 (calibration corpus and length ladder), 3.5 (positive-only voice descriptions), 7.2
(canonical form, join, spoken length, exact spans), 9.1 and 9.3 (text checks and their fixtures), 10.1 (the
canary), 11.2 (the alignment benchmark), 15 (material ships versioned and hashed), 16 (defaults), and 20
(the Phase 3 QA fixtures and the Phase 4 demo script).

The material is judged by the service's own code: ``narration.text`` plans every spoken-form segment as a
caller's would be planned, ``narration.lint`` reads the canary's description, and the published tool schemas
check that every paragraph can be sent as it is. What is tested here beyond those is the material's own
rules: the ladder's lengths, number words marked exact, the alignment benchmark's pause marks, and so on.
The text and lint fixtures themselves are run through the pipeline by ``tests/text`` and ``tests/lint``.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
import tomllib
import unicodedata
from pathlib import Path
from typing import Any, get_args

import jsonschema
import pytest

from narration.config import VoiceDesignConfig
from narration.contracts import codes, names, schemas
from narration.contracts.models import Hint, SegmentIn, SegmentText
from narration.contracts.serial import from_json, to_json
from narration.lint import lint
from narration.qa.profile import QaProfile
from narration.text import TextPipeline, canonical_form, spoken_length, words

ROOT = Path(__file__).resolve().parents[2]
MATERIAL = ROOT / "material"

CALIBRATION = ("calibration", names.CORPUS)
ALIGNMENT = ("alignment", names.BENCHMARK)
CANARY = ("canary", "canary.v1")
DEMO = ("demo", "demo-en.v1")
TEXT_FIXTURES = ("fixtures", "text-v1")
QA_FIXTURES = ("fixtures", "qa-faults-v1")
EXPECTED_SETS = {CALIBRATION, ALIGNMENT, CANARY, DEMO, TEXT_FIXTURES, QA_FIXTURES}

SENTENCE_END = re.compile(r"[.!?…][\"”’')]*$")

_SUBMIT_JOB = schemas.TOOLS_BY_NAME["submit_job"].input_schema["properties"]
SEGMENT_SCHEMA = jsonschema.Draft202012Validator(_SUBMIT_JOB["segments"]["items"])
HINT_SCHEMA = jsonschema.Draft202012Validator(_SUBMIT_JOB["hints"]["items"])
_DESIGN_VOICE = schemas.TOOLS_BY_NAME["design_voice"].input_schema["properties"]
MAX_DESCRIPTION_CHARS = _DESIGN_VOICE["description"]["maxLength"]
MAX_DESIGN_TEXT_CHARS = _DESIGN_VOICE["design_text"]["maxLength"]
VERDICTS = get_args(names.Verdict)
"""pass, warn, fail: the contract lists the verdicts from best to worst (design 11.1)."""
FAILING = VERDICTS[-1]
EXACT_MATCHES = get_args(names.ExactMatch)
TEXT_WARNING_KINDS = get_args(names.TextWarningKind)
TEXT_WARNING_SEVERITIES = codes.FLAGS[codes.WRITTEN_FORM_TOKEN].severities

_ONES = ("zero", "nought", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
_TEENS = ("eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
_TENS = ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
CARDINALS = frozenset(_ONES + _TEENS + _TENS + ("hundred", "thousand", "million"))
"""The number words the calibration corpus marks as exact spans: the material's own rule (design 3.2, 11.3)."""
SPAN_NUMBER_WORDS = CARDINALS | {"and", "point"}


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
    out = [(f"{names.CORPUS}/{p['segment_id']}", p) for p in calibration() + ladder()]
    out += [(f"{names.BENCHMARK}/{p['segment_id']}", p) for p in alignment()]
    out += [(f"demo-en.v1/{p['segment_id']}", p) for p in demo()]
    out += [(f"qa-faults-v1/{s['name']}", s["segment"]) for s in qa_specs()]
    return out


def spoken_texts() -> list[tuple[str, str]]:
    """The other spoken-form texts: the canary's design and gate texts, and what the QA specs render."""
    canary = load(CANARY, "canary.json")
    out = []
    if canary:
        out += [("canary.v1/design_text", canary["voice"]["design_text"]), ("canary.v1/gate", canary["gate"]["text"])]
    out += [(f"qa-faults-v1/{s['name']}/render", s["render"]["text"]) for s in qa_specs()]
    return out


def manifests() -> list[Path]:
    return sorted(MATERIAL.glob("*/*/manifest.json"))


def material_files() -> list[Path]:
    return sorted(p for p in MATERIAL.glob("*/*/*") if p.is_file())


def ids(items: list[tuple[str, Any]]) -> list[str]:
    return [name for name, _ in items]


# ============================================================================ through the service's own code


def as_sent(segment: dict[str, Any]) -> dict[str, Any]:
    """A material paragraph as a caller sends it: its Segment fields, without the material's metadata."""
    return {k: v for k, v in segment.items() if k in {"segment_id", "cues", "text"}}


def plan(segment: dict[str, Any]) -> SegmentText:
    """The segment through the text pipeline (design 9.1), as a caller's would go; raises on a refusal."""
    return TextPipeline().plan_segment(from_json(SegmentIn, as_sent(segment)), [])


def findings(planned: SegmentText) -> list[Any]:
    """Every text finding and segment flag the pipeline raised, info included."""
    return [to_json(w) for c in planned.cues for w in c.warnings] + [to_json(f) for f in planned.warnings]


def one_cue(text: str) -> dict[str, Any]:
    return {"segment_id": "t", "cues": [{"text": text}]}


def core_edges(text: str) -> tuple[set[int], set[int]]:
    """Where the words' cores start and end (``narration.text.words``, design 7.2)."""
    ws = words(text)
    return {w.core_start for w in ws}, {w.core_end for w in ws}


def covered(text: str, start: int, end: int) -> list[str]:
    """The words whose core lies inside ``[start, end)``."""
    return [w.text for w in words(text) if start <= w.core_start and w.core_end <= end]


def number_parts(word: str) -> list[str]:
    return word.lower().split("-")


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
    assert m["status"] in get_args(names.MaterialStatus)
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
def test_spoken_form_segment_can_be_sent_as_it_is_s7_2(name: str, segment: dict[str, Any]) -> None:
    """The Segment schema of submit_job accepts it, and its cues are already in canonical form."""
    errors = [e.message for e in SEGMENT_SCHEMA.iter_errors(as_sent(segment))]
    assert errors == []
    for cue in segment["cues"]:
        assert cue["text"] == canonical_form(cue["text"])


@pytest.mark.parametrize(("name", "segment"), spoken_segments(), ids=ids(spoken_segments()))
def test_spoken_form_segment_raises_no_text_finding_s9_1(name: str, segment: dict[str, Any]) -> None:
    """The text pipeline accepts it and finds nothing: no digit, symbol, unit or lone letter, info included."""
    planned = plan(segment)
    assert findings(planned) == []
    if "spoken_chars" in segment:
        assert segment["spoken_chars"] == planned.spoken_chars == spoken_length(planned.spoken_text)


@pytest.mark.parametrize(("name", "text"), spoken_texts(), ids=ids(spoken_texts()))
def test_other_spoken_texts_raise_no_text_finding_s9_1(name: str, text: str) -> None:
    assert text == canonical_form(text)
    assert findings(plan(one_cue(text))) == []


@pytest.mark.parametrize(("name", "segment"), spoken_segments(), ids=ids(spoken_segments()))
def test_exact_spans_cover_whole_words_and_no_punctuation_s7_2(name: str, segment: dict[str, Any]) -> None:
    """The pipeline accepted the spans above; the service's own material also keeps punctuation out of them."""
    for index, cue in enumerate(segment["cues"]):
        starts, ends = core_edges(cue["text"])
        for span in cue.get("exact", []):
            assert span["start"] in starts and span["end"] in ends, f"cue {index}: {span} in {cue['text']!r}"


# ============================================================================ calibration and ladder (section 3.2)


def test_corpus_is_the_one_the_design_configures_s3_2() -> None:
    assert design_config()["measurement"]["corpus"] == names.CORPUS


def test_ladder_rungs_are_the_designs_s3_2() -> None:
    targets = [p["target_spoken_chars"] for p in ladder()]
    assert targets == design_config()["measurement"]["length_ladder_spoken_chars"]


@pytest.mark.parametrize("paragraph", ladder(), ids=lambda p: p["segment_id"])
def test_ladder_paragraph_hits_its_target_within_five_per_cent_s3_2(paragraph: dict[str, Any]) -> None:
    target = paragraph["target_spoken_chars"]
    spoken = plan(paragraph).spoken_chars
    assert abs(spoken - target) <= 0.05 * target, f"{spoken} spoken characters for a target of {target}"


def test_calibration_has_three_paragraphs_of_150_to_300_spoken_characters_s3_2() -> None:
    lengths = [plan(p).spoken_chars for p in calibration()]
    assert len(lengths) == 3
    assert all(150 <= n <= 300 for n in lengths), lengths


@pytest.mark.parametrize("paragraph", calibration() + ladder(), ids=lambda p: p["segment_id"])
def test_calibration_number_words_are_marked_exact_s3_2(paragraph: dict[str, Any]) -> None:
    """Spans hold only number words, and every number word but "one" (so often not a count) is in a span."""
    for cue in paragraph["cues"]:
        text = cue["text"]
        spans = cue.get("exact", [])
        for span in spans:
            for word in covered(text, span["start"], span["end"]):
                assert all(part in SPAN_NUMBER_WORDS for part in number_parts(word)), f"{word!r} in a span"
        for w in words(text):
            if any(part in CARDINALS - {"one"} for part in number_parts(w.text)):
                assert any(s["start"] <= w.core_start and w.core_end <= s["end"] for s in spans), (
                    f"number word {w.text!r} is not marked exact"
                )


def test_most_calibration_paragraphs_have_number_words_s3_2() -> None:
    paragraphs = calibration() + ladder()
    with_numbers = [p for p in paragraphs if any(c.get("exact") for c in p["cues"])]
    assert len(with_numbers) * 3 >= len(paragraphs) * 2


SETS_WITH_NAMES = [CALIBRATION, ALIGNMENT, DEMO]


@pytest.mark.parametrize("key", SETS_WITH_NAMES, ids=lambda k: k[1])
def test_each_set_has_invented_names_that_occur_in_it_s3_2(key: tuple[str, str]) -> None:
    content = load(key, "paragraphs.json")
    paragraphs = content.get("calibration", []) + content.get("ladder", []) + content.get("paragraphs", [])
    text = " ".join(plan(p).spoken_text for p in paragraphs)
    assert content["invented_names"], "at least one invented name per set"
    for name in content["invented_names"]:
        assert re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text), f"{name!r} does not occur"


# ============================================================================ alignment benchmark (section 11.2)


def test_alignment_benchmark_is_the_configured_one_of_about_twelve_paragraphs_s11_2() -> None:
    assert design_config()["alignment"]["benchmark"] == names.BENCHMARK
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
    all_words = [w.text for p in alignment() for c in p["cues"] for w in words(c["text"])]
    assert sum(1 for w in all_words if set(number_parts(w)) & CARDINALS) >= 5


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
    assert 50 <= plan(paragraph).spoken_chars <= 450


def test_demo_hints_apply_to_the_script_s9_1() -> None:
    """Every hint is a valid Hint with a respelling, and applies somewhere in the script, inside a cue."""
    content = load(DEMO, "paragraphs.json")
    raw_hints = content["hints"]
    assert len(raw_hints) >= 2
    for hint in raw_hints:
        assert [e.message for e in HINT_SCHEMA.iter_errors(hint)] == []
        assert hint["respell"]
    planned = TextPipeline().plan_request(
        [from_json(SegmentIn, as_sent(p)) for p in demo()],
        [from_json(Hint, h) for h in raw_hints],
        strict_text=False,
    )
    applied = {h.term for segment in planned for cue in segment.cues for h in cue.hints_applied}
    assert applied == {h["term"] for h in raw_hints}
    assert all(segment.warnings == () for segment in planned), "no term split across cues"
    assert sum(len(c.get("exact", [])) for p in demo() for c in p["cues"]) >= 2, "a few exact spans"


# ============================================================================ canary (sections 3.5, 10.1, 16)


def test_canary_description_is_positive_only_s3_5() -> None:
    description = load(CANARY, "canary.json")["voice"]["description"]
    assert lint(description).findings == ()
    assert 0 < len(description) <= MAX_DESCRIPTION_CHARS
    assert description == unicodedata.normalize("NFC", description)


def test_canary_design_text_is_the_designs_default_s16() -> None:
    """One default design text: design section 16, the code's default, the shipped config and the canary."""
    design_text = load(CANARY, "canary.json")["voice"]["design_text"]
    assert design_text == design_config()["voice_design"]["design_text"]
    assert design_text == VoiceDesignConfig().design_text
    example = tomllib.loads((ROOT / "narration.example.toml").read_text(encoding="utf-8"))
    assert design_text == example["voice_design"]["design_text"]
    assert len(design_text) <= MAX_DESIGN_TEXT_CHARS


def test_canary_gate_text_and_seeds_are_fixed_s10_1() -> None:
    canary = load(CANARY, "canary.json")
    gate_text = canary["gate"]["text"]
    assert 1 <= len(re.findall(r"[.!?…](?=\s|$)", gate_text)) <= 2, "one or two sentences"
    assert len(gate_text) <= MAX_DESIGN_TEXT_CHARS
    for seed in (canary["voice"]["seed"], canary["gate"]["seed"]):
        assert isinstance(seed, int) and 0 <= seed <= 0x7FFFFFFF, "a seed as design 10.3 derives them"


# ============================================================================ text fixtures (section 9.3)

REQUIRED_TEXT_CASES = {
    "canonical_form_collapses_whitespace_runs",
    "canonical_form_applies_nfc",
    "canonical_form_tab_and_newline",
    "join_is_cues_joined_by_one_space",
    "text_differing_from_join_is_refused",
    "hint_matches_whole_words_only",
    "hint_matches_case_sensitively",
    "hint_longest_term_first",
    "hint_possessive_replaces_the_term_only",
    "hint_never_applied_across_a_cue_boundary",
    "hint_split_across_three_cues_is_flagged_once",
    "hint_repeated_term_is_refused",
    "hint_empty_respelling_is_refused",
    "hint_respelled_term_with_digits_raises_no_warning",
    "warn_digit",
    "warn_symbols",
    "warn_unit_like",
    "warn_unit_like_lone_lower_case_m_s_g",
    "format_character_warns_and_is_not_refused",
    "no_warning_for_a_A_I_in_am",
    "lone_capitals_are_letter_info",
    "passing_punctuation_raises_no_warning",
    "exact_span_offsets_are_code_points_beyond_the_bmp",
    "exact_span_word_range_survives_canonical_form",
    "exact_span_that_cuts_a_hyphenated_word_is_refused",
    "markup_square_brackets_are_refused",
    "markup_special_token_delimiters_are_refused",
    "control_character_nul_is_refused",
    "control_character_vertical_tab_is_refused",
    "control_character_form_feed_is_refused",
    "control_character_next_line_is_refused",
}


def test_text_fixtures_cover_design_9_3_s9_3() -> None:
    names_ = [c["name"] for c in text_cases()]
    assert len(names_) == len(set(names_)), "case names are unique"
    assert set(names_) >= REQUIRED_TEXT_CASES
    lint_names = [c["name"] for c in lint_cases()]
    assert len(lint_names) == len(set(lint_names))


@pytest.mark.parametrize("case", text_cases(), ids=lambda c: c["name"])
def test_text_fixture_is_well_formed_s9_3(case: dict[str, Any]) -> None:
    """The shape the fixture's conventions state. ``tests/text`` runs each case through the pipeline."""
    assert {"name", "design", "about", "input", "expect"} <= set(case)
    assert re.fullmatch(r"[A-Za-z0-9_]+", case["name"])
    expect = case["expect"]
    assert expect["outcome"] in {"ok", "refused"}
    if expect["outcome"] == "refused":
        assert expect["error"]["code"] in {codes.INVALID_ARGUMENT, codes.TEXT_REFUSED}
    assert "unsure" not in case, "every open point is settled (WP10); a case's note says which rule it pins"
    for flag in expect.get("flags", []):
        assert flag["code"] in codes.FLAGS and flag["severity"] in names.SEVERITIES
    for cue_expect in expect.get("cues", []):
        for warning in cue_expect.get("warnings", []):
            assert warning["kind"] in TEXT_WARNING_KINDS
            assert warning["severity"] in TEXT_WARNING_SEVERITIES


@pytest.mark.parametrize("case", lint_cases(), ids=lambda c: c["name"])
def test_lint_fixture_is_well_formed_s3_5(case: dict[str, Any]) -> None:
    """The shape the fixture's conventions state. ``tests/lint`` runs each case through the lint."""
    assert {"name", "design", "description", "expect"} <= set(case)
    for finding in case["expect"]["findings"]:
        assert case["description"][finding["offset"] :].startswith(finding["trigger"])
        assert finding["phrase"].startswith(finding["trigger"])


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
    names_ = [s["name"] for s in qa_specs()]
    assert len(names_) == len(set(names_))
    assert set(names_) >= REQUIRED_QA_SPECS


@pytest.mark.parametrize("spec", qa_specs(), ids=lambda s: s["name"])
def test_qa_spec_is_well_formed_s20(spec: dict[str, Any]) -> None:
    assert spec["kind"] in {"fault", "non-fault"}
    assert spec["asr"]["text"].strip()
    expect = spec["expect"]
    for flag in expect.get("flags", []):
        assert flag["code"] in codes.FLAGS and flag["severity"] in codes.FLAGS[flag["code"]].severities
        if "cue" in flag:
            assert 0 <= flag["cue"] < len(spec["segment"]["cues"])
    for code in expect.get("absent", []):
        assert code in codes.FLAGS
    for item in expect.get("absent_severity", []):
        assert item["code"] in codes.FLAGS and item["severity"] in names.SEVERITIES
    spans = [s for c in spec["segment"]["cues"] for s in c.get("exact", [])]
    assert len(expect.get("exact", [])) <= len(spans)
    for exact in expect.get("exact", []):
        assert exact["match"] in EXACT_MATCHES
    if "verdict" in expect:
        assert expect["verdict"] in VERDICTS
    if spec["kind"] == "fault":
        assert expect["flags"], "a planted fault names the flag that catches it"
    else:
        assert expect["verdict"] != FAILING
        assert all(f["severity"] != FAILING for f in expect.get("flags", []))
    assert "unsure" not in spec, "every open point is settled; a spec's note says which rule it pins"


def test_the_reference_bleed_spec_isolates_the_bleed_rule_s11_1() -> None:
    """The spec plants the tail of the transcript it names (the canary's design text), and too few words for
    a head insertion to fail on its count alone, so only the bleed rule can fail it (QA's profile)."""
    spec = next(s for s in qa_specs() if s["name"] == "fault_head_insertion_reference_bleed")
    transcript = spec["voice_transcript"]
    assert transcript == load(CANARY, "canary.json")["voice"]["design_text"]
    spoken = plan(spec["segment"]).spoken_text
    assert spec["render"]["text"].endswith(spoken) and spec["asr"]["text"].endswith(spoken)
    bleed = spec["render"]["text"][: -len(spoken)].strip()
    assert bleed and transcript.lower().endswith(bleed.lower())
    profile = QaProfile()
    assert profile.bleed_min_words <= len(words(bleed)) < profile.insertion_fail_words
