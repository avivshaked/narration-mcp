"""Spike (b), part 3 (plan.md DC-11): the wildcard, measured on the bake-off's six clone takes.

The lead approved the wildcard on two conditions: it must fix the boundary after n05's left-out numbers on
all six takes, and it must move no other boundary beyond noise. This script measures both, and what a
wildcard's own span is worth (for a cue made of wildcards alone):

1. **Boundaries.** The 24 paragraph pairs of the evidence tests (``tests/align/test_bakeoff_evidence_s11_2.py``),
   one cue per sentence, go through ``narration.align`` twice: *before*, the transcript with its wildcards
   taken out (exactly what the aligner built before DC-11), and *after*, the transcript as built now. Every
   cue boundary is compared. The boundary between the two paragraphs has a known place, the 0.9 s of
   silence the bake-off inserted: its error is how much of that silence falls outside the boundary. A
   boundary between sentences has no hand mark; its reference is the model's own greedy reading (the
   argmax path) of the words on either side, and its error the mean distance of the two edges from it.
2. **Words.** Every word the greedy reading spells as the transcript does: its start and end against the
   greedy reading, before and after, overall and for the words next to a wildcard.
3. **Wildcard spans.** For each wildcard between two words the greedy reading matched, the greedy words
   between them are the speech it stands for: the span's start and end against theirs.
4. **Planted wildcards.** A ``*`` added where nothing was left out, in a pause between sentences and between
   two words inside a sentence: the span it takes and its score, against the real ones.

Run it from the checkout with the server's venv, both environment variables set, and a scratch folder
inside the project::

    uv run python spikes/b-forced-align-cpu/wildcard.py --scratch .dev/<folder>

It starts ``wildcard_emit.py`` with the QA worker's venv, reads the takes in place (never copied), and
writes ``wildcard.json`` and ``wildcard_boundaries.csv`` next to itself. No audio is written, and no text:
the bake-off's text is private, so its words are named only by (cue, word) index.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import difflib
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from narration.align import WILDCARD, CtcAligner, wildcard_runs
from narration.contracts.interfaces import AlignTranscript
from narration.contracts.models import Alignment, CueIn, SegmentIn
from narration.contracts.worker import AlignReply
from narration.text import TextPipeline

HERE = Path(__file__).resolve().parent
CHECKOUT = HERE.parents[1]
REPO = "facebook/wav2vec2-large-960h-lv60-self"
REVISION = "54074b1c16f4de6a5ad59affb4caa8f2ea03a119"
SNAPSHOT = Path("models--facebook--wav2vec2-large-960h-lv60-self") / "snapshots" / REVISION
SCRIPT = "r48_names_probe"
PAIRS = (("n01", "n02"), ("n03", "n04"), ("n05", "n06"), ("n07", "n08"))
FRAME_S = 0.02
THREADS = "4"
NOISE_S = 0.02
"""One energy frame: a boundary that moves by at most this has not moved beyond noise."""


# ---------------------------------------------------------------------- transcripts


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def strip_wildcards(t: AlignTranscript) -> AlignTranscript:
    """The transcript as the aligner built it before DC-11: every ``*`` and the separator joining it gone."""
    tokens: list[str] = []
    owners: list[tuple[int, int] | None] = []
    for token, owner in zip(t.tokens, t.token_words, strict=True):
        if token == WILDCARD:
            if tokens and tokens[-1] == "|":
                tokens.pop()
                owners.pop()
            continue
        if token == "|" and (not tokens or tokens[-1] == "|"):
            continue
        tokens.append(token)
        owners.append(owner)
    while tokens and tokens[-1] == "|":
        tokens.pop()
        owners.pop()
    return dataclasses.replace(t, tokens=tuple(tokens), token_words=tuple(owners))


@dataclasses.dataclass
class Group:
    """A run of tokens between separators: a spelled word (or a hyphen's half), or one wildcard."""

    owner: tuple[int, int]
    letters: str
    tokens: list[int]

    @property
    def wildcard(self) -> bool:
        return self.letters == WILDCARD


def groups(tokens: tuple[str, ...] | list[str], owners: Any) -> list[Group]:
    out: list[Group] = []
    current: Group | None = None
    for i, (token, owner) in enumerate(zip(tokens, owners, strict=True)):
        if token == "|" or owner is None:
            current = None
            continue
        if token == WILDCARD:
            out.append(Group(owner, WILDCARD, [i]))
            current = None
            continue
        if current is None:
            current = Group(owner, "", [])
            out.append(current)
        current.letters += token
        current.tokens.append(i)
    return out


def frames_of(group: Group, spans: list[dict[str, Any]]) -> tuple[int, int]:
    return min(spans[i]["start_frame"] for i in group.tokens), max(spans[i]["end_frame"] for i in group.tokens)


def match(gs: list[Group], greedy: list[list[Any]]) -> dict[int, int]:
    """Group index → greedy word index, for the letter groups the greedy reading spells the same."""
    letters = [g.letters for g in gs]
    matcher = difflib.SequenceMatcher(a=letters, b=[w[0] for w in greedy], autojunk=False)
    out: dict[int, int] = {}
    for i, j, size in matcher.get_matching_blocks():
        for n in range(size):
            if not gs[i + n].wildcard:
                out[i + n] = j + n
    return out


# ---------------------------------------------------------------------- the model side


def run_emitter(jobs: list[dict[str, Any]], scratch: Path, snapshot: Path) -> dict[str, Any]:
    venv = CHECKOUT / "workers" / "qa" / ".venv"
    python = next((p for p in (venv / "Scripts" / "python.exe", venv / "bin" / "python") if p.is_file()), None)
    if python is None:
        raise SystemExit("the QA worker's venv is not synced (workers/qa/.venv)")
    jobs_path, out_path = scratch / "wildcard_jobs.json", scratch / "wildcard_emitted.json"
    jobs_path.write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(CHECKOUT / "workers" / "qa" / "src"),
        OMP_NUM_THREADS=THREADS,
        MKL_NUM_THREADS=THREADS,
        OPENBLAS_NUM_THREADS=THREADS,
        NUMEXPR_NUM_THREADS=THREADS,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
    )
    argv = [str(python), str(HERE / "wildcard_emit.py"), "--snapshot", str(snapshot), "--jobs", str(jobs_path)]
    subprocess.run([*argv, "--out", str(out_path)], env=env, check=True, timeout=1800)
    return json.loads(out_path.read_text(encoding="utf-8"))


def as_reply(spans: list[dict[str, Any]], num_frames: int) -> AlignReply:
    return {
        "id": 0,
        "ok": True,
        "frame_s": FRAME_S,
        "num_frames": num_frames,
        "spans": spans,  # type: ignore[typeddict-item]
        "model": REPO,
        "revision": REVISION,
        "device": "cpu",
    }


# ---------------------------------------------------------------------- measuring


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    absolute = sorted(abs(v) for v in values)
    return {
        "n": len(values),
        "mean_abs": round(statistics.fmean(absolute), 2),
        "p50_abs": round(float(np.percentile(absolute, 50)), 2),
        "p95_abs": round(float(np.percentile(absolute, 95)), 2),
        "max_abs": round(absolute[-1], 2),
    }


def plants(gs: list[Group], cue_of_group: list[int], first_cues: int) -> dict[str, int]:
    """Where to plant a ``*``: after which group (a letter group followed by a letter group, no wildcard near).

    ``pause``: the first sentence boundary inside the first paragraph (or the second, if the first has one
    sentence); ``speech``: the middle of the first sentence, between two words.
    """
    out: dict[str, int] = {}
    boundary_cue = 0 if first_cues > 1 else first_cues
    for i in range(len(gs) - 1):
        a, b = gs[i], gs[i + 1]
        if a.wildcard or b.wildcard or a.owner == b.owner:
            continue
        if cue_of_group[i] == boundary_cue and cue_of_group[i + 1] == boundary_cue + 1:
            out.setdefault("pause", i)
    first = [i for i in range(len(gs) - 1) if cue_of_group[i] == 0 and cue_of_group[i + 1] == 0]
    for i in first[len(first) // 2 :]:
        a, b = gs[i], gs[i + 1]
        if not a.wildcard and not b.wildcard and a.owner != b.owner and not gs[i - 1].wildcard:
            out["speech"] = i
            break
    return out


def reference_s(
    gi: int,
    edge: int,
    gs: list[Group],
    wild_ref: dict[int, tuple[int, int] | None],
    matched: dict[int, int],
    greedy: list[list[Any]],
) -> float | None:
    """The greedy reading's time of group ``gi``'s start (``edge`` 0) or end (1), in seconds, if it has one."""
    if gs[gi].wildcard:
        ref = wild_ref.get(gi)
        return None if ref is None else ref[edge] * FRAME_S
    wi = matched.get(gi)
    return None if wi is None else greedy[wi][1 + edge] * FRAME_S


def plant(tokens: tuple[str, ...], gs: list[Group], after: int) -> list[str]:
    cut = gs[after].tokens[-1] + 1
    return [*tokens[:cut], "|", WILDCARD, *tokens[cut:]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--scratch", type=Path, required=True, help="a folder inside the project for the job files")
    args = parser.parse_args()
    bakeoff = Path(os.environ["NARRATION_BAKEOFF_ROOT"])
    snapshot = Path(os.environ["NARRATION_MODELS_ROOT"]) / SNAPSHOT
    args.scratch.mkdir(parents=True, exist_ok=True)
    aligner = CtcAligner(revision=REVISION)
    pipeline = TextPipeline()

    cases: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    takes = sorted(p for p in (bakeoff / "outputs").iterdir() if (p / f"{SCRIPT}.json").is_file())
    for take in takes:
        meta = json.loads((take / f"{SCRIPT}.json").read_text(encoding="utf-8"))
        rate = int(meta["sample_rate"])
        paragraphs = {p["id"]: p for p in meta["segments"]}
        for first, second in PAIRS:
            a, b = paragraphs[first], paragraphs[second]
            first_cues = sentences(a["text"])
            cues = tuple(CueIn(text=c) for c in (*first_cues, *sentences(b["text"])))
            segment = pipeline.plan_segment(SegmentIn(segment_id=f"{first}-{second}", cues=cues), [])
            after = aligner.build_transcript(segment, [])
            before = strip_wildcards(after)
            gs = groups(after.tokens, after.token_words)
            cue_of_group = [g.owner[0] for g in gs]
            variants: dict[str, list[str]] = {"before": list(before.tokens), "after": list(after.tokens)}
            where = plants(gs, cue_of_group, len(first_cues))
            for kind, index in where.items():
                variants[f"plant_{kind}"] = plant(after.tokens, gs, index)
            start, end = int(a["start_sample"]), int(b["end_sample"])
            jobs.append(
                {"wav": str(take / f"{SCRIPT}.wav"), "start_s": start / rate, "end_s": end / rate, "variants": variants}
            )
            cases.append(
                {
                    "take": take.name,
                    "pair": f"{first}+{second}",
                    "wav": take / f"{SCRIPT}.wav",
                    "start": start,
                    "end": end,
                    "rate": rate,
                    "first_cues": len(first_cues),
                    "gap": ((int(a["end_sample"]) - start) / rate, (int(b["start_sample"]) - start) / rate),
                    "before": before,
                    "after": after,
                    "plants": where,
                }
            )
    emitted = run_emitter(jobs, args.scratch, snapshot)

    rows: list[dict[str, Any]] = []
    word_errors: dict[str, list[float]] = {"before": [], "after": []}
    near_errors: dict[str, list[float]] = {"before": [], "after": []}
    wild_start: list[float] = []
    wild_end: list[float] = []
    real_wild: list[dict[str, Any]] = []
    planted: list[dict[str, Any]] = []
    for case, out in zip(cases, emitted["jobs"], strict=True):
        greedy = out["greedy"]
        frames = int(out["num_frames"])
        audio, rate = sf.read(str(case["wav"]), dtype="float32", start=case["start"], stop=case["end"], always_2d=True)
        audio = np.asarray(audio, dtype=np.float32)
        results: dict[str, Alignment] = {}
        for name in ("before", "after"):
            reply = out["replies"][name]
            assert reply["ok"], (case["take"], case["pair"], name, reply)
            results[name] = aligner.resolve(case[name], as_reply(reply["spans"], frames), audio, rate, [], None)

        # words, before and after, against the greedy reading
        after_groups = groups(case["after"].tokens, case["after"].token_words)
        near_wild = set()
        for i, g in enumerate(after_groups):
            if g.wildcard:
                near_wild.update({after_groups[k].owner for k in (i - 1, i + 1) if 0 <= k < len(after_groups)})
        for name in ("before", "after"):
            t = case[name]
            gs = groups(t.tokens, t.token_words)
            spans = out["replies"][name]["spans"]
            for gi, wi in match(gs, greedy).items():
                s, e = frames_of(gs[gi], spans)
                errs = [s - greedy[wi][1], e - greedy[wi][2]]
                word_errors[name].extend(errs)
                if gs[gi].owner in near_wild:
                    near_errors[name].extend(errs)

        # the wildcards' own spans against the greedy words between their matched neighbours
        spans = out["replies"]["after"]["spans"]
        matched = match(after_groups, greedy)
        runs = wildcard_runs(case["after"])
        wild_ref: dict[int, tuple[int, int] | None] = {}
        for gi, g in enumerate(after_groups):
            if not g.wildcard:
                continue
            span = spans[g.tokens[0]]
            record = {
                "take": case["take"],
                "pair": case["pair"],
                "words": [list(k) for k in runs[g.tokens[0]]],
                "frames": span["end_frame"] - span["start_frame"],
                "score": round(span["score"], 3),
            }
            ref = None
            jp, jn = matched.get(gi - 1), matched.get(gi + 1)
            if jp is not None and jn is not None and jn - jp >= 2:
                ref = (greedy[jp + 1][1], greedy[jn - 1][2])
                wild_start.append(span["start_frame"] - ref[0])
                wild_end.append(span["end_frame"] - ref[1])
                record["reference_frames"] = list(ref)
                record["greedy_words_between"] = jn - jp - 1
            wild_ref[gi] = ref
            record["span_frames"] = [span["start_frame"], span["end_frame"]]
            real_wild.append(record)

        # planted wildcards: what they take, and how far they push their neighbours
        after_spans = out["replies"]["after"]["spans"]
        for kind, index in case["plants"].items():
            reply = out["replies"][f"plant_{kind}"]
            if not reply["ok"]:
                planted.append({"take": case["take"], "pair": case["pair"], "kind": kind, "error": reply})
                continue
            tokens = plant(case["after"].tokens, after_groups, index)
            star = tokens.index(WILDCARD, after_groups[index].tokens[-1] + 1)
            span = reply["spans"][star]
            shift = []
            for gi in (index, index + 1):
                old = frames_of(after_groups[gi], after_spans)
                offset = 0 if gi == index else 2  # the planted "|" and "*" come before the later word's tokens
                new_tokens = [i + offset for i in after_groups[gi].tokens]
                new = (
                    min(reply["spans"][i]["start_frame"] for i in new_tokens),
                    max(reply["spans"][i]["end_frame"] for i in new_tokens),
                )
                shift.extend([new[0] - old[0], new[1] - old[1]])
            planted.append(
                {
                    "take": case["take"],
                    "pair": case["pair"],
                    "kind": kind,
                    "frames": span["end_frame"] - span["start_frame"],
                    "score": round(span["score"], 3),
                    "neighbour_shift_frames": shift,
                }
            )

        # boundaries
        before, after = results["before"], results["after"]
        gap_lo, gap_hi = case["gap"]
        last_group: dict[int, int] = {}
        first_group: dict[int, int] = {}
        for gi, g in enumerate(after_groups):
            first_group.setdefault(g.owner[0], gi)
            last_group[g.owner[0]] = gi

        for k in range(len(after.cues) - 1):
            b0, b1 = before.cues[k].end_s, before.cues[k + 1].start_s
            a0, a1 = after.cues[k].end_s, after.cues[k + 1].start_s
            kind = "paragraph" if k == case["first_cues"] - 1 else "sentence"
            row: dict[str, Any] = {
                "take": case["take"],
                "pair": case["pair"],
                "boundary": f"{k}-{k + 1}",
                "kind": kind,
                "before_end_s": b0,
                "before_start_s": b1,
                "after_end_s": a0,
                "after_start_s": a1,
            }
            if None in (b0, b1, a0, a1):
                row["note"] = "a cue was not placed"
                rows.append(row)
                continue
            assert b0 is not None and b1 is not None and a0 is not None and a1 is not None
            row["moved_s"] = round(max(abs(a0 - b0), abs(a1 - b1)), 3)
            if kind == "paragraph":
                row["ref_end_s"], row["ref_start_s"] = round(gap_lo, 3), round(gap_hi, 3)
                row["error_before_s"] = round(max(0.0, b0 - gap_lo) + max(0.0, gap_hi - b1), 3)
                row["error_after_s"] = round(max(0.0, a0 - gap_lo) + max(0.0, gap_hi - a1), 3)
            else:
                where = (after_groups, wild_ref, matched, greedy)
                ref_end = reference_s(last_group[k], 1, *where) if k in last_group else None
                ref_start = reference_s(first_group[k + 1], 0, *where) if k + 1 in first_group else None
                row["ref_end_s"] = None if ref_end is None else round(ref_end, 3)
                row["ref_start_s"] = None if ref_start is None else round(ref_start, 3)
                if ref_end is not None and ref_start is not None:
                    row["error_before_s"] = round((abs(b0 - ref_end) + abs(b1 - ref_start)) / 2, 3)
                    row["error_after_s"] = round((abs(a0 - ref_end) + abs(a1 - ref_start)) / 2, 3)
            rows.append(row)

    # ------------------------------------------------------------------ summary
    measured = [r for r in rows if "moved_s" in r]
    moved = [r for r in measured if r["moved_s"] > NOISE_S]
    paragraph = [r for r in measured if r["kind"] == "paragraph"]
    n05 = [r for r in paragraph if r["pair"] == "n05+n06"]
    sentence = [r for r in measured if r["kind"] == "sentence" and "error_after_s" in r]
    summary = {
        "boundaries": {
            "total": len(rows),
            "measured": len(measured),
            "moved_not_at_all": sum(1 for r in measured if r["moved_s"] == 0),
            "moved_at_most_one_frame": len(measured) - len(moved),
            "moved_more": [
                {
                    k: r[k]
                    for k in ("take", "pair", "boundary", "kind", "moved_s", "error_before_s", "error_after_s")
                    if k in r
                }
                for r in moved
            ],
            "n05_n06_paragraph_boundary": [
                {
                    k: r[k]
                    for k in (
                        "take",
                        "before_start_s",
                        "after_start_s",
                        "ref_start_s",
                        "error_before_s",
                        "error_after_s",
                    )
                }
                for r in n05
            ],
            "paragraph_boundaries_holding_the_gap": {
                "before": sum(1 for r in paragraph if r["error_before_s"] <= NOISE_S),
                "after": sum(1 for r in paragraph if r["error_after_s"] <= NOISE_S),
                "of": len(paragraph),
            },
            "sentence_error_s": {
                "before": stats([r["error_before_s"] for r in sentence]),
                "after": stats([r["error_after_s"] for r in sentence]),
            },
        },
        "words_vs_greedy_frames": {name: stats(word_errors[name]) for name in word_errors},
        "words_next_to_a_wildcard_vs_greedy_frames": {name: stats(near_errors[name]) for name in near_errors},
        "wildcard_span_vs_greedy_frames": {"start": stats(wild_start), "end": stats(wild_end)},
        "wildcard_spans": {
            "real": {
                "n": len(real_wild),
                "frames": stats([w["frames"] for w in real_wild]),
                "score": stats([w["score"] for w in real_wild]),
            },
            "planted": {
                kind: {
                    "n": len(ps),
                    "frames": stats([p["frames"] for p in ps]),
                    "score": stats([p["score"] for p in ps]),
                    "neighbour_shift_frames": stats([s for p in ps for s in p["neighbour_shift_frames"]]),
                }
                for kind in ("pause", "speech")
                if (ps := [p for p in planted if p["kind"] == kind and "frames" in p])
            },
        },
    }
    result = {
        "spike": "b-forced-align-cpu/wildcard",
        "date": dt.date.today().isoformat(),
        "model": f"{REPO}@{REVISION}",
        "takes": len(takes),
        "pairs": len(cases),
        "noise_s": NOISE_S,
        "load_s": emitted["load_s"],
        "align_s": emitted["align_s"],
        "summary": summary,
        "wildcards": real_wild,
        "planted": planted,
    }
    (HERE / "wildcard.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    fields = [
        "take",
        "pair",
        "boundary",
        "kind",
        "before_end_s",
        "before_start_s",
        "after_end_s",
        "after_start_s",
        "moved_s",
        "ref_end_s",
        "ref_start_s",
        "error_before_s",
        "error_after_s",
        "note",
    ]
    with (HERE / "wildcard_boundaries.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k, "")) for k in fields})
    print("wrote wildcard.json and wildcard_boundaries.csv", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
