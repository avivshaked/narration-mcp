"""Spike (f): re-run and save the "Marufubens" check, with silence prepended (design sections 1, 20).

The bakeoff's names probe heard "Marufubens", the first word of its segment n05, split or misspelt in every
take. Design section 1 says this is BELIEVED not to be clipping: an earlier session checked it with 0.5 s of
silence prepended, but saved neither script nor output. This spike repeats the check and saves both.

What it does, for each of the bakeoff's six clone takes (d2 and d4, seeds 1 to 3):

- cuts segment n05 (it starts with "Marufubens") out of the take by its recorded sample boundaries, and
  measures its onset: how long before the first 20 ms frame within 40 dB of the segment's p95 frame level
  (the relative rule of design section 13), and the level of its first 20 ms;
- transcribes it with Whisper-large-v3 exactly as the bakeoff did (``eval/evaluate.py``: the transformers
  ASR pipeline in fp16, English, ``return_timestamps=True``), as it is and with 0.5 s and 1.0 s of digital
  silence prepended;
- transcribes n04 as a control, where the same name comes mid-sentence.

If silence changes nothing, the split is not Whisper stumbling on speech that starts at the very first
sample. The onset measure shows whether the take could have lost its first phoneme to qwen-tts's
proportional cut of the reference (plan.md section 1.3).

The segments' texts are the bakeoff caller's, and Whisper's transcripts of them, the heard forms of the name
included, are transcripts of the caller's takes. They are read from the bakeoff at run time and never copied
into this repository. The published results hold numbers only: for each transcription, whether the name was
heard as written, how many words it was heard as, and the character edit distance of the heard form to the
written name; and how often prepending silence changed the heard form. The name itself is the one word
published, as design section 1 cites it.

Run from the checkout in the QA worker's venv, with the GPU lock held (a few minutes)::

    workers/qa/.venv/Scripts/python.exe spikes/f-marufubens-silence/run.py

Writes ``spikes/f-marufubens-silence/results.json`` (published), and ``.dev/spikes/f/results-local.json``
(the heard forms) and ``.dev/spikes/f/heard-full.json`` (the full transcripts), which stay local.
``--publish-only`` rebuilds the published file from ``results-local.json``, with no GPU and no model.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_spike_common as common

HERE = Path(__file__).resolve().parent
LOCAL = common.DEV_SPIKES / "f"
NAME = "marufubens"
"""The name as Whisper's English normaliser writes it; found in the reference by this, never by position."""
TAKES = [
    f"qwen3-tts-1.7b-clone-{voice}-seed{seed}"
    for voice in ("d2-late-night_take1", "d4-radio-drama_take2")
    for seed in (1, 2, 3)
]
SCRIPT = "r48_names_probe"
SILENCES_S = (0.0, 0.5, 1.0)
FRAME_S = 0.02
RELATIVE_DB = 40.0


def onset(audio: Any, rate: int) -> dict[str, float]:
    """Leading silence by design section 13's relative rule, and the level of the first 20 ms."""
    import numpy as np

    frame = int(rate * FRAME_S)
    count = audio.size // frame
    frames = audio[: count * frame].reshape(count, frame).astype(np.float64)
    rms_db = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-12)
    threshold = float(np.percentile(rms_db, 95)) - RELATIVE_DB
    first = int(np.argmax(rms_db > threshold))
    return {
        "leading_silence_s": round(first * FRAME_S, 3),
        "first_20ms_rms_db": round(float(rms_db[0]), 1),
        "threshold_db": round(threshold, 1),
        "first_sample_abs": round(float(abs(audio[0])), 5),
    }


def align(reference: list[str], hypothesis: list[str]) -> list[tuple[int | None, int | None]]:
    """A minimal word-level edit alignment: (reference index, hypothesis index) pairs, None for a gap."""
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    cost = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(cols):
        cost[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost[i][j] = min(
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
                cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
            )
    pairs: list[tuple[int | None, int | None]] = []
    i, j = rows - 1, cols - 1
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif j and cost[i][j] == cost[i][j - 1] + 1:
            pairs.append((None, j - 1))
            j -= 1
        else:
            pairs.append((i - 1, None))
            i -= 1
    return pairs[::-1]


def heard_name(reference: list[str], hypothesis: list[str]) -> str:
    """The transcript's words that stand where the reference has the name.

    These are the words aligned to the name, plus any inserted between its neighbours' places. An empty
    result means the name was dropped.
    """
    at = reference.index(NAME)
    pairs = align(reference, hypothesis)
    before = max((k for k, (r, _) in enumerate(pairs) if r is not None and r < at), default=-1)
    after = min((k for k, (r, _) in enumerate(pairs) if r is not None and r > at), default=len(pairs))
    return " ".join(hypothesis[h] for _, h in pairs[before + 1 : after] if h is not None)


def edit_distance(a: str, b: str) -> int:
    """The Levenshtein distance between two strings, in characters."""
    previous = list(range(len(b) + 1))
    for i, char in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def measured(heard_as: str) -> dict[str, Any]:
    """What is published about one heard form: numbers, never the form itself.

    ``edit_distance`` compares the heard words joined without spaces with the written name, so a split is
    counted by ``words``, not as an extra character.
    """
    words = heard_as.split()
    return {
        "as_written": heard_as == NAME,
        "words": len(words),
        "edit_distance": edit_distance("".join(words), NAME),
    }


def distances(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def published(local: dict[str, Any]) -> dict[str, Any]:
    """The results to publish, from the local ones: every heard form replaced by ``measured``'s numbers."""
    rows: list[dict[str, Any]] = []
    for row in local["rows"]:
        forms = {silence: str(v["name_heard_as"]) for silence, v in row["heard"].items()}
        out: dict[str, Any] = {"take": row["take"], "segment": row["segment"]}
        if "onset" in row:
            out["onset"] = row["onset"]
        out["heard"] = {silence: measured(form) for silence, form in forms.items()}
        out["heard_form_changed_by_silence"] = len(set(forms.values())) > 1
        rows.append(out)

    def segment_summary(segment: str) -> dict[str, Any]:
        mine = [r for r in rows if r["segment"] == segment]
        heard = [v for r in mine for v in r["heard"].values()]
        words: dict[str, int] = {}
        for v in heard:
            words[str(v["words"])] = words.get(str(v["words"]), 0) + 1
        return {
            "takes": len(mine),
            "transcriptions": len(heard),
            "heard_as_written": sum(1 for v in heard if v["as_written"]),
            "heard_as_words": dict(sorted(words.items())),
            "edit_distance_when_not_as_written": distances(
                [int(v["edit_distance"]) for v in heard if not v["as_written"]]
            ),
            "takes_whose_heard_form_silence_changed": sum(1 for r in mine if r["heard_form_changed_by_silence"]),
        }

    return {
        "spike": local["spike"],
        "design_sections": local["design_sections"],
        "ran_at": local["ran_at"],
        "software": local["software"],
        "asr": local["asr"],
        "silences_s": local["silences_s"],
        "onset_rule": local["onset_rule"],
        "name": NAME,
        "edit_distance_rule": "Levenshtein distance in characters from the heard words, joined without spaces,"
        " to the written name (as Whisper's English normaliser writes both)",
        "rows": rows,
        "summary": {
            "n05": segment_summary("n05"),
            "n04": segment_summary("n04"),
            "n05_leading_silence_s": [r["onset"]["leading_silence_s"] for r in rows if r["segment"] == "n05"],
        },
        "network_attempts": local["network_attempts"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--publish-only",
        action="store_true",
        help="rebuild results.json from .dev/spikes/f/results-local.json (no GPU, no model)",
    )
    if parser.parse_args().publish_only:
        local = json.loads((LOCAL / "results-local.json").read_text(encoding="utf-8"))
        common.write_json(HERE / "results.json", published(local))
        return 0
    common.prepare_process_env()
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import WhisperProcessor, pipeline

    guard = common.NetworkGuard().install()
    snapshot, revision = common.snapshot(common.MODEL_ASR)
    asr = pipeline("automatic-speech-recognition", model=str(snapshot), dtype=torch.float16, device="cuda:0")
    normalise = WhisperProcessor.from_pretrained(str(snapshot)).tokenizer.normalize
    from wavlm_similarity import to16k

    def transcribe(audio16: Any) -> str:
        out: dict[str, Any] = cast(
            dict[str, Any],
            asr(
                {"raw": audio16, "sampling_rate": 16000},
                return_timestamps=True,
                generate_kwargs={"language": "english", "task": "transcribe"},
            ),
        )
        return str(out["text"]).strip()

    rows: list[dict[str, Any]] = []
    full: list[dict[str, Any]] = []
    for take in TAKES:
        folder = common.bakeoff_root() / "outputs" / take
        meta = json.loads((folder / f"{SCRIPT}.json").read_text(encoding="utf-8"))
        audio, rate = sf.read(str(folder / f"{SCRIPT}.wav"), dtype="float32")
        for segment_id in ("n05", "n04"):
            segment = next(s for s in meta["segments"] if s["id"] == segment_id)
            piece = audio[segment["start_sample"] : segment["end_sample"]]
            row: dict[str, Any] = {"take": take.removeprefix("qwen3-tts-1.7b-clone-"), "segment": segment_id}
            if segment_id == "n05":
                row["onset"] = onset(piece, rate)
            reference = normalise(segment["text"]).split()
            heard: dict[str, dict[str, Any]] = {}
            for silence in SILENCES_S if segment_id == "n05" else (0.0, 0.5):
                padded = np.concatenate([np.zeros(round(silence * rate), dtype=np.float32), piece])
                text = transcribe(to16k(padded, rate))
                name = heard_name(reference, normalise(text).split())
                heard[f"{silence:.1f}s"] = {"name_heard_as": name}  # local only: published() keeps numbers
                full.append({"take": row["take"], "segment": segment_id, "silence_s": silence, "text": text})
            row["heard"] = heard
            rows.append(row)
            print(row["take"], segment_id, {k: measured(v["name_heard_as"]) for k, v in heard.items()})

    common.write_json(LOCAL / "heard-full.json", full)
    local = {
        "spike": "f (the Marufubens check, silence prepended)",
        "design_sections": ["1", "20 (f)"],
        "ran_at": common.now(),
        "software": common.software_facts(torch),
        "asr": {
            "model": common.MODEL_ASR,
            "revision": revision,
            "method": "bakeoff eval/evaluate.py Scorer.transcribe",
        },
        "silences_s": list(SILENCES_S),
        "onset_rule": f"first {int(FRAME_S * 1000)} ms frame within {RELATIVE_DB:.0f} dB of the p95 frame level",
        "rows": rows,
        "network_attempts": guard.attempts,
    }
    common.write_json(LOCAL / "results-local.json", local)
    common.write_json(HERE / "results.json", published(local))
    return 0


if __name__ == "__main__":
    sys.exit(main())
