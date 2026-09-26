"""Spike (b): does torchaudio's ``forced_align`` run on this build's CPU, and does the pinned wav2vec2 load?

Design section 11.2 step 3 BELIEVED that ``torchaudio.functional.forced_align`` + ``merge_tokens`` run on
the CPU of the QA worker's torch/torchaudio 2.11.0+cu128 build; section 1 says an earlier smoke test was
never saved. This script re-checks it and saves the answer (plan.md WP15), and records whether
``facebook/wav2vec2-large-960h-lv60-self`` at the pinned revision, which has only ``pytorch_model.bin``,
loads under transformers 5.17.0 and torch 2.11 (``torch.load``'s ``weights_only`` default).

Run it with the QA worker's interpreter, from the repository root (``--out`` names the results file)::

    workers/qa/.venv/Scripts/python.exe spikes/b-forced-align-cpu/spike_b.py --out <results.json>

Needs ``NARRATION_MODELS_ROOT`` (the pinned snapshot). With ``NARRATION_BAKEOFF_ROOT`` it also aligns four
paragraphs from one bake-off clone take, read in place by path and never copied. Nothing is written except
the results file, and it holds numbers only: the bake-off's text is private, so no text, word or transcript
from it is ever written. The GPU is never used (``CUDA_VISIBLE_DEVICES`` is emptied before torch is
imported), and the CPU thread cap is 4.
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import hashlib
import inspect
import json
import os
import platform
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

THREADS = 4
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = str(THREADS)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "workers" / "qa" / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
import torchaudio.functional as taf  # noqa: E402
import transformers  # noqa: E402
from narration_worker_qa import align as qa_align  # noqa: E402

MODEL_REPO = "facebook/wav2vec2-large-960h-lv60-self"
MODEL_REVISION = "54074b1c16f4de6a5ad59affb4caa8f2ea03a119"
SNAPSHOT = Path("models--facebook--wav2vec2-large-960h-lv60-self") / "snapshots" / MODEL_REVISION
TAKE = Path("outputs") / "qwen3-tts-1.7b-clone-d2-late-night_take1-seed1"
"""One of the bake-off's six clone takes (plan.md section 1.2)."""
PARAGRAPHS = ("n08", "n01", "n05", "n06")
"""n08 is short and has one number; n01 is the take's longest paragraph (19.8 s), for the speed figure; n05
and n06 are dense with numbers, for the left-out-word experiment."""


def _versions() -> dict[str, str]:
    import scipy

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "soundfile": sf.__version__,
    }


def _record_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [f"{w.category.__name__}: {str(w.message)[:300]}" for w in caught]


def check_api() -> dict[str, Any]:
    """forced_align and merge_tokens on a tiny synthetic emission: present, working, deterministic, warnings."""
    out: dict[str, Any] = {
        "forced_align_present": hasattr(taf, "forced_align"),
        "merge_tokens_present": hasattr(taf, "merge_tokens"),
        "forced_align_doc_mentions_deprecation": "deprecat" in (taf.forced_align.__doc__ or "").lower(),
    }
    gen = torch.Generator().manual_seed(1234)
    logits = torch.randn(1, 60, 6, generator=gen)
    # Make the path unambiguous: targets 1,2,2,3 peak at frames 10, 20, 30, 40, blank elsewhere.
    for frame, token in ((10, 1), (20, 2), (30, 2), (40, 3)):
        logits[0, frame, token] += 12.0
    logits[0, :, 0] += 4.0
    emission = torch.log_softmax(logits, dim=-1)
    targets = torch.tensor([[1, 2, 2, 3]], dtype=torch.int32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runs = []
        for _ in range(2):
            path, scores = taf.forced_align(emission, targets, blank=0)
            spans = taf.merge_tokens(path[0], scores[0].exp(), blank=0)
            runs.append([(int(s.token), int(s.start), int(s.end), round(float(s.score), 6)) for s in spans])
    out["synthetic_spans"] = runs[0]
    out["synthetic_deterministic"] = runs[0] == runs[1]
    out["warnings"] = _record_warnings(caught)
    # The guard: fewer frames than tokens + repeats (4 tokens, 1 repeat, 4 frames) must raise.
    try:
        taf.forced_align(emission[:, :4], targets, blank=0)
        out["too_short_raises"] = False
    except Exception as exc:
        out["too_short_raises"] = True
        # The message starts with torchaudio's build-time source path, which says nothing here; keep the rest.
        message = re.sub(r"^.*?compute\.cpp:\d+, ", "", str(exc))
        out["too_short_exception"] = f"{type(exc).__name__}: {message[:300]}"
    try:
        qa_align.align_emission(torch, emission[0, :4], [1, 2, 2, 3], 0)
        out["worker_guard"] = "no error"
    except qa_align.AlignmentFailure as exc:
        out["worker_guard"] = {"code": exc.code, "details": exc.details}
    return out


def check_load(models_root: Path) -> tuple[dict[str, Any], Any]:
    """Load the pinned snapshot offline; record the weights file's hash, torch.load's weights_only, warnings."""
    snapshot = models_root / SNAPSHOT
    out: dict[str, Any] = {"snapshot": f"<models_root>/{SNAPSHOT.as_posix()}", "files": sorted(os.listdir(snapshot))}
    manifest_path = models_root / "manifest.json"
    weights = snapshot / "pytorch_model.bin"
    digest = hashlib.sha256()
    with weights.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    out["pytorch_model_bin_sha256"] = digest.hexdigest()
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest.get(f"{MODEL_REPO}@{MODEL_REVISION}", {})
        out["sha256_matches_models_manifest"] = (
            entry.get("files_sha256", {}).get("pytorch_model.bin") == (out["pytorch_model_bin_sha256"])
        )
    out["torch_load_weights_only_default"] = repr(inspect.signature(torch.load).parameters["weights_only"].default)

    calls: list[dict[str, Any]] = []
    real_load = torch.load

    def spying_load(*args: Any, **kwargs: Any) -> Any:
        calls.append({"weights_only": repr(kwargs.get("weights_only", "<default>"))})
        return real_load(*args, **kwargs)

    torch.load = spying_load  # type: ignore[assignment]
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            start = time.perf_counter()
            model, info = transformers.Wav2Vec2ForCTC.from_pretrained(
                str(snapshot), local_files_only=True, dtype=torch.float32, output_loading_info=True
            )
            out["load_s"] = round(time.perf_counter() - start, 2)
    except Exception as exc:
        out["loads"] = False
        out["error"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
        return out, None
    finally:
        torch.load = real_load  # type: ignore[assignment]
    out["loads"] = True
    out["torch_load_calls"] = calls
    out["loading_info"] = {k: sorted(map(str, v)) for k, v in info.items()}
    out["warnings"] = _record_warnings(caught)
    out["parameters"] = sum(p.numel() for p in model.parameters())
    out["dtype"] = str(next(model.parameters()).dtype)
    out["attn_implementation"] = str(getattr(model.config, "_attn_implementation", None))
    return out, model


def _tokens(text: str, wildcard: bool = False) -> tuple[list[str], list[str]]:
    """A minimal transcript for the spike (the service's own is ``narration.align``): upper-case letters and
    the apostrophe, ``|`` between words; a word with anything else (a digit) is left out, or, with
    ``wildcard``, a run of such words becomes one ``*`` token."""
    words, tokens = [], []
    for raw in text.split():
        word = raw.strip('.,;:!?"()').upper()
        if not word:
            continue
        if any(not (ch.isascii() and (ch.isalpha() or ch == "'")) for ch in word):
            if wildcard and (not tokens or tokens[-1] != "*"):
                tokens.extend(["|", "*"] if tokens else ["*"])
            continue
        if tokens:
            tokens.append("|")
        tokens.extend(word)
        words.append(word)
    return words, tokens


def _word_times(spans: list[dict[str, Any]], tokens: list[str]) -> list[tuple[int, int]]:
    """(first frame, end frame) of each word's letters, in order, skipping ``|`` and ``*``."""
    out, current = [], []
    for span, token in zip(spans, tokens, strict=True):
        if token in ("|", "*"):
            if current:
                out.append((current[0], current[-1]))
                current = []
            continue
        current += [span["start_frame"], span["end_frame"]]
    if current:
        out.append((current[0], current[-1]))
    return out


def _greedy_words(emission: Any, labels: dict[int, str]) -> list[tuple[str, int, int]]:
    """Unconstrained CTC: the argmax path's words with the frames of their first and last letters."""
    best = emission.argmax(dim=-1).tolist()
    words: list[tuple[str, int, int]] = []
    letters, first, last, prev = "", -1, -1, 0
    for frame, label in enumerate(best):
        if label != 0 and label != prev:
            char = labels[label]
            if char == "|":
                if letters:
                    words.append((letters, first, last))
                letters = ""
            else:
                if not letters:
                    first = frame
                letters += char
                last = frame + 1
        elif label != 0 and label == prev and labels[label] != "|":
            last = frame + 1
        prev = label
    if letters:
        words.append((letters, first, last))
    return words


def wildcard_experiment(aligner: Any, piece: Any, rate: int, text: str, labels: dict[int, str]) -> dict[str, Any]:
    """Words left out of the transcript (digits): dropped (the design) versus one ``*`` wildcard token per run.

    The wildcard is an extra emission column whose log-probability is log(1 − P(blank)) per frame: it can
    absorb speech but not silence. The reference is the unconstrained greedy CTC path, for the words it spells
    exactly as the text does. Reported: the mean absolute error of word starts and ends, in frames, for all
    such words and for the words next to a left-out word.
    """
    import difflib

    mono = qa_align.to_mono_16k(piece, rate)
    emission = aligner.emission(mono)
    vocab = {v: k for k, v in labels.items()}
    greedy = _greedy_words(emission, labels)
    words, plain = _tokens(text)
    _, starred = _tokens(text, wildcard=True)
    star_col = torch.log1p(-emission[:, 0].exp().clamp(max=1 - 1e-6)).unsqueeze(1)
    extended = torch.cat([emission, star_col], dim=1)
    star_id = extended.shape[1] - 1
    results: dict[str, Any] = {}
    neighbours: set[int] = set()
    k = 0
    for raw in text.split():
        word = raw.strip('.,;:!?"()').upper()
        if word and any(not (ch.isascii() and (ch.isalpha() or ch == "'")) for ch in word):
            neighbours.update({k - 1, k})
        elif word:
            k += 1
    matcher = difflib.SequenceMatcher(a=words, b=[g[0] for g in greedy], autojunk=False)
    pairs = [(i + n, j + n) for i, j, size in matcher.get_matching_blocks() for n in range(size)]
    for name, tokens, emis in (("dropped", plain, emission), ("wildcard", starred, extended)):
        ids = [star_id if t == "*" else vocab[t] for t in tokens]
        spans = qa_align.align_emission(torch, emis, ids, 0)
        times = _word_times(spans, tokens)
        errors = [(abs(times[i][0] - greedy[j][1]), abs(times[i][1] - greedy[j][2])) for i, j in pairs]
        near = [e for (i, _), e in zip(pairs, errors, strict=True) if i in neighbours]
        results[name] = {
            "words_compared": len(errors),
            "mean_abs_error_frames": round(sum(a + b for a, b in errors) / (2 * len(errors)), 2) if errors else None,
            "next_to_left_out_compared": len(near),
            "next_to_left_out_mean_abs_error_frames": (
                round(sum(a + b for a, b in near) / (2 * len(near)), 2) if near else None
            ),
            "worst_word_error_frames": max((max(e) for e in errors), default=None),
        }
    return results


def check_alignment(bakeoff: Path, models_root: Path) -> dict[str, Any]:
    """Align paragraphs of a bake-off take through the worker's own code; record speed and word times.

    Numbers only: the take's text is read at run time and never written (it is private).
    """
    take_dir = bakeoff / TAKE
    meta = json.loads((take_dir / "r48_names_probe.json").read_text(encoding="utf-8"))
    audio, rate = sf.read(str(take_dir / "r48_names_probe.wav"), dtype="float32", always_2d=True)
    aligner = qa_align.Wav2Vec2Aligner(torch)
    load_s = aligner.load(MODEL_REPO, MODEL_REVISION, models_root / SNAPSHOT)
    vocab = json.loads((models_root / SNAPSHOT / "vocab.json").read_text(encoding="utf-8"))
    labels = {v: k for k, v in vocab.items()}
    out: dict[str, Any] = {
        "take": f"<bakeoff>/{TAKE.as_posix()}/r48_names_probe.wav",
        "aligner_load_s": round(load_s, 2),
    }
    for seg in meta["segments"]:
        if seg["id"] not in PARAGRAPHS:
            continue
        piece = audio[seg["start_sample"] : seg["end_sample"]]
        words, tokens = _tokens(seg["text"])
        mono = qa_align.to_mono_16k(piece, rate)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            emission = aligner.emission(mono)
            t1 = time.perf_counter()
            reply = aligner.align(piece, rate, tokens)
            t2 = time.perf_counter()
            again = aligner.align(piece, rate, tokens)
        heard = [w[0] for w in _greedy_words(emission, labels)]
        same = difflib.SequenceMatcher(a=words, b=heard, autojunk=False)
        spelled_alike = sum(size for _, _, size in same.get_matching_blocks())
        spans = reply["spans"]
        word_times, k = [], 0
        for word in words:
            first, last = spans[k], spans[k + len(word) - 1]
            word_times.append([round(first["start_frame"] * 0.02, 2), round(last["end_frame"] * 0.02, 2)])
            k += len(word) + 1
        audio_s = piece.shape[0] / rate
        out[seg["id"]] = {
            "audio_s": round(audio_s, 2),
            "samples_16k": int(mono.shape[0]),
            "frames": reply["num_frames"],
            "frames_formula": qa_align.ctc_frames(int(mono.shape[0])),
            "tokens": len(tokens),
            "repeats": qa_align.repeats(tokens),
            "emission_s": round(t1 - t0, 2),
            "align_call_s": round(t2 - t1, 2),
            "rtf_align_call": round((t2 - t1) / audio_s, 3),
            "deterministic": reply == again,
            "transcript_words": len(words),
            "greedy_words": len(heard),
            "greedy_words_spelled_as_the_text": spelled_alike,
            "mean_token_score": round(sum(s["score"] for s in spans) / len(spans), 4),
            "word_times_s": word_times,
            "warnings": _record_warnings(caught),
            "left_out_words_dropped_vs_wildcard": wildcard_experiment(aligner, piece, rate, seg["text"], labels),
        }
    short = audio[: int(0.1 * rate)]
    try:
        aligner.align(short, rate, _tokens("Winds came over the ridge")[1])  # invented: 25 tokens, no repeat
        out["too_short_0_1_s"] = "no error"
    except qa_align.AlignmentFailure as exc:
        out["too_short_0_1_s"] = {"code": exc.code, "message": exc.message, "details": exc.details}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results: dict[str, Any] = {
        "spike": "b-forced-align-cpu",
        "run_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "versions": _versions(),
        "machine": {"system": platform.system(), "release": platform.release(), "cpu_count": os.cpu_count()},
        "threads": {"cap": THREADS, "torch_intra_op": torch.get_num_threads()},
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    }
    torch.set_num_threads(THREADS)
    results["threads"]["torch_intra_op"] = torch.get_num_threads()
    results["api"] = check_api()
    models_root = os.environ.get("NARRATION_MODELS_ROOT")
    if models_root:
        results["load"], _model = check_load(Path(models_root))
        bakeoff = os.environ.get("NARRATION_BAKEOFF_ROOT")
        if bakeoff and results["load"].get("loads"):
            results["alignment"] = check_alignment(Path(bakeoff), Path(models_root))
        else:
            results["alignment"] = "skipped: NARRATION_BAKEOFF_ROOT is not set or the model did not load"
    else:
        results["load"] = "skipped: NARRATION_MODELS_ROOT is not set"
    text = json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
