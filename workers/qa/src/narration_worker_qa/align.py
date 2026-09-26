"""The QA worker's CTC aligner (design sections 4 and 11.2, steps 2 and 3; Appendix A ``align``; plan.md WP15).

``facebook/wav2vec2-large-960h-lv60-self`` computes CTC emissions **on the CPU**, and torchaudio's
``forced_align`` finds the best path through the token sequence the server built. ``merge_tokens`` then
turns that path into one span of frames per token, with the token's mean posterior. That is all this module
decides: it is a model runner (plan.md P1). The server builds the tokens, keeps the token → (cue, word) map,
turns frames into cue times, snaps boundaries into pauses, scores confidence and raises every flag
(``narration.align``).

**The wildcard** (``*``; plan.md DC-11). The server puts one ``*`` in place of each run of words it cannot
spell in the model's alphabet (a number in digits, a symbol). The worker aligns it through one extra
emission column whose log-probability in each frame is log(1 − P(blank)): it can absorb any speech, but not
silence. Its span is the frames of that speech, and its score their mean 1 − P(blank). Without it, the
speech of the words left out had nowhere to go, and it pulled their neighbours off by up to 1.8 s (spike (b)).

Failures of the alignment itself raise ``AlignmentFailure``, the protocol's ``ALIGNMENT_ERROR``:

- **the guard.** ``forced_align`` needs at least one frame per token, plus a blank frame between two equal
  tokens: ``T ≥ L + R``. The worker checks it before the model runs, so audio too short for its text costs
  no model time. It checks it again on the real frame count, before ``forced_align`` sees the input;
- any exception raised by ``forced_align`` or ``merge_tokens``.

A failure to compute the emissions is not an alignment failure: it says nothing about the take, so it
propagates, and the request loop reports it as ``INTERNAL``.

Audio I/O uses soundfile, never torchaudio's loader. The model loads offline from a snapshot directory
named by its revision (section 4). ``torch`` is passed in by the caller: the worker's handler gets it from
``WorkerHandler.torch()``, which applies the CPU thread cap first (section 4.1).

The worker protocol reaches this code through ``AlignOp``: the QA role's handler (WP22) owns one, and
delegates the ``align`` op and the aligner's share of ``load``, ``unload`` and ``shutdown`` to it.

``python -m narration_worker_qa.align`` is a development entry point (spike (b) and the evidence tests): it
aligns a list of jobs and prints the replies as JSON, or writes them to a file.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from math import gcd
from pathlib import Path
from typing import Any, ClassVar, Final

import numpy as np
import numpy.typing as npt
from narration_worker.errors import OpError
from narration_worker.handler import Request, WorkerHandler, require_str, require_str_list
from narration_worker.protocol import WorkerErrorCode
from narration_worker.threads import cap_threads_env, cap_torch_threads

SAMPLE_RATE: Final = 16_000
"""wav2vec2's input rate; every clip is resampled to it (design section 11.2)."""
FRAME_STRIDE: Final = 320
"""Samples between two emission frames: the product of the feature encoder's strides (5 · 2⁶)."""
FRAME_WINDOW: Final = 400
"""Samples one emission frame sees: the feature encoder's receptive field."""
FRAME_S: Final = FRAME_STRIDE / SAMPLE_RATE
"""One emission frame in seconds (0.02). Frame ``i`` starts at ``i * FRAME_S``."""
BLANK: Final = "<pad>"
"""The CTC blank: wav2vec2's pad token."""
NOT_TARGETS: Final = frozenset({"<pad>", "<s>", "</s>", "<unk>"})
"""Vocabulary entries that can never be alignment targets."""
WILDCARD: Final = "*"
"""The token that stands for a run of words outside the model's alphabet (plan.md DC-11)."""
WILDCARD_FLOOR: Final = 1e-6
"""The smallest 1 − P(blank) the wildcard's column takes, so its log-probability stays finite (-13.8)."""
DEVICE: Final = "cpu"
"""The only device the aligner runs on (design section 11.2: the GPU path is untested and not needed)."""


# ---------------------------------------------------------------------- errors


class AlignerError(Exception):
    """A failure the ``align`` op reports as ``ok: false``; ``code`` is the protocol's error code.

    ``AlignOp`` turns it into the worker's ``OpError`` (``op_error``).
    """

    code: ClassVar[WorkerErrorCode] = "INTERNAL"

    def __init__(self, message: str, details: Mapping[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, object] = dict(details or {})


class AlignmentFailure(AlignerError):
    """The guard failed or ``forced_align`` raised: ``ALIGNMENT_ERROR``, with ``details`` {reason, frames,
    tokens, repeats} (and the exception's type and message when one was raised)."""

    code: ClassVar[WorkerErrorCode] = "ALIGNMENT_ERROR"


class InvalidRequest(AlignerError):
    """A token outside the model's vocabulary, or a load request the aligner cannot serve."""

    code: ClassVar[WorkerErrorCode] = "INVALID_REQUEST"


class UnreadableAudio(AlignerError):
    """A file soundfile cannot read, or one with non-finite samples."""

    code: ClassVar[WorkerErrorCode] = "UNSUPPORTED_AUDIO"


class BackendMissing(AlignerError):
    """The snapshot directory or one of its files is missing."""

    code: ClassVar[WorkerErrorCode] = "BACKEND_NOT_INSTALLED"


class NotLoaded(AlignerError):
    """``align`` before ``load`` (or after ``unload``)."""

    code: ClassVar[WorkerErrorCode] = "NOT_LOADED"


def op_error(error: AlignerError) -> OpError:
    """The worker's ``OpError`` for ``error``: the same code, message and details."""
    return OpError(error.code, error.message, error.details)


# ---------------------------------------------------------------------- pure helpers


def ctc_frames(samples_16k: int) -> int:
    """wav2vec2's frame count for ``samples_16k`` samples at 16 kHz: a 400-sample window every 320 samples.

    The encoder's seven convolutions (kernels 10,3,3,3,3,2,2; strides 5,2,2,2,2,2,2) compose exactly to this,
    because nested floor divisions by integers compose.
    """
    return 0 if samples_16k < FRAME_WINDOW else (samples_16k - FRAME_WINDOW) // FRAME_STRIDE + 1


def resample_ratio(sample_rate: int) -> tuple[int, int]:
    """(up, down) for ``resample_poly`` from ``sample_rate`` to 16 kHz, reduced by their gcd."""
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, not {sample_rate}")
    g = gcd(SAMPLE_RATE, sample_rate)
    return SAMPLE_RATE // g, sample_rate // g


def resampled_length(samples: int, sample_rate: int) -> int:
    """How many samples ``to_mono_16k`` returns for ``samples`` samples at ``sample_rate``."""
    up, down = resample_ratio(sample_rate)
    return -(-samples * up // down)


def repeats(tokens: Sequence[object]) -> int:
    """``R``: how many tokens equal the one before them (CTC needs a blank frame between the two)."""
    return sum(1 for a, b in itertools.pairwise(tokens) if a == b)


def check_guard(frames: int, tokens: Sequence[object]) -> None:
    """Raise ``AlignmentFailure`` unless ``frames ≥ tokens + repeats`` (and there is a token to align)."""
    if not tokens:
        raise AlignmentFailure(
            "there are no tokens to align", {"reason": "no_tokens", "frames": frames, "tokens": 0, "repeats": 0}
        )
    r = repeats(tokens)
    if frames < len(tokens) + r:
        raise AlignmentFailure(
            f"the audio is too short for its text: {frames} frames for {len(tokens)} tokens and {r} repeats "
            "(forced alignment needs frames ≥ tokens + repeats)",
            {"reason": "too_short", "frames": frames, "tokens": len(tokens), "repeats": r},
        )


def to_mono_16k(audio: npt.NDArray[np.floating[Any]], sample_rate: int) -> npt.NDArray[np.float32]:
    """Mean of the channels, resampled to 16 kHz with ``scipy.signal.resample_poly`` by the reduced ratio.

    Ported from the bake-off's ``eval/evaluate.py`` (``to16k``).
    """
    from scipy.signal import resample_poly

    data = np.asarray(audio, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1, dtype=np.float32)
    if sample_rate == SAMPLE_RATE or data.size == 0:
        return np.ascontiguousarray(data, dtype=np.float32)
    up, down = resample_ratio(sample_rate)
    return np.ascontiguousarray(resample_poly(data, up, down), dtype=np.float32)


def read_audio(path: Path) -> tuple[npt.NDArray[np.float32], int]:
    """A WAV as float32 frames × channels and its sample rate (soundfile; never torchaudio's loader)."""
    import soundfile as sf

    try:
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile raises LibsndfileError, RuntimeError or TypeError by version
        raise UnreadableAudio(
            f"cannot read {path.name} as audio: {exc}", {"path": str(path), "type": type(exc).__name__}
        ) from exc
    samples = np.asarray(data, dtype=np.float32)
    if not np.isfinite(samples).all():
        raise UnreadableAudio(f"{path.name} has samples that are not finite numbers", {"path": str(path)})
    return samples, int(rate)


def token_ids(tokens: Sequence[str], vocab: Mapping[str, int], wildcard_id: int | None = None) -> list[int]:
    """The vocabulary ids of ``tokens``, with ``*`` as ``wildcard_id`` (the extra column ``with_wildcard``
    adds); ``InvalidRequest`` names the first token the model cannot align (``*`` too, without an id)."""
    ids: list[int] = []
    for i, token in enumerate(tokens):
        if token == WILDCARD and wildcard_id is not None:
            ids.append(wildcard_id)
            continue
        if token in NOT_TARGETS or token not in vocab:
            raise InvalidRequest(
                f"tokens[{i}] = {token!r} is not a letter of the aligner's alphabet",
                {"field": "tokens", "index": i, "token": token},
            )
        ids.append(vocab[token])
    return ids


def with_wildcard(torch: Any, emission: Any, blank_id: int) -> Any:
    """``emission`` (log-probabilities, frames × classes) with one more column, the wildcard's:
    log(1 − P(blank)) per frame, floored at log(``WILDCARD_FLOOR``). Its id is the old class count."""
    speech = torch.log1p(-emission[:, blank_id].exp().clamp(max=1.0 - WILDCARD_FLOOR))
    return torch.cat([emission, speech.unsqueeze(1)], dim=1)


def align_emission(torch: Any, emission: Any, ids: Sequence[int], blank_id: int) -> list[dict[str, Any]]:
    """Forced alignment of ``ids`` through ``emission`` (log-probabilities, frames × classes, on the CPU).

    Returns one ``TokenSpan`` per token: ``[start_frame, end_frame)`` and the mean posterior over those
    frames, as ``torchaudio.functional.merge_tokens`` computes them. Checks the guard first; raises
    ``AlignmentFailure`` if it fails or if ``forced_align`` or ``merge_tokens`` raises.
    """
    import torchaudio.functional as taf

    frames = int(emission.shape[0])
    check_guard(frames, ids)
    facts: dict[str, object] = {"frames": frames, "tokens": len(ids), "repeats": repeats(ids)}
    try:
        targets = torch.tensor([list(ids)], dtype=torch.int32)
        path, scores = taf.forced_align(emission.unsqueeze(0), targets, blank=blank_id)
        spans = taf.merge_tokens(path[0], scores[0].exp(), blank=blank_id)
    except Exception as exc:
        raise AlignmentFailure(
            f"forced alignment raised {type(exc).__name__}: {str(exc)[:500]}",
            {"reason": "forced_align_raised", "type": type(exc).__name__, **facts},
        ) from exc
    if len(spans) != len(ids) or any(int(s.token) != t for s, t in zip(spans, ids, strict=False)):
        raise AlignmentFailure(
            f"forced alignment returned {len(spans)} spans for {len(ids)} tokens",
            {"reason": "span_mismatch", "spans": len(spans), **facts},
        )
    return [
        {"token_index": i, "start_frame": int(s.start), "end_frame": int(s.end), "score": float(s.score)}
        for i, s in enumerate(spans)
    ]


# ---------------------------------------------------------------------- the model


def snapshot_vocabulary(revision: str, snapshot_dir: str | Path, device: str = DEVICE) -> tuple[Path, dict[str, int]]:
    """Check a snapshot before the model loads from it; returns its directory and its ``vocab.json``.

    ``InvalidRequest`` for a device other than the CPU or a directory not named by ``revision`` (section 4);
    ``BackendMissing`` for a missing directory, a missing vocabulary, or a vocabulary with no blank.
    """
    if device != DEVICE:
        raise InvalidRequest(
            f"the aligner runs on the CPU only, not {device!r} (design section 11.2)", {"field": "device"}
        )
    path = Path(snapshot_dir)
    if not path.is_dir():
        raise BackendMissing(f"the aligner's snapshot directory is missing: {path}", {"path": str(path)})
    if path.name != revision:
        raise InvalidRequest(
            f"the snapshot directory {path.name!r} is not named by the revision {revision!r} (section 4)",
            {"field": "snapshot_dir", "revision": revision},
        )
    vocab_path = path / "vocab.json"
    if not vocab_path.is_file():
        raise BackendMissing(f"the aligner's vocabulary is missing: {vocab_path}", {"path": str(vocab_path)})
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    if not isinstance(vocab, dict) or BLANK not in vocab:
        raise BackendMissing(f"the aligner's vocabulary has no blank token {BLANK!r}", {"path": str(vocab_path)})
    return path, {str(k): int(v) for k, v in vocab.items()}


class Wav2Vec2Aligner:
    """wav2vec2 CTC emissions on the CPU and forced alignment of a token sequence.

    ``torch`` is the imported module, with the thread cap already applied (``WorkerHandler.torch()``).
    """

    def __init__(self, torch: Any) -> None:
        self._torch = torch
        self._model: Any = None
        self._extractor: Any = None
        self._vocab: dict[str, int] = {}
        self._repo = ""
        self._revision = ""

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self, repo: str, revision: str, snapshot_dir: str | Path, device: str = DEVICE) -> float:
        """Load the model and its vocabulary from a snapshot directory, offline; returns the seconds taken.

        ``snapshot_dir`` must be the directory named by ``revision`` (section 4). ``BackendMissing`` if it or
        a file the model needs is absent; ``InvalidRequest`` for a device other than the CPU or a directory
        not named by the revision.
        """
        path, vocab = snapshot_vocabulary(revision, snapshot_dir, device)
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        start = time.perf_counter()
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(path), local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(str(path), local_files_only=True, dtype=self._torch.float32)
        model.eval()
        self._extractor, self._model = extractor, model
        self._vocab = vocab
        self._repo, self._revision = repo, revision
        return time.perf_counter() - start

    def unload(self) -> None:
        self._model = None
        self._extractor = None
        self._vocab = {}

    def emission(self, audio_16k: npt.NDArray[np.float32]) -> Any:
        """Log-probabilities, frames × classes, for mono 16 kHz audio (one pass; no chunking)."""
        if self._model is None or self._extractor is None:
            raise NotLoaded("the aligner is not loaded; send load first")
        torch = self._torch
        features = self._extractor(audio_16k, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            logits = self._model(features.input_values, attention_mask=features.get("attention_mask")).logits
        return torch.log_softmax(logits[0].float(), dim=-1)

    def align(self, audio: npt.NDArray[np.floating[Any]], sample_rate: int, tokens: Sequence[str]) -> dict[str, Any]:
        """The ``AlignReply`` members other than ``id`` and ``ok``, for audio at any rate and channel count."""
        if self._model is None:
            raise NotLoaded("the aligner is not loaded; send load first")
        ids = token_ids(tokens, self._vocab, wildcard_id=len(self._vocab))
        mono = to_mono_16k(audio, sample_rate)
        check_guard(ctc_frames(mono.shape[0]), ids)
        emission = self.emission(mono)
        return {
            "frame_s": FRAME_S,
            "num_frames": int(emission.shape[0]),
            "spans": self.spans(emission, tokens),
            "model": self._repo,
            "revision": self._revision,
            "device": DEVICE,
        }

    def spans(self, emission: Any, tokens: Sequence[str]) -> list[dict[str, Any]]:
        """The ``TokenSpan``s of ``tokens`` through ``emission`` (from ``emission``), wildcards included.

        The wildcard's column is added only when a ``*`` is among the tokens; the other tokens' spans do not
        depend on it, since forced alignment only ever scores the target tokens and the blank.
        """
        if self._model is None:
            raise NotLoaded("the aligner is not loaded; send load first")
        classes = int(emission.shape[1])
        if classes != len(self._vocab):
            raise BackendMissing(
                f"the aligner's vocabulary has {len(self._vocab)} entries but its model emits {classes} classes: "
                "the snapshot is damaged; install the models again",
                {"vocabulary": len(self._vocab), "classes": classes},
            )
        ids = token_ids(tokens, self._vocab, wildcard_id=classes)
        blank = self._vocab[BLANK]
        if WILDCARD in tokens:
            emission = with_wildcard(self._torch, emission, blank)
        return align_emission(self._torch, emission, ids, blank)

    def align_file(self, path: Path, tokens: Sequence[str]) -> dict[str, Any]:
        """``align`` for a WAV file (the ``align`` op's ``wav``)."""
        if self._model is None:
            raise NotLoaded("the aligner is not loaded; send load first")
        audio, rate = read_audio(path)
        return self.align(audio, rate, tokens)


# ---------------------------------------------------------------------- the align op


class AlignOp:
    """The QA role's ``align`` op, and the aligner's share of ``load`` and ``unload`` (design Appendix A).

    The QA handler (WP22) owns one and delegates to it::

        class QaHandler(WorkerHandler):
            role = "qa"

            def __init__(self, context: WorkerContext) -> None:
                super().__init__(context)
                self.aligner = AlignOp(self)

            def op_load(self, request):  # after checking that every snapshot directory exists
                ...
                load_s += self.aligner.load(request["models"]["aligner"])

            def op_align(self, request):
                return self.aligner.handle(request)

            def shutdown(self) -> None:  # op_unload calls it too
                self.aligner.unload()
                ...

    The aligner always runs on the CPU (section 11.2), whatever ``device`` the ``load`` request names for the
    role's other models. torch comes from ``handler.torch()``, with the CPU thread cap applied (section 4.1).
    ``factory`` builds the aligner from torch; tests replace it.
    """

    def __init__(self, handler: WorkerHandler, factory: Callable[[Any], Wav2Vec2Aligner] = Wav2Vec2Aligner) -> None:
        self._handler = handler
        self._factory = factory
        self._aligner: Wav2Vec2Aligner | None = None

    @property
    def loaded(self) -> bool:
        """Whether ``align`` can run."""
        return self._aligner is not None and self._aligner.loaded

    def load(self, ref: object, field: str = "models.aligner") -> float:
        """Load the aligner from a ``ModelRef`` {repo, revision, snapshot_dir}; returns the seconds taken.

        ``INVALID_REQUEST`` naming ``field`` for a malformed ref or a snapshot directory not named by its
        revision; ``BACKEND_NOT_INSTALLED`` for a missing snapshot, a missing vocabulary, or no torch. A load
        that fails leaves the aligner unloaded.
        """
        if not isinstance(ref, Mapping):
            raise OpError(
                "INVALID_REQUEST", f"{field} must be an object {{repo, revision, snapshot_dir}}", {"field": field}
            )
        model_ref: Mapping[str, Any] = ref
        try:
            repo, revision, snapshot_dir = (require_str(model_ref, k) for k in ("repo", "revision", "snapshot_dir"))
        except OpError as exc:
            member = (exc.details or {}).get("field")
            raise OpError(exc.code, f"{field}: {exc.message}", {"field": f"{field}.{member}"}) from exc
        self.unload()
        torch = self._handler.torch()
        if torch is None:
            raise OpError(
                "BACKEND_NOT_INSTALLED",
                "torch is not installed in the QA worker's environment: sync workers/qa",
                {"field": field, "package": "torch"},
            )
        aligner = self._factory(torch)
        try:
            load_s = aligner.load(repo, revision, snapshot_dir)
        except AlignerError as exc:
            raise op_error(exc) from exc
        self._aligner = aligner
        return load_s

    def unload(self) -> None:
        """Release the model; ``align`` replies ``NOT_LOADED`` until the next ``load``."""
        if self._aligner is not None:
            self._aligner.unload()
        self._aligner = None

    def handle(self, request: Request) -> dict[str, Any]:
        """The ``align`` op: the ``AlignReply`` members other than ``id`` and ``ok``.

        ``NOT_LOADED`` comes first, before any member is read. Then ``wav`` must be an absolute path inside the
        store naming a readable file, and ``tokens`` a list of labels in the model's alphabet. The guard or
        ``forced_align`` failing is ``ALIGNMENT_ERROR``, with ``details`` {reason, frames, tokens, repeats}.
        """
        aligner = self._aligner
        if aligner is None or not aligner.loaded:
            raise OpError("NOT_LOADED", "align needs the aligner loaded: send load first")
        path = self._handler.input_file(request, "wav")
        tokens = require_str_list(request, "tokens")
        try:
            return aligner.align_file(path, tokens)
        except AlignerError as exc:
            raise op_error(exc) from exc


# ---------------------------------------------------------------------- development entry point


def _job_audio(job: Mapping[str, Any]) -> tuple[npt.NDArray[np.float32], int]:
    audio, rate = read_audio(Path(str(job["wav"])))
    start = job.get("start_s")
    end = job.get("end_s")
    first = 0 if start is None else max(0, round(float(start) * rate))
    last = audio.shape[0] if end is None else min(audio.shape[0], round(float(end) * rate))
    return audio[first:last], rate


def main(argv: Sequence[str] | None = None) -> int:
    """Align a JSON list of jobs ``{wav, tokens, start_s?, end_s?}`` and print one reply per job as JSON.

    Development only (spike (b), the evidence tests). A slice of a file is aligned in memory; nothing is
    written but ``--out``. The model loads offline, and the CPU thread cap is applied before torch is
    imported. numpy is loaded with this module, so a caller that wants numpy capped too sets the thread
    variables in the environment it starts this with (as the evidence tests do).
    """
    parser = argparse.ArgumentParser(prog="python -m narration_worker_qa.align", description=main.__doc__)
    parser.add_argument("--snapshot", required=True, help="the model's snapshot directory, named by its revision")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--jobs", required=True, help="a JSON file: a list of {wav, tokens, start_s?, end_s?}")
    parser.add_argument("--threads", type=int, default=4, help="the CPU thread cap (design section 4.1)")
    parser.add_argument("--out", help="write the replies to this JSON file instead of printing them")
    args = parser.parse_args(argv)

    cap_threads_env(args.threads)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # the aligner never touches a GPU
    import torch

    cap_torch_threads(args.threads, torch)

    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    aligner = Wav2Vec2Aligner(torch)
    load_s = aligner.load(args.repo, args.revision, args.snapshot)
    replies: list[dict[str, Any]] = []
    for job in jobs:
        began = time.perf_counter()
        try:
            audio, rate = _job_audio(job)
            reply: dict[str, Any] = {"ok": True, **aligner.align(audio, rate, list(job["tokens"]))}
        except AlignerError as exc:
            reply = {"ok": False, "error": {"code": exc.code, "message": exc.message, "details": exc.details}}
        reply["align_s"] = round(time.perf_counter() - began, 3)
        replies.append(reply)
    text = json.dumps({"load_s": round(load_s, 3), "replies": replies}, ensure_ascii=False) + "\n"
    if args.out is None:
        sys.stdout.write(text)
        return 0
    out = Path(args.out)
    partial = out.with_name(out.name + ".tmp")
    partial.write_text(text, encoding="utf-8", newline="\n")
    os.replace(partial, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
