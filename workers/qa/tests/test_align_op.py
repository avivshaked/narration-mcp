"""The QA worker's ``align`` op (design Appendix A; plan.md WP15 and WP16): ``AlignOp`` in a handler.

The QA role's handler is WP22's, so these tests serve the op from a handler that has only the aligner. The
default tests replace the model with a stub (blank-heavy emissions), so the whole op runs with no model:
the request's members, the store's paths, reading the file, the guard, forced alignment and the reply. The
``model`` test runs the op with the pinned wav2vec2 snapshot.
"""

from __future__ import annotations

import io
import itertools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import soundfile as sf
from narration_worker.errors import OpError
from narration_worker.handler import Request, WorkerContext, WorkerHandler
from narration_worker.loop import serve
from narration_worker.protocol import AlignReply, Reply
from narration_worker_qa import align as qa

torch = pytest.importorskip("torch", reason="needs torch and transformers: the QA worker's full venv")
pytest.importorskip("transformers", reason="needs torch and transformers: the QA worker's full venv")

REPO = "facebook/wav2vec2-large-960h-lv60-self"
REVISION = "54074b1c16f4de6a5ad59affb4caa8f2ea03a119"
PINNED_VOCABULARY = ("<pad>", "<s>", "</s>", "<unk>", "|", *"ETAONIHSRDLUMWCFGYPBVK'XJQZ")
REPLY_MEMBERS = set(AlignReply.__annotations__) - set(Reply.__annotations__)


class _StubAligner(qa.Wav2Vec2Aligner):
    """The real aligner with the model replaced: the blank is likeliest in every frame."""

    def load(self, repo: str, revision: str, snapshot_dir: str | Path, device: str = qa.DEVICE) -> float:
        _, self._vocab = qa.snapshot_vocabulary(revision, snapshot_dir, device)
        self._model, self._extractor = object(), object()
        self._repo, self._revision = repo, revision
        return 0.0

    def emission(self, audio_16k: npt.NDArray[np.float32]) -> Any:
        logits = torch.zeros(qa.ctc_frames(audio_16k.shape[0]), len(self._vocab))
        logits[:, 0] = 4.0
        return torch.log_softmax(logits, dim=-1)


class _AlignHandler(WorkerHandler):
    """A QA handler with only the aligner: what WP22's handler delegates to ``AlignOp``."""

    role = "qa"

    def __init__(self, context: WorkerContext, factory: Any = _StubAligner) -> None:
        super().__init__(context)
        self.aligner = qa.AlignOp(self, factory)

    def op_load(self, request: Request) -> dict[str, Any]:
        return {"load_s": self.aligner.load(request["models"]["aligner"]), "vram_mb": None}

    def op_unload(self, request: Request) -> dict[str, Any]:
        self.shutdown()
        return {}

    def op_align(self, request: Request) -> dict[str, Any]:
        return self.aligner.handle(request)

    def shutdown(self) -> None:
        self.aligner.unload()


class _NoTorchHandler(_AlignHandler):
    uses_torch = False


@pytest.fixture
def store(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    (root / "scratch").mkdir(parents=True)
    return root


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    path = tmp_path / "models" / REVISION
    path.mkdir(parents=True)
    vocab = {token: i for i, token in enumerate(PINNED_VOCABULARY)}
    (path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    return path


def _handler(store: Path, cls: type[_AlignHandler] = _AlignHandler, **kwargs: Any) -> _AlignHandler:
    return cls(WorkerContext(role="qa", store_root=store, cpu_threads=4), **kwargs)


def _ref(snapshot: Path) -> dict[str, str]:
    return {"repo": REPO, "revision": REVISION, "snapshot_dir": str(snapshot)}


def _wav(store: Path, seconds: float, name: str = "take.wav", rate: int = 48_000) -> Path:
    path = store / "scratch" / name
    sf.write(str(path), np.zeros(round(seconds * rate), dtype=np.float32), rate, subtype="FLOAT")
    return path


def _error(handler: _AlignHandler, op: str, request: Request) -> OpError:
    with pytest.raises(OpError) as caught:
        handler.handle(op, request)
    return caught.value


# ---------------------------------------------------------------------- load / unload


def test_align_before_load_is_not_loaded_before_any_member_is_read_appA(store: Path) -> None:
    handler = _handler(store)
    assert _error(handler, "align", {"wav": "not absolute", "tokens": 5}).code == "NOT_LOADED"
    assert _error(handler, "align", {}).code == "NOT_LOADED"


def test_load_with_a_missing_snapshot_is_backend_not_installed_s14(store: Path, snapshot: Path) -> None:
    handler = _handler(store)
    error = _error(handler, "load", {"models": {"aligner": _ref(snapshot.parent / "gone" / REVISION)}})
    assert error.code == "BACKEND_NOT_INSTALLED"
    assert not handler.aligner.loaded


def test_load_with_a_malformed_ref_is_invalid_request_naming_the_field_appA(store: Path, snapshot: Path) -> None:
    handler = _handler(store)
    error = _error(handler, "load", {"models": {"aligner": "facebook/wav2vec2"}})
    assert (error.code, error.details) == ("INVALID_REQUEST", {"field": "models.aligner"})
    error = _error(handler, "load", {"models": {"aligner": {"repo": REPO, "snapshot_dir": str(snapshot)}}})
    assert (error.code, error.details) == ("INVALID_REQUEST", {"field": "models.aligner.revision"})


def test_load_from_a_directory_not_named_by_the_revision_is_invalid_request_s4(store: Path, snapshot: Path) -> None:
    ref = {**_ref(snapshot), "revision": "f" * 40}
    error = _error(_handler(store), "load", {"models": {"aligner": ref}})
    assert error.code == "INVALID_REQUEST" and error.details is not None
    assert error.details["field"] == "snapshot_dir"


def test_load_without_torch_is_backend_not_installed_s14(store: Path, snapshot: Path) -> None:
    error = _error(_handler(store, _NoTorchHandler), "load", {"models": {"aligner": _ref(snapshot)}})
    assert error.code == "BACKEND_NOT_INSTALLED"


def test_a_failed_load_leaves_the_aligner_unloaded_appA(store: Path, snapshot: Path) -> None:
    handler = _handler(store)
    handler.handle("load", {"models": {"aligner": _ref(snapshot)}})
    assert handler.aligner.loaded
    _error(handler, "load", {"models": {"aligner": _ref(snapshot.parent / "gone" / REVISION)}})
    assert not handler.aligner.loaded
    wav = _wav(store, 1.0)
    assert _error(handler, "align", {"wav": str(wav), "tokens": ["A"]}).code == "NOT_LOADED"


def test_unload_then_align_is_not_loaded_appA(store: Path, snapshot: Path) -> None:
    handler = _handler(store)
    handler.handle("load", {"models": {"aligner": _ref(snapshot)}})
    wav = _wav(store, 1.0)
    assert handler.handle("align", {"wav": str(wav), "tokens": ["A"]})["num_frames"] == 49
    handler.handle("unload", {})
    assert _error(handler, "align", {"wav": str(wav), "tokens": ["A"]}).code == "NOT_LOADED"


# ---------------------------------------------------------------------- align


@pytest.fixture
def loaded(store: Path, snapshot: Path) -> _AlignHandler:
    handler = _handler(store)
    handler.handle("load", {"models": {"aligner": _ref(snapshot)}})
    return handler


def test_align_replies_one_span_per_token_s11_2(loaded: _AlignHandler, store: Path) -> None:
    tokens = list("RAIN|CAME")
    reply = loaded.handle("align", {"wav": str(_wav(store, 1.0)), "tokens": tokens})
    assert set(reply) == REPLY_MEMBERS
    assert (reply["model"], reply["revision"], reply["device"], reply["frame_s"]) == (REPO, REVISION, "cpu", 0.02)
    assert reply["num_frames"] == qa.ctc_frames(16_000)
    assert [s["token_index"] for s in reply["spans"]] == list(range(len(tokens)))
    ends = [(s["start_frame"], s["end_frame"]) for s in reply["spans"]]
    assert all(a < b for a, b in ends) and all(b1 <= a2 for (_, b1), (a2, _) in itertools.pairwise(ends))


def test_a_wildcard_gets_a_span_like_any_token_s11_2_dc11(loaded: _AlignHandler, store: Path) -> None:
    tokens = [*"RAIN", "|", "*", "|", *"CAME"]
    reply = loaded.handle("align", {"wav": str(_wav(store, 1.0)), "tokens": tokens})
    assert [s["token_index"] for s in reply["spans"]] == list(range(len(tokens)))


def test_a_vocabulary_that_does_not_fit_the_model_is_backend_not_installed_s11_2(store: Path, snapshot: Path) -> None:
    class Narrow(_StubAligner):
        def emission(self, audio_16k: npt.NDArray[np.float32]) -> Any:
            return super().emission(audio_16k)[:, :-1]

    handler = _handler(store, factory=Narrow)
    handler.handle("load", {"models": {"aligner": _ref(snapshot)}})
    error = _error(handler, "align", {"wav": str(_wav(store, 1.0)), "tokens": ["A", "|", "*"]})
    assert error.code == "BACKEND_NOT_INSTALLED"


def test_too_short_audio_is_alignment_error_with_the_counts_s11_2(loaded: _AlignHandler, store: Path) -> None:
    tokens = list("WINDS|CAME|OVER|THE|RIDGE")  # invented: 25 tokens, no repeat
    error = _error(loaded, "align", {"wav": str(_wav(store, 0.1)), "tokens": tokens})
    assert error.code == "ALIGNMENT_ERROR"
    assert error.details == {"reason": "too_short", "frames": 4, "tokens": 25, "repeats": 0}


def test_no_tokens_is_alignment_error_s11_2(loaded: _AlignHandler, store: Path) -> None:
    error = _error(loaded, "align", {"wav": str(_wav(store, 1.0)), "tokens": []})
    assert error.code == "ALIGNMENT_ERROR" and error.details is not None
    assert error.details["reason"] == "no_tokens"


@pytest.mark.parametrize("tokens", [["a"], ["<pad>"], ["AB"], ["A", "1"]])
def test_a_token_outside_the_alphabet_is_invalid_request_s11_2(
    loaded: _AlignHandler, store: Path, tokens: list[str]
) -> None:
    error = _error(loaded, "align", {"wav": str(_wav(store, 1.0)), "tokens": tokens})
    assert error.code == "INVALID_REQUEST" and error.details is not None
    assert error.details["field"] == "tokens"


def test_tokens_that_are_not_a_list_of_strings_are_invalid_request_appA(loaded: _AlignHandler, store: Path) -> None:
    wav = str(_wav(store, 1.0))
    for tokens in ("RAIN", [1, 2], None):
        error = _error(loaded, "align", {"wav": wav, "tokens": tokens})
        assert (error.code, error.details) == ("INVALID_REQUEST", {"field": "tokens"})
    assert _error(loaded, "align", {"wav": wav}).details == {"field": "tokens"}


def test_wav_must_be_an_absolute_path_inside_the_store_appA(loaded: _AlignHandler, store: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.wav"
    sf.write(str(outside), np.zeros(16_000, dtype=np.float32), 16_000)
    for wav in (str(outside), "scratch/take.wav"):
        error = _error(loaded, "align", {"wav": wav, "tokens": ["A"]})
        assert error.code == "INVALID_REQUEST" and error.details is not None
        assert error.details["field"] == "wav"


def test_a_missing_or_unreadable_wav_is_unsupported_audio_appA(loaded: _AlignHandler, store: Path) -> None:
    missing = store / "scratch" / "missing.wav"
    assert _error(loaded, "align", {"wav": str(missing), "tokens": ["A"]}).code == "UNSUPPORTED_AUDIO"
    junk = store / "scratch" / "junk.wav"
    junk.write_bytes(b"not a wav file")
    assert _error(loaded, "align", {"wav": str(junk), "tokens": ["A"]}).code == "UNSUPPORTED_AUDIO"


def test_the_request_loop_replies_with_the_ops_results_appA(store: Path, snapshot: Path) -> None:
    handler = _handler(store)
    wav = str(_wav(store, 1.0))
    requests = [
        {"id": 1, "op": "align", "wav": wav, "tokens": ["A"]},
        {"id": 2, "op": "load", "models": {"aligner": _ref(snapshot)}},
        {"id": 3, "op": "align", "wav": wav, "tokens": list("RAIN")},
        {"id": 4, "op": "align", "wav": str(_wav(store, 0.05, "short.wav")), "tokens": list("RAIN")},
        {"id": 5, "op": "shutdown"},
    ]
    reader = io.BytesIO(b"".join(json.dumps(r).encode("utf-8") + b"\n" for r in requests))
    writer = io.BytesIO()
    assert serve(handler, reader, writer) == 0
    replies = [json.loads(line) for line in writer.getvalue().splitlines()]
    assert [(r["id"], r["ok"]) for r in replies] == [(1, False), (2, True), (3, True), (4, False), (5, True)]
    assert replies[0]["error"]["code"] == "NOT_LOADED"
    assert set(replies[2]) - {"id", "ok"} == REPLY_MEMBERS
    assert replies[3]["error"] == {
        "code": "ALIGNMENT_ERROR",
        "message": replies[3]["error"]["message"],
        "details": {"reason": "too_short", "frames": 2, "tokens": 4, "repeats": 0},
    }
    assert not handler.aligner.loaded


# ---------------------------------------------------------------------- with the pinned model (CPU)


@pytest.mark.model
def test_align_op_with_the_pinned_model_s11_2(store: Path) -> None:
    root = os.environ.get("NARRATION_MODELS_ROOT")
    if not root:
        pytest.skip("NARRATION_MODELS_ROOT is not set: the pinned wav2vec2 snapshot is needed")
    snapshot = Path(root) / "models--facebook--wav2vec2-large-960h-lv60-self" / "snapshots" / REVISION
    if not snapshot.is_dir():
        pytest.skip("the pinned wav2vec2 snapshot is not installed under NARRATION_MODELS_ROOT")
    handler = _handler(store, factory=qa.Wav2Vec2Aligner)
    loaded = handler.handle("load", {"models": {"aligner": _ref(snapshot)}})
    assert loaded["load_s"] >= 0
    reply = handler.handle("align", {"wav": str(_wav(store, 1.0)), "tokens": list("RAIN|CAME")})
    assert set(reply) == REPLY_MEMBERS and reply["model"] == REPO and reply["revision"] == REVISION
    assert len(reply["spans"]) == 9
    handler.handle("unload", {})
    assert not handler.aligner.loaded
