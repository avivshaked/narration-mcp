"""The worker protocol: daemon ↔ worker messages (design Appendix A; plan.md P1).

**Framing.** JSON lines over the worker's stdin and stdout, UTF-8, one object per line. The worker's stdout
carries protocol messages only; its logs go to stderr. Every request has an integer ``id`` and an ``op``;
the reply echoes the ``id`` and has ``ok``. A failed request replies ``{"id", "ok": false, "error":
{"code", "message", "details"?}}`` with a code from ``WORKER_ERROR_CODES``.

**Model runners only.** A worker loads models, runs them and returns raw outputs: audio (as files),
transcripts with word times, embeddings, token alignments, f0 contours, profile numbers. Every verdict,
threshold, flag and key is computed in the server package.

**Files.** Audio travels as absolute paths under ``<store_root>/scratch/``. A caller's voice clip is
copied into the store before a worker sees it. The daemon computes every hash. All times are seconds.

The request and reply shapes are ``TypedDict``s, so the daemon's client and each worker type-check
against one definition. The ``fake`` role (plan.md WP16) implements every op with deterministic synthetic
outputs and must pass the same contract tests as the real workers.
"""

from __future__ import annotations

from typing import Final, Literal, NotRequired, TypedDict

from .names import WorkerRole

PROTOCOL_VERSION: Final = 1
MAX_LINE_BYTES: Final = 64 * 1024 * 1024
"""The largest message line either side accepts (an embedding or alignment reply is well under this)."""

# ---------------------------------------------------------------- ops per role
COMMON_OPS: Final = ("hello", "load", "unload", "shutdown")
QWEN3_OPS: Final = (*COMMON_OPS, "prepare_voice", "synthesize", "design")
QA_OPS: Final = (*COMMON_OPS, "transcribe", "embed", "f0", "align", "profile")
FAKE_OPS: Final = tuple(dict.fromkeys((*QWEN3_OPS, *QA_OPS)))
OPS_BY_ROLE: Final[dict[WorkerRole, tuple[str, ...]]] = {"qwen3": QWEN3_OPS, "qa": QA_OPS, "fake": FAKE_OPS}

Op = Literal[
    "hello",
    "load",
    "unload",
    "shutdown",
    "prepare_voice",
    "synthesize",
    "design",
    "transcribe",
    "embed",
    "f0",
    "align",
    "profile",
]

# ---------------------------------------------------------------- error codes a worker may reply with
WORKER_ERROR_CODES: Final = (
    "GPU_OOM",  # CUDA out of memory; the daemon unloads, waits, retries once (section 4)
    "BACKEND_NOT_INSTALLED",  # a model snapshot or package is missing
    "NOT_LOADED",  # an op that needs a model before load
    "VOICE_NOT_PREPARED",  # synthesize for a voice_hash without prepare_voice
    "ALIGNMENT_ERROR",  # forced alignment raised or its guard failed (T < L + R)
    "UNSUPPORTED_AUDIO",  # a file the worker cannot read
    "INVALID_REQUEST",  # a malformed request or an unknown op
    "RENDER_FAILED",  # generation raised for another reason
    "INTERNAL",  # anything else; details carry the exception type
)
WorkerErrorCode = Literal[
    "GPU_OOM",
    "BACKEND_NOT_INSTALLED",
    "NOT_LOADED",
    "VOICE_NOT_PREPARED",
    "ALIGNMENT_ERROR",
    "UNSUPPORTED_AUDIO",
    "INVALID_REQUEST",
    "RENDER_FAILED",
    "INTERNAL",
]


class WorkerError(TypedDict):
    code: WorkerErrorCode
    message: str
    details: NotRequired[dict[str, object]]


class Request(TypedDict):
    id: int
    op: Op


class Reply(TypedDict):
    id: int
    ok: bool
    error: NotRequired[WorkerError]


# ---------------------------------------------------------------- hello
class Controls(TypedDict):
    pace: bool
    context: bool
    instruct: bool


class Capabilities(TypedDict):
    ops: list[str]
    controls: NotRequired[Controls]


class Fingerprint(TypedDict):
    """What the worker observes of its environment (section 10.1, ENGINE_DRIFT)."""

    python: str
    platform: str
    packages: dict[str, str]
    cuda: str | None
    cudnn: str | None
    gpu: str | None
    driver: str | None
    cpu_threads: int
    env: dict[str, str]


class HelloReply(Reply):
    role: WorkerRole
    protocol: int
    capabilities: Capabilities
    fingerprint: Fingerprint


# ---------------------------------------------------------------- load / unload / shutdown
class DeterminismSettings(TypedDict):
    tf32: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    deterministic_algorithms: Literal["warn_only", "on", "off"]


class Generation(TypedDict, total=False):
    """Audio-changing sampling values, always passed explicitly (section 10.1)."""

    do_sample: bool
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    subtalker_dosample: bool
    subtalker_top_k: int
    subtalker_top_p: float
    subtalker_temperature: float
    max_new_tokens: int


class QwenSettings(TypedDict):
    non_streaming_mode: bool
    generation: Generation


class ModelRef(TypedDict):
    """A model snapshot: its repo, the 40-hex revision, and the local snapshot directory named by it."""

    repo: str
    revision: str
    snapshot_dir: str


class LoadRequest(Request):
    """Qwen: ``model``, ``engine_profile_id``, ``determinism``, ``settings``, ``dtype``, ``attn_implementation``.
    QA: ``models`` by use (``asr``, ``sv``, ``aligner``) and ``device``."""

    device: str
    model: NotRequired[ModelRef]
    engine_profile_id: NotRequired[str]
    dtype: NotRequired[str]
    attn_implementation: NotRequired[str]
    determinism: NotRequired[DeterminismSettings]
    settings: NotRequired[QwenSettings]
    models: NotRequired[dict[str, ModelRef]]


class LoadReply(Reply):
    load_s: float
    vram_mb: int | None


# ---------------------------------------------------------------- qwen3: prepare_voice / synthesize / design
class PrepareVoiceRequest(Request):
    voice_hash: str
    ref_wav: str
    ref_text: str
    x_vector_only_mode: bool


class SynthesizeRequest(Request):
    voice_hash: str
    engine_text: str
    language: str
    seed: int
    out_path: str


class AudioReply(Reply):
    """A generated WAV written to ``out_path``: float32 mono at ``sample_rate``."""

    sample_rate: int
    samples: int
    gen_s: float
    hit_token_cap: bool
    new_tokens: NotRequired[int]


class DesignRequest(Request):
    description: str
    design_text: str
    language: str
    seed: int
    out_path: str


# ---------------------------------------------------------------- qa: transcribe / embed / f0 / align / profile
class TranscribeRequest(Request):
    """Whisper-large-v3: fp16, English, greedy, word timestamps, sequential long-form (section 11.1)."""

    wav: str
    language: str
    word_timestamps: bool
    long_form: bool


class AsrWord(TypedDict):
    text: str
    start_s: float | None
    end_s: float | None
    probability: NotRequired[float | None]


class TranscribeReply(Reply):
    text: str
    words: list[AsrWord]
    model: str
    revision: str


class EmbedRequest(Request):
    wav: str
    device: Literal["cuda", "cpu"]


class EmbedReply(Reply):
    """A WavLM-SV x-vector, L2-normalised."""

    embedding: list[float]
    dim: int
    model: str
    revision: str


class F0Request(Request):
    wav: str
    fmin_hz: float
    fmax_hz: float


class F0Reply(Reply):
    hop_s: float
    f0_hz: list[float | None]
    voiced_probability: list[float]
    method: str


class AlignRequest(Request):
    """CTC forced alignment of a token sequence the server built (plan.md WP15).

    ``tokens`` are labels in the aligner model's alphabet (letters, apostrophe, ``|`` between words). The
    server keeps the token → (cue, word) map; the worker only aligns.
    """

    wav: str
    tokens: list[str]


class TokenSpan(TypedDict):
    """One token's frames from ``merge_tokens``: [start_frame, end_frame), and its mean posterior."""

    token_index: int
    start_frame: int
    end_frame: int
    score: float


class AlignReply(Reply):
    """On the guard failing (frames < tokens + repeats) or ``forced_align`` raising, the worker replies
    ``ok: false`` with ``ALIGNMENT_ERROR`` and ``details`` {reason, frames, tokens, repeats}."""

    frame_s: float
    num_frames: int
    spans: list[TokenSpan]
    model: str
    revision: str
    device: str


class ProfileRequest(Request):
    wav: str
    out_dir: str
    transcript: NotRequired[str | None]


class ProfileReply(Reply):
    """The measurements of ``models.ProfileMeasurements`` and the pictures written to ``out_dir``."""

    measurements: dict[str, float | None]
    pictures: dict[str, str]
    method: dict[str, str]
