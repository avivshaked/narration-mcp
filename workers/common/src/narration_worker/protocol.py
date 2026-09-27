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
against one definition. This module is that definition. It lives in ``narration_worker`` (standard
library only) because the worker venvs carry this package and never the ``narration`` server package;
the server re-exports it as ``narration.contracts.worker``. It is a contract: only the lead changes it.

The ``fake`` role (plan.md WP16) implements every op with deterministic synthetic outputs and must pass
the same contract tests as the real workers.
"""

# No ``from __future__ import annotations`` here: it would hide ``NotRequired`` from ``TypedDict`` at run
# time, so ``__required_keys__`` would list every key (WP16's review).
from typing import Final, Literal, NotRequired, TypedDict

WorkerRole = Literal["qwen3", "qa", "fake"]
"""The worker roles: the Qwen3-TTS worker, the QA worker, and the deterministic fake of plan.md WP16."""

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
    """Clone the prepared ``voice_hash`` speaking ``engine_text``.

    ``max_new_tokens`` is this call's own generation cap (design section 10.1, DC-4). The daemon computes it
    from the text the call speaks with ``narration.contracts.names.max_new_tokens_for``, so a runaway render
    stops early. It is required and must be from 2 (qwen-tts fixes ``min_new_tokens`` at 2) to the loaded
    ceiling (``load``'s ``settings.generation.max_new_tokens``); anything else is ``INVALID_REQUEST``.
    """

    voice_hash: str
    engine_text: str
    language: str
    seed: int
    max_new_tokens: int
    out_path: str


class AudioReply(Reply):
    """A generated WAV written to ``out_path``: float32 mono at ``sample_rate``.

    Generation runs in talker steps. Each step samples either one codec frame or the end token, and the
    frames are decoded at 12.5 per second of audio. The call's ``max_new_tokens`` bounds the steps.

    - ``hit_token_cap``: the last sampled token was not the end token, so generation stopped at the cap
      (mid-text, or in a runaway). An end token on the cap-th step is **not** a hit. ``TOKEN_CAP_HIT`` is
      judged from this flag only, never by comparing ``new_tokens`` with the cap.
    - ``new_tokens``: the decoded frames, which is the talker steps minus one, so at most
      ``max_new_tokens`` - 1. A render whose end token comes at step N has N - 1 frames and is not a hit
      under any cap from N up; under a cap of N - 1 it is a hit with N - 2 frames.
    - ``max_new_tokens``: the cap the worker applied, echoed from the request.
    """

    sample_rate: int
    samples: int
    gen_s: float
    hit_token_cap: bool
    max_new_tokens: int
    new_tokens: NotRequired[int]


class DesignRequest(Request):
    """Design a voice from ``description``, speaking ``design_text`` (VoiceDesign). ``max_new_tokens`` is the
    call's own cap, exactly as for ``SynthesizeRequest``."""

    description: str
    design_text: str
    language: str
    seed: int
    max_new_tokens: int
    out_path: str


# ---------------------------------------------------------------- qa: transcribe / embed / f0 / align / profile
class TranscribeRequest(Request):
    """Whisper-large-v3: fp16, English, five beams not conditioned on the previous window and no temperature
    fallback (DC-14, ADR 0004), word timestamps, sequential long-form (section 11.1)."""

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

    ``tokens`` are labels in the aligner model's alphabet (letters, apostrophe, ``|`` between words), and
    ``*``, a wildcard for a run of words the alphabet cannot spell (plan.md DC-11): the worker aligns it
    through an extra emission column, log(1 − P(blank)) per frame. The server keeps the token → (cue, word)
    map; the worker only aligns.
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
