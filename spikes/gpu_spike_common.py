"""Helpers shared by WP20's Phase 0 GPU spikes (design section 20: d, e, f, h, i).

The spikes run in a worker venv (``workers/qwen3tts/.venv`` for Qwen, ``workers/qa/.venv`` for Whisper and
WavLM), take the GPU lock first (AGENTS.md section 5), and write audio only under ``<checkout>/.dev/spikes/``
(gitignored). Their committed results are JSON with no local paths and no GPU model name; machine facts
that identify this machine go to a local, gitignored file next to the audio.

Call ``prepare_process_env()`` before anything imports torch, numpy or transformers.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import os
import platform
import socket
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

CHECKOUT = Path(__file__).resolve().parents[1]
DEV_SPIKES = CHECKOUT / ".dev" / "spikes"

CPU_THREADS = 8
"""The design's default ``[workers] cpu_threads`` (section 4.1)."""

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")

MODEL_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
MODEL_ASR = "openai/whisper-large-v3"
MODEL_SV = "microsoft/wavlm-base-plus-sv"
SV_REVISION_EVIDENCE = "feb593a6"
"""The WavLM-SV snapshot whose revision starts with this: ``main``, the one the bakeoff's evidence used."""

LANGUAGE = "English"

CEILING = 8192
"""The pinned snapshots' ``max_new_tokens``: the loaded ceiling, and the cap every spike render used (the cap
only truncates, ADR 0003, so a render that ends under it is the same under any higher cap)."""

# The two voices the bakeoff designed (design section 16; plan.md 1.2). Their clips live in the bakeoff
# (read-only) and are read from there, never copied into the repository.
VOICES: Mapping[str, str] = {
    "d2": "refs/auditions/qwen3-tts-voicedesign_d2-late-night_take1.wav",
    "d4": "refs/auditions/qwen3-tts-voicedesign_d4-radio-drama_take2.wav",
}
PER_CHAR = 2.5
FLOOR = 128
"""The engines' shipped ``max_new_tokens_per_char`` and ``max_new_tokens_floor`` (``narration.example.toml``)."""
ALLOW_ENV = "NARRATION_SPIKE_ALLOW_SHA256"
"""The owner's allowlist for the spikes, as ``[voices] allow_sha256`` is for the service (section 17.4): the
sha256 of each clip a spike may clone, separated by commas or spaces. It is a local fact, so it is never
written into the repository."""


def prepare_process_env(cpu_threads: int = CPU_THREADS) -> None:
    """The worker's start-up environment (sections 4, 4.1, 10.1), set before torch or transformers load."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(cpu_threads)


def cap_torch_threads(torch: Any, cpu_threads: int = CPU_THREADS) -> None:
    torch.set_num_threads(cpu_threads)
    with contextlib.suppress(RuntimeError):  # torch allows it only once, before any inter-op work
        torch.set_num_interop_threads(min(cpu_threads, 4))


def apply_switches(torch: Any, on: bool) -> dict[str, object]:
    """The determinism switches of section 10.1 (as ``narration_worker.determinism.apply_determinism``), or,
    with ``on`` false, torch's defaults (what the bakeoff ran with). Returns what is in force."""
    torch.backends.cuda.matmul.allow_tf32 = False  # torch's default is already False
    torch.backends.cudnn.allow_tf32 = not on  # torch's default is True
    torch.backends.cudnn.deterministic = on
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(on, warn_only=True)
    return {
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": "warn_only" if on else "off",
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def seed_all(torch: Any, seed: int) -> None:
    """As ``narration_worker.determinism.seed_everything``: random, numpy, torch and torch.cuda."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------- the network guard (spike i)
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


class NetworkGuard:
    """Refuses every connection and name lookup that leaves the machine, and records each attempt.

    Loopback is allowed and recorded apart (``loopback``): on Windows ``socket.socketpair`` connects over
    127.0.0.1, and that never leaves the machine.
    """

    def __init__(self) -> None:
        self.attempts: list[str] = []
        self.loopback: list[str] = []

    def install(self) -> NetworkGuard:
        def host_of(address: object) -> str | None:
            if isinstance(address, tuple) and address and isinstance(address[0], str):
                return address[0]
            return None

        def guarded(kind: str, original: Callable[..., Any], address_arg: int) -> Callable[..., Any]:
            def call(*args: Any, **kwargs: Any) -> Any:
                target = args[address_arg] if len(args) > address_arg else None
                host = target if isinstance(target, str) else host_of(target)
                if host in LOOPBACK_HOSTS:
                    self.loopback.append(f"{kind} {host}")
                    return original(*args, **kwargs)
                self.attempts.append(f"{kind} {target!r}")
                raise OSError(f"network access refused by the spike's guard: {kind} {target!r}")

            return call

        socket.socket.connect = guarded("socket.connect", socket.socket.connect, 1)  # type: ignore[method-assign]
        socket.socket.connect_ex = guarded("socket.connect_ex", socket.socket.connect_ex, 1)  # type: ignore[method-assign]
        socket.create_connection = guarded("socket.create_connection", socket.create_connection, 0)  # type: ignore[assignment]
        socket.getaddrinfo = guarded("socket.getaddrinfo", socket.getaddrinfo, 0)  # type: ignore[assignment]
        return self


# ---------------------------------------------------------------------- models and material
def models_root() -> Path:
    value = os.environ.get("NARRATION_MODELS_ROOT")
    if not value:
        sys.exit("set NARRATION_MODELS_ROOT to the models root (AGENTS.local.md)")
    return Path(value)


def bakeoff_root() -> Path:
    value = os.environ.get("NARRATION_BAKEOFF_ROOT")
    if not value:
        sys.exit("set NARRATION_BAKEOFF_ROOT to the bakeoff's folder (AGENTS.local.md)")
    return Path(value)


def snapshot(repo: str, revision_prefix: str = "") -> tuple[Path, str]:
    """A model's snapshot folder and its 40-hex revision, from the models root's ``manifest.json``."""
    root = models_root()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest.values():
        if entry["repo"] == repo and entry["revision"].startswith(revision_prefix):
            return root / entry["snapshot_dir"], entry["revision"]
    sys.exit(f"{repo} is not in {root / 'manifest.json'}")


def manifest_hashes(repo: str, revision: str) -> dict[str, str]:
    manifest = json.loads((models_root() / "manifest.json").read_text(encoding="utf-8"))
    return dict(manifest[f"{repo}@{revision}"]["files_sha256"])


def material_root(override: str | None) -> Path:
    root = Path(override) if override else CHECKOUT / "material"
    if not (root / "calibration").is_dir():
        sys.exit(f"no service material at {root}: pass --material <folder with calibration/, canary/>")
    return root


def spoken_text(paragraph: Mapping[str, Any]) -> str:
    """A paragraph's spoken text: its cues joined with one space (design section 7.2)."""
    return " ".join(cue["text"] for cue in paragraph["cues"])


def calibration_paragraphs(material: Path) -> dict[str, str]:
    """``narration-en.v1``'s paragraphs by segment id, as spoken text."""
    data = json.loads((material / "calibration" / "narration-en.v1" / "paragraphs.json").read_text(encoding="utf-8"))
    return {p["segment_id"]: spoken_text(p) for p in [*data["calibration"], *data["ladder"]]}


def canary(material: Path) -> dict[str, Any]:
    return json.loads((material / "canary" / "canary.v1" / "canary.json").read_text(encoding="utf-8"))


def call_cap(text: str) -> int:
    """The ``max_new_tokens`` the daemon passes for ``text``: ``narration.contracts.names.max_new_tokens_for``
    with the shipped engine keys and the snapshots' ceiling. Loaded from this checkout by path, because the
    worker venvs do not have the server installed."""
    import importlib.util

    name = "_narration_contract_names"
    names = sys.modules.get(name)
    if names is None:
        path = CHECKOUT / "src" / "narration" / "contracts" / "names.py"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            sys.exit(f"cannot load the contracts' names module from {path}")
        names = importlib.util.module_from_spec(spec)
        sys.modules[name] = names  # its dataclasses look their module up while the class is built
        spec.loader.exec_module(names)
    return int(names.max_new_tokens_for(text, per_char=PER_CHAR, floor=FLOOR, ceiling=CEILING))


def allowed_sha256() -> frozenset[str]:
    """The clips the spikes may clone, from ``$NARRATION_SPIKE_ALLOW_SHA256`` (synthetic voices only, 17.4)."""
    value = os.environ.get(ALLOW_ENV, "")
    digests = frozenset(value.replace(",", " ").lower().split())
    if not digests:
        sys.exit(f"set {ALLOW_ENV} to the sha256 of each clip the spikes may clone (AGENTS.local.md)")
    return digests


def voice_clip(name: str) -> tuple[Path, str, dict[str, Any]]:
    """An allowlisted bakeoff voice: its clip (whose sha256 must be in ``allowed_sha256``), its exact transcript,
    and its sidecar."""
    clip = bakeoff_root() / VOICES[name]
    digest = sha256_file(clip)
    if digest not in allowed_sha256():
        sys.exit(f"{name}: the clip's sha256 is not in {ALLOW_ENV}; a spike clones only an allowlisted clip")
    sidecar = json.loads(clip.with_suffix(".json").read_text(encoding="utf-8"))
    return clip, sidecar["text"], sidecar


# ---------------------------------------------------------------------- measuring
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: Any) -> str:
    """sha256 of an array's bytes (float32 audio as generated: the bit-identity test)."""
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def as_bakeoff_pcm16(audio: Any, sample_rate: int) -> Any:
    """The 16-bit samples the bakeoff would have stored for this float audio.

    The bakeoff wrote every WAV with ``soundfile.write(path, audio, sr)``, whose default WAV subtype is
    PCM_16, so its files hold libsndfile's float-to-16-bit conversion of the float render. This repeats that
    conversion in memory. Comparing its result with the bakeoff's file is the bit-identity test against the
    bakeoff.

    Do not compare by reading a FLOAT WAV with ``dtype="int16"``. libsndfile does not scale float to
    integer on read by default, so every sample comes back as -1, 0 or 1 (KNOW: see
    ``h-i-qwen-load/README.md``, "Correction").
    """
    import io

    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)
    return sf.read(buffer, dtype="int16")[0]


def mb(value: int) -> int:
    return int(value // (1024 * 1024))


def gpu_memory(torch: Any, device: str = "cuda:0") -> dict[str, int]:
    """This process's allocator figures and the device's free memory, in MiB."""
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_mb": mb(torch.cuda.memory_allocated(device)),
        "reserved_mb": mb(torch.cuda.memory_reserved(device)),
        "max_allocated_mb": mb(torch.cuda.max_memory_allocated(device)),
        "max_reserved_mb": mb(torch.cuda.max_memory_reserved(device)),
        "device_free_mb": mb(free),
        "device_total_mb": mb(total),
    }


def software_facts(torch: Any) -> dict[str, object]:
    """What may be published: versions and a generic description of the GPU, never its name or driver."""
    from importlib.metadata import PackageNotFoundError, version

    packages: dict[str, str] = {}
    for name in ("qwen-tts", "transformers", "accelerate", "torch", "torchaudio", "numpy", "soundfile", "scipy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue
    facts: dict[str, object] = {
        "python": platform.python_version(),
        "os": platform.system(),
        "packages": packages,
        "cuda": torch.version.cuda,
        "cudnn": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
    }
    if torch.cuda.is_available():
        total_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3)
        facts["gpu"] = f"one {total_gb} GB consumer NVIDIA GPU, shared with other jobs"
    return facts


def local_facts(torch: Any) -> dict[str, object]:
    """Machine facts kept out of git (the GPU's name, the driver): written only under ``.dev``."""
    facts: dict[str, object] = {"platform": platform.platform()}
    if torch.cuda.is_available():
        facts["gpu_name"] = torch.cuda.get_device_name(0)
    try:
        import pynvml

        pynvml.nvmlInit()
        facts["driver"] = pynvml.nvmlSystemGetDriverVersion()
        pynvml.nvmlShutdown()
    except Exception as exc:  # the driver version is a nicety here
        facts["driver"] = f"unknown ({type(exc).__name__})"
    return facts


def now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def write_json(path: Path, data: object) -> None:
    """Write JSON (UTF-8, not ASCII-escaped) through a temp name and a rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_wav(path: Path, audio: Any, sample_rate: int) -> None:
    """Write float32 audio (WAV FLOAT, as the worker writes raw renders) through a temp name and a rename."""
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    sf.write(str(tmp), audio, sample_rate, subtype="FLOAT", format="WAV")
    os.replace(tmp, path)


@contextmanager
def captured_warnings() -> Iterator[list[str]]:
    """Collect the distinct warnings raised inside the block (e.g. warn_only's non-deterministic ops)."""
    import warnings

    seen: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield seen
    for item in caught:
        text = f"{item.category.__name__}: {str(item.message).splitlines()[0][:300]}"
        if text not in seen:
            seen.append(text)
