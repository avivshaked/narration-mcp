"""The service's configuration (design section 16), loaded from TOML and validated.

The design assumes no install location. ``store_root`` and ``models_root`` are required; every other key
has the design's default. A relative path is resolved against the folder that holds the config file
(``<service_root>`` in the design). Unknown sections and keys are refused, so a typo fails loudly instead
of silently falling back to a default. ``allow_sha256`` ships empty: no clip is allowlisted by default.
"""

from __future__ import annotations

import dataclasses
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

from .contracts.errors import ConfigError
from .contracts.names import BENCHMARK, CORPUS, MODEL_ALIGNER, QA_PROFILE, TEXT_CHECKS_VERSION

_SHA256 = re.compile(r"[0-9a-f]{64}")  # used with fullmatch: $ would accept a final newline


@dataclass(frozen=True, slots=True, kw_only=True)
class ServerConfig:
    store_root: Path
    models_root: Path


@dataclass(frozen=True, slots=True, kw_only=True)
class VoicesConfig:
    allow_sha256: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class RetentionConfig:
    retention_days: int = 30
    measurement_retention_days: int = 365


@dataclass(frozen=True, slots=True, kw_only=True)
class DaemonConfig:
    autostart: bool = True
    idle_unload_s: int = 120
    idle_exit_min: int = 15


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkerProject:
    project: Path


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkersConfig:
    """``[workers]``. ``env`` is added to every worker's environment; an operator's table replaces the
    default one. Whatever it says, the daemon always starts workers with ``HF_HUB_OFFLINE=1``,
    ``TRANSFORMERS_OFFLINE=1`` and ``CUBLAS_WORKSPACE_CONFIG`` set (sections 4, 10.1)."""

    cpu_threads: int = 8
    priority: Literal["below_normal", "normal"] = "below_normal"
    env: dict[str, str] = field(
        default_factory=lambda: {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    qwen3: WorkerProject | None = None
    qa: WorkerProject | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class GpuConfig:
    device: str = "cuda:0"
    one_group_at_a_time: bool = True
    min_free_margin_mb: int = 1024
    wait_timeout_min: int = 30


@dataclass(frozen=True, slots=True, kw_only=True)
class LimitsConfig:
    max_segments_per_job: int = 200
    max_cues_per_segment: int = 40
    max_chars_per_segment: int = 1200
    max_chars_per_job: int = 60000
    max_hints_per_job: int = 500
    max_queued_jobs: int = 20
    max_submits_per_min: int = 10
    max_description_chars: int = 600
    max_clip_seconds: int = 30
    min_free_disk_gb: int = 5


@dataclass(frozen=True, slots=True, kw_only=True)
class DefaultsConfig:
    takes: int = 1
    max_retakes: int = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class TextConfig:
    checks: str = TEXT_CHECKS_VERSION
    refuse: tuple[str, ...] = ("[", "]", "<|", "|>")


@dataclass(frozen=True, slots=True, kw_only=True)
class DeliveryConfig:
    sample_rate: int = 48000
    subtype: str = "PCM_24"
    target_lufs: float = -23.0  # plan.md DC-8: EBU R128's pair with -1 dBTP; -16 was never reached on real output
    true_peak_dbtp: float = -1.0
    trim_rel_db: float = -40.0
    trim_floor_dbfs: float = -70.0  # plan.md DC-10: the speech threshold never goes below this
    trim_pad_s: float = 0.08
    fade_s: float = 0.01


@dataclass(frozen=True, slots=True, kw_only=True)
class VoiceDesignConfig:
    design_text: str = (
        "Good bread asks for patience: the dough is mixed, folded and left to rise through the morning. When "
        "the loaves come out golden and crisp, a gentle warmth fills the whole kitchen."
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class MeasurementConfig:
    corpus: str = CORPUS
    seeds: int = 3
    length_ladder_spoken_chars: tuple[int, ...] = (80, 150, 250, 300, 350, 400, 450, 500, 560)
    trend_band_max_chars: int = 300
    pace_tol_min: float = 0.10
    sim_warn_margin: float = 0.01
    sim_fail_floor: float = 0.90


@dataclass(frozen=True, slots=True, kw_only=True)
class AlignmentConfig:
    model: str = MODEL_ALIGNER
    device: str = "cpu"
    disagree_threshold_s: float = 0.25
    low_confidence_below: float = 0.75
    unplaced_below: float = 0.50
    benchmark: str = BENCHMARK


@dataclass(frozen=True, slots=True, kw_only=True)
class QaConfig:
    """``[qa]``. ``profile`` is informational: nothing reads it, and every take is scored with the QA profile
    this build pins (``names.QA_PROFILE``). A value that differs is warned about, never refused
    (``qa_profile_mismatch``)."""

    profile: str = QA_PROFILE


@dataclass(frozen=True, slots=True, kw_only=True)
class QwenBaseConfig:
    non_streaming_mode: bool = False
    x_vector_only_mode: bool = False
    max_new_tokens_per_char: float = 2.5
    max_new_tokens_floor: int = 128


@dataclass(frozen=True, slots=True, kw_only=True)
class QwenDesignConfig:
    non_streaming_mode: bool = True
    max_new_tokens_per_char: float = 2.5
    max_new_tokens_floor: int = 128


@dataclass(frozen=True, slots=True, kw_only=True)
class EnginesConfig:
    qwen3_base: QwenBaseConfig = field(default_factory=QwenBaseConfig)
    qwen3_design: QwenDesignConfig = field(default_factory=QwenDesignConfig)


@dataclass(frozen=True, slots=True, kw_only=True)
class Config:
    """The whole configuration. ``path`` is the file it came from (None for ``Config.for_tests``)."""

    server: ServerConfig
    voices: VoicesConfig = field(default_factory=VoicesConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    workers: WorkersConfig = field(default_factory=WorkersConfig)
    gpu: GpuConfig = field(default_factory=GpuConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    text: TextConfig = field(default_factory=TextConfig)
    delivery: DeliveryConfig = field(default_factory=DeliveryConfig)
    voice_design: VoiceDesignConfig = field(default_factory=VoiceDesignConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    qa: QaConfig = field(default_factory=QaConfig)
    engines: EnginesConfig = field(default_factory=EnginesConfig)
    path: Path | None = None

    @property
    def service_root(self) -> Path | None:
        """The folder that holds the config file."""
        return self.path.parent if self.path else None

    @classmethod
    def for_tests(cls, store_root: Path, models_root: Path | None = None) -> Config:
        """A config with the design's defaults and the given roots, for tests."""
        return cls(server=ServerConfig(store_root=store_root, models_root=models_root or store_root / "models"))


# ---------------------------------------------------------------- range checks (the design's bounds)
_RANGES: dict[tuple[str, str], tuple[float, float]] = {
    ("retention", "retention_days"): (1, 36500),
    ("retention", "measurement_retention_days"): (1, 36500),
    ("daemon", "idle_unload_s"): (0, 86400),
    ("daemon", "idle_exit_min"): (0, 10080),
    ("workers", "cpu_threads"): (1, 256),
    ("gpu", "min_free_margin_mb"): (0, 1 << 20),
    ("gpu", "wait_timeout_min"): (0, 1440),
    ("defaults", "takes"): (1, 3),
    ("defaults", "max_retakes"): (0, 3),
    ("delivery", "sample_rate"): (8000, 192000),
    ("delivery", "trim_pad_s"): (0, 1),
    ("delivery", "trim_floor_dbfs"): (-120, 0),
    ("delivery", "fade_s"): (0, 1),
    ("measurement", "seeds"): (1, 10),
    ("measurement", "pace_tol_min"): (0, 1),
    ("measurement", "sim_warn_margin"): (0, 1),
    ("measurement", "sim_fail_floor"): (0, 1),
    ("alignment", "disagree_threshold_s"): (0, 10),
    ("alignment", "low_confidence_below"): (0, 1),
    ("alignment", "unplaced_below"): (0, 1),
    ("engines.qwen3_base", "max_new_tokens_per_char"): (0.1, 100),
    ("engines.qwen3_base", "max_new_tokens_floor"): (2, 8192),
    ("engines.qwen3_design", "max_new_tokens_per_char"): (0.1, 100),
    ("engines.qwen3_design", "max_new_tokens_floor"): (2, 8192),
}


def qa_profile_mismatch(config: Config) -> tuple[str, str] | None:
    """What to tell the operator when ``[qa] profile`` names another QA profile than the one this build scores
    with (``names.QA_PROFILE``): (what was found, what to do). None when they agree. The setting changes
    nothing, so this is a warning (``narration-admin doctor``, and the daemon's log at start), never a
    refusal."""
    if config.qa.profile == QA_PROFILE:
        return None
    return (
        f"[qa] profile is {config.qa.profile!r}, but this build scores every take with QA profile {QA_PROFILE!r}; "
        "the setting changes nothing",
        f'Update the line in [qa] to profile = "{QA_PROFILE}" (narration.example.toml has it), or remove it.',
    )


CONFIG_FILE_NAME = "narration.toml"
"""The configuration's file name in the service's folder (``<service_root>``, design section 16)."""

CONFIG_ENV = "NARRATION_CONFIG"
"""An environment variable that names the configuration file, when no path is given."""


def service_root() -> Path | None:
    """The service's folder when this package runs from a source checkout (the v1 distribution: a git clone
    plus uv): the folder holding ``pyproject.toml`` above ``src/narration``. None when installed elsewhere."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").is_file() and (root / "src" / "narration").is_dir() else None


def find_config(explicit: str | Path | None = None, *, environ: Mapping[str, str] | None = None) -> Path:
    """The configuration file the server and the operator CLI both use, found by one rule:

    1. ``explicit`` (``--config``), when given;
    2. else the file ``NARRATION_CONFIG`` names, when set;
    3. else ``narration.toml`` in the service's folder (``service_root``).

    Raises ``ConfigError`` saying what to do when the file is not there. The path is returned absolute; it is
    read by ``load_config``.
    """
    env = os.environ if environ is None else environ
    if explicit is not None and str(explicit) != "":
        candidate, source = Path(explicit), "--config"
    elif env.get(CONFIG_ENV):
        candidate, source = Path(env[CONFIG_ENV]), CONFIG_ENV
    else:
        root = service_root()
        if root is None:
            raise ConfigError(f"no configuration given: pass --config <path to {CONFIG_FILE_NAME}> or set {CONFIG_ENV}")
        candidate, source = root / CONFIG_FILE_NAME, "the service's folder"
    candidate = candidate.expanduser().resolve()
    if not candidate.is_file():
        raise ConfigError(
            f"no configuration file at {candidate} (from {source}); copy narration.example.toml to "
            f"{CONFIG_FILE_NAME} and edit it, or pass --config <path>"
        )
    return candidate


def load_config(path: Path) -> Config:
    """Load and validate a configuration file; raises ``ConfigError`` with the offending key."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"no configuration file at {path}; copy narration.example.toml and edit it") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: not valid TOML: {exc}") from exc
    return parse_config(data, base=path.resolve().parent, path=path.resolve())


def parse_config(data: dict[str, Any], *, base: Path, path: Path | None = None) -> Config:
    """Validate parsed TOML against the design's keys; relative paths resolve against ``base``."""
    if "server" not in data:
        raise ConfigError("[server] is required, with store_root and models_root")
    raw_workers = data.get("workers", {})
    if not isinstance(raw_workers, dict):
        raise ConfigError("[workers] must be a table")
    workers = dict(raw_workers)
    worker_projects = {role: workers.pop(role) for role in ("qwen3", "qa") if role in workers}
    sections: dict[str, Any] = {}
    for name, tp in get_type_hints(Config).items():
        if name == "path" or name not in data:
            continue
        raw = workers if name == "workers" else data[name]
        sections[name] = _build(tp, raw, name, base)
    unknown = sorted(set(data) - {f.name for f in dataclasses.fields(Config)} - {"path"})
    if unknown or "path" in data:
        raise ConfigError(f"unknown section(s): {', '.join(unknown or ['path'])}")
    if worker_projects:
        current = sections.get("workers", WorkersConfig())
        extra = {role: _build(WorkerProject, raw, f"workers.{role}", base) for role, raw in worker_projects.items()}
        sections["workers"] = dataclasses.replace(current, **extra)
    config = Config(**sections, path=path)
    if config.alignment.unplaced_below > config.alignment.low_confidence_below:
        raise ConfigError("alignment.unplaced_below must not be above alignment.low_confidence_below")
    for sha in config.voices.allow_sha256:
        if not isinstance(sha, str) or not _SHA256.fullmatch(sha):
            raise ConfigError(f"voices.allow_sha256: {sha!r} is not 64 lower-case hex characters")
    return config


def _build(cls: Any, raw: Any, where: str, base: Path) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"[{where}] must be a table")
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - names)
    if unknown:
        raise ConfigError(f"[{where}] unknown key(s): {', '.join(unknown)}")
    values: dict[str, Any] = {}
    for key, value in raw.items():
        values[key] = _value(hints[key], value, f"{where}.{key}", base)
        bounds = _RANGES.get((where, key))
        if bounds and not (bounds[0] <= values[key] <= bounds[1]):
            raise ConfigError(f"{where}.{key} = {value!r} is outside {bounds[0]}..{bounds[1]}")
    try:
        return cls(**values)
    except TypeError as exc:
        raise ConfigError(f"[{where}] {exc}") from exc


def _value(tp: Any, value: Any, where: str, base: Path) -> Any:
    origin = get_origin(tp)
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, where, base)
    if tp is Path:
        if not isinstance(value, str) or not value:
            raise ConfigError(f"{where} must be a non-empty path")
        p = Path(value)
        return p if p.is_absolute() else (base / p)
    if origin is Literal:
        if value not in get_args(tp):
            raise ConfigError(f"{where} must be one of {', '.join(map(repr, get_args(tp)))}")
        return value
    if origin is tuple:
        if not isinstance(value, list):
            raise ConfigError(f"{where} must be an array")
        (item, _) = get_args(tp)
        return tuple(_value(item, v, f"{where}[{i}]", base) for i, v in enumerate(value))
    if origin is dict:
        if not isinstance(value, dict) or not all(isinstance(v, str) for v in value.values()):
            raise ConfigError(f"{where} must be a table of strings")
        return {str(k): v for k, v in value.items()}
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where} must be true or false")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where} must be an integer")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where} must be a number")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where} must be a string")
        return value
    if origin is not None and type(None) in get_args(tp):
        (inner,) = [a for a in get_args(tp) if a is not type(None)]
        return _value(inner, value, where, base)
    raise ConfigError(f"{where}: unsupported setting type")
