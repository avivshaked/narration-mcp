"""The configuration loader (design section 16)."""

from __future__ import annotations

from pathlib import Path

import pytest

from narration.config import Config, load_config, parse_config
from narration.contracts.errors import ConfigError

ROOT = Path(__file__).resolve().parents[2]


def test_the_shipped_example_config_loads_with_its_placeholders(tmp_path: Path) -> None:
    example = ROOT / "narration.example.toml"
    if not example.is_file():
        pytest.skip("narration.example.toml is not on this branch")
    config = load_config(example)
    assert config.voices.allow_sha256 == ()
    assert config.measurement.length_ladder_spoken_chars == (80, 150, 250, 300, 350, 400, 450, 500, 560)
    assert config.workers.qwen3 is not None


def test_defaults_are_the_design_defaults_s16(tmp_path: Path) -> None:
    config = Config.for_tests(tmp_path)
    assert config.defaults.takes == 1 and config.defaults.max_retakes == 2
    assert config.delivery.target_lufs == -23.0 and config.delivery.true_peak_dbtp == -1.0
    assert config.gpu.min_free_margin_mb == 1024 and config.gpu.wait_timeout_min == 30
    assert config.limits.max_submits_per_min == 10 and config.limits.max_queued_jobs == 20
    assert config.workers.env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert config.engines.qwen3_base.non_streaming_mode is False
    assert config.engines.qwen3_design.non_streaming_mode is True
    assert config.voices.allow_sha256 == ()
    for engine in (config.engines.qwen3_base, config.engines.qwen3_design):
        assert (engine.max_new_tokens_per_char, engine.max_new_tokens_floor) == (2.5, 128)  # DC-4
    assert (config.alignment.low_confidence_below, config.alignment.unplaced_below) == (0.75, 0.50)


def test_store_and_models_roots_are_required(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="server"):
        parse_config({}, base=tmp_path)


def test_relative_paths_resolve_against_the_config_folder(tmp_path: Path) -> None:
    config = parse_config({"server": {"store_root": "store", "models_root": "models"}}, base=tmp_path)
    assert config.server.store_root == tmp_path / "store"


@pytest.mark.parametrize(
    "data",
    [
        {"server": {"store_root": "s", "models_root": "m", "typo": 1}},
        {"server": {"store_root": "s", "models_root": "m"}, "sever": {}},
        {"server": {"store_root": "s", "models_root": "m"}, "defaults": {"takes": 4}},
        {"server": {"store_root": "s", "models_root": "m"}, "voices": {"allow_sha256": ["not-a-hash"]}},
        {"server": {"store_root": "s", "models_root": "m"}, "voices": {"allow_sha256": ["a" * 64 + "\n"]}},
        {"server": {"store_root": "s", "models_root": "m"}, "workers": {"priority": "high"}},
        {"server": {"store_root": "s", "models_root": "m"}, "workers": 5},
        {"server": {"store_root": "s", "models_root": "m"}, "workers": [1]},
        {"server": {"store_root": "s", "models_root": "m"}, "engines": {"qwen3_base": {"typo": True}}},
        {"server": {"store_root": "s", "models_root": "m"}, "alignment": {"unplaced_below": 0.9}},
        {"server": {"store_root": "s", "models_root": "m"}, "alignment": {"low_confidence_below": 1.5}},
        {"server": {"store_root": "s", "models_root": "m"}, "engines": {"qwen3_base": {"max_new_tokens_floor": 1}}},
        {
            "server": {"store_root": "s", "models_root": "m"},
            "engines": {"qwen3_design": {"max_new_tokens_per_char": 0}},
        },
    ],
)
def test_unknown_keys_and_bad_values_are_refused(tmp_path: Path, data: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        parse_config(data, base=tmp_path)


def test_worker_projects_are_read_from_their_subtables(tmp_path: Path) -> None:
    config = parse_config(
        {
            "server": {"store_root": "s", "models_root": "m"},
            "workers": {"cpu_threads": 4, "qwen3": {"project": "workers/qwen3tts"}, "qa": {"project": "workers/qa"}},
        },
        base=tmp_path,
    )
    assert config.workers.cpu_threads == 4
    assert config.workers.qwen3 is not None and config.workers.qwen3.project == tmp_path / "workers/qwen3tts"


def test_engine_subtables_are_read(tmp_path: Path) -> None:
    config = parse_config(
        {"server": {"store_root": "s", "models_root": "m"}, "engines": {"qwen3_base": {"non_streaming_mode": True}}},
        base=tmp_path,
    )
    assert config.engines.qwen3_base.non_streaming_mode is True
    assert config.engines.qwen3_design.non_streaming_mode is True


def test_config_defaults_are_the_contract_names() -> None:
    from narration.contracts import names

    config = Config.for_tests(Path("s"))
    assert config.text.checks == names.TEXT_CHECKS_VERSION
    assert config.qa.profile == names.QA_PROFILE
    assert config.measurement.corpus == names.CORPUS
    assert (config.alignment.model, config.alignment.benchmark) == (names.MODEL_ALIGNER, names.BENCHMARK)
