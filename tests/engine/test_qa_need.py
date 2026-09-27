"""The QA group's VRAM need holds only for the configuration it was measured with (design section 4 item 2;
WP22's spike h). These tests fail when the QA pins, the spike's record or the QA worker's load settings move
away from ``QA_MEASURED``: then spike h must be run again and ``QA_VRAM_NEED_MB`` set from it.

The worker's settings are read from its source with ``ast`` (its venv is not the server's), never imported. A
missing file fails the test rather than skipping it, so a move cannot turn the check off unnoticed.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from narration.engine.qa import QA_MEASURED, QA_VRAM_NEED_MB, aligner, qa_pins

from .support import make_install

REPO = Path(__file__).resolve().parents[2]
SPIKE_RESULTS = REPO / "spikes" / "h-i-qa-load" / "results.json"
QA_WORKER = REPO / "workers" / "qa" / "src" / "narration_worker_qa"
AGAIN = "the QA group's VRAM need was measured with other settings: run spike h again and set QA_VRAM_NEED_MB"


def test_the_qa_pins_are_the_models_spike_h_measured_s4(tmp_path: Path) -> None:
    config = make_install(tmp_path).config
    pins = qa_pins(config)
    assert (pins.asr.repo, pins.asr.revision) == (QA_MEASURED.asr_repo, QA_MEASURED.asr_revision), AGAIN
    assert (pins.sv.repo, pins.sv.revision) == (QA_MEASURED.sv_repo, QA_MEASURED.sv_revision), AGAIN
    assert aligner(config).device == QA_MEASURED.aligner_device, AGAIN
    assert pins.vram_need_mb == QA_VRAM_NEED_MB == 11_500


def test_the_measured_configuration_is_the_spikes_record_s4() -> None:
    assert SPIKE_RESULTS.is_file(), f"spike h's results are not at {SPIKE_RESULTS}: {AGAIN}"
    results = json.loads(SPIKE_RESULTS.read_text(encoding="utf-8"))
    models = results["models"]
    assert (models["asr"]["repo"], models["asr"]["revision"]) == (QA_MEASURED.asr_repo, QA_MEASURED.asr_revision)
    assert (models["sv"]["repo"], models["sv"]["revision"]) == (QA_MEASURED.sv_repo, QA_MEASURED.sv_revision)


def _module(name: str) -> ast.Module:
    path = QA_WORKER / f"{name}.py"
    assert path.is_file(), f"the QA worker's {name}.py is not at {path}: update this test to where it went"
    return ast.parse(path.read_text(encoding="utf-8"))


def _constant(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            assert node.value is not None
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"the QA worker no longer defines {name}: {AGAIN}")


def _dtypes(tree: ast.Module) -> set[str]:
    """The dtypes a module's ``from_pretrained`` calls load the model with: ``dtype=torch.<x>``, or a name
    assigned ``torch.<x>`` or ``torch.<gpu> if <cuda> else torch.<cpu>`` (the GPU's is the one measured)."""
    assigned: dict[str, ast.expr] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned[target.id] = node.value
    found: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "from_pretrained":
            continue
        for keyword in node.keywords:
            if keyword.arg != "dtype":
                continue
            value = keyword.value
            if isinstance(value, ast.Name):
                value = assigned.get(value.id, value)
            if isinstance(value, ast.IfExp):
                value = value.body
            found.add(value.attr if isinstance(value, ast.Attribute) else ast.unparse(value))
    return found


def test_the_qa_worker_loads_whisper_as_spike_h_measured_s4() -> None:
    asr = _module("asr")
    assert _constant(asr, "ATTN_IMPLEMENTATION") == QA_MEASURED.asr_attn_implementation, AGAIN
    assert _constant(asr, "DECODING")["num_beams"] == QA_MEASURED.asr_num_beams, AGAIN
    assert _dtypes(asr) == {QA_MEASURED.asr_dtype}, AGAIN


def test_the_qa_worker_loads_wavlm_as_spike_h_measured_s4() -> None:
    sv = _module("sv")
    assert _constant(sv, "WINDOW_S") == QA_MEASURED.sv_window_s, AGAIN
    assert _dtypes(sv) == {QA_MEASURED.sv_dtype}, AGAIN
