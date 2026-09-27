"""``narration-admin gc`` (a dry run unless ``--apply``) and ``verify`` (design section 15).

The store's own ``gc`` and ``verify`` are WP12's and tested there; these tests pin what the operator sees:
a dry run removes nothing, ``--apply`` does, and ``verify`` finds a changed file in the store and in the
models root.
"""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

import pytest

from narration.admin.cli import EXIT_FAILED, EXIT_OK
from narration.config import load_config
from narration.contracts import names
from narration.platform.testing import StandInPlatform
from narration.store import NarrationStore
from tests.store.factories import audio_bytes, render_record, scratch_file

from .conftest import AdminRun
from .test_doctor import REV_A, install_model

DAY = 86400.0


@pytest.fixture
def old_render(config_path: Path, platform: StandInPlatform) -> Path:
    """A store holding one render last used 400 days ago; returns its raw file."""
    config = load_config(config_path)
    with NarrationStore(config.server.store_root, platform, clock=lambda: time.time() - 400 * DAY) as store:
        render = store.put_render(render_record(), scratch_file(store, "raw.wav", audio_bytes("raw")))
        return store.root / render.raw.path


def test_gc_with_no_store_has_nothing_to_collect_s15(admin: AdminRun, config_path: Path) -> None:
    ran = admin("--config", str(config_path), "gc")
    assert ran.code == EXIT_OK and "nothing to collect" in ran.out
    assert not (config_path.parent / "store").exists()


def test_gc_is_a_dry_run_unless_applied_s15(admin: AdminRun, config_path: Path, old_render: Path) -> None:
    ran = admin("--config", str(config_path), "gc")
    assert ran.code == EXIT_OK
    assert "A dry run: 1 renders" in ran.out and "Nothing was removed" in ran.out and "gc --apply" in ran.out
    assert old_render.is_file()
    ran = admin("--config", str(config_path), "gc", "--apply")
    assert ran.code == EXIT_OK and "Removed 1 renders" in ran.out
    assert not old_render.exists()
    assert "A dry run: nothing" in admin("--config", str(config_path), "gc").out


def test_gc_as_json_is_the_stores_report_s15(admin: AdminRun, config_path: Path, old_render: Path) -> None:
    report = json.loads(admin("--config", str(config_path), "gc", "--json").out)
    assert report["dry_run"] is True and len(report["items"]["renders"]) == 1


def test_verify_with_nothing_installed_passes_s15(admin: AdminRun, config_path: Path) -> None:
    ran = admin("--config", str(config_path), "verify")
    assert ran.code == EXIT_OK and "no store" in ran.out and "Everything checked matches" in ran.out


def test_verify_finds_a_changed_file_in_the_store_s15(admin: AdminRun, config_path: Path, old_render: Path) -> None:
    assert admin("--config", str(config_path), "verify").code == EXIT_OK
    os.chmod(old_render, stat.S_IWRITE | stat.S_IREAD)
    old_render.write_bytes(b"not the render")
    ran = admin("--config", str(config_path), "verify")
    assert ran.code == EXIT_FAILED and "changed:" in ran.out and "tell the maintainers" in ran.out


def test_verify_finds_a_changed_model_file_s15(admin: AdminRun, config_path: Path) -> None:
    model = install_model(config_path.parent / "models", names.MODEL_ALIGNER, REV_A, {"model.bin": b"weights"})
    ran = admin("--config", str(config_path), "verify")
    assert ran.code == EXIT_OK and "1 file(s) checked in 1 model(s)" in ran.out
    (model.snapshot_dir / "model.bin").write_bytes(b"other weights")
    ran = admin("--config", str(config_path), "verify", "--json")
    data = json.loads(ran.out)
    assert ran.code == EXIT_FAILED and data["ok"] is False
    assert data["models"]["problems"] == [f"{model.key}: model.bin does not match its recorded sha256"]
