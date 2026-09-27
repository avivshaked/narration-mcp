"""``narration-admin install``'s hardening (the WP37 review): no file outside a snapshot folder, TLS only,
nothing half-written inside a snapshot, readable errors, and the Hub's listing parsed as the Hub writes it
(design sections 4, 15 and 17.8).

The Hub is faked (a fake Hub, or a fake opener under ``HubClient``): nothing here touches the network.
"""

from __future__ import annotations

import dataclasses
import http.client
import io
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from narration.admin import install, models
from narration.admin.cli import EXIT_OK, AdminError
from narration.admin.install import HubClient, RemoteFile, install_model, run_install, sync_worker
from narration.admin.models import InstalledModel, check_files, read_manifest, snapshot_dir, write_manifest
from narration.config import load_config
from narration.platform.testing import StandInPlatform
from narration.store import NarrationStore

from .test_daemon_commands import status
from .test_doctor import worker_projects
from .test_install import REPO, REV, FakeHub, FakeUv, args, git_sha1, make_admin, sources


class ListingHub(FakeHub):
    """A Hub whose listing names ``path``, publishing the hash of what it serves, so the hash check passes."""

    def __init__(self, path: str) -> None:
        super().__init__({path: b"not a model file"})


@pytest.mark.parametrize(
    "path",
    [
        "../../../../escaped.txt",
        "a/../../../../../escaped.txt",
        "/abs/escaped.txt",
        "sub\\..\\x.txt",
        "C:x.txt",
        "./x.txt",
        "a//b.txt",
        "",
    ],
)
def test_a_listed_file_name_that_leaves_the_snapshot_folder_is_refused_s17_2(tmp_path: Path, path: str) -> None:
    models_root = tmp_path / "models"
    with pytest.raises(ValueError, match="not a file name inside the model's folder"):
        install_model(models_root, REPO, REV, ListingHub(path))
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert written == [], written


def test_an_absolute_listed_file_name_writes_nothing_s17_2(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere" / "absolute.txt"
    target.parent.mkdir()
    with pytest.raises(ValueError):
        install_model(tmp_path / "models", REPO, REV, ListingHub(target.as_posix()))
    assert not target.exists()


def test_a_traversal_in_the_listing_stops_the_install_with_an_error_s17_2(
    config_path: Path, platform: StandInPlatform
) -> None:
    admin, _, _ = make_admin(config_path, platform)
    with pytest.raises(AdminError, match="could not be installed"):
        run_install(admin, args(models_only=True), sources=sources(ListingHub("../../escaped.txt")))


def test_an_install_record_that_points_outside_its_folder_is_refused_s17_2(tmp_path: Path) -> None:
    root = tmp_path / "models"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"x": {"repo": "a/b", "revision": REV, "snapshot_dir": "../outside", "files_sha256": {}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="snapshot_dir"):
        read_manifest(root)
    folder = snapshot_dir(root, "a/b", REV)
    folder.mkdir(parents=True)
    model = InstalledModel(repo="a/b", revision=REV, snapshot_dir=folder, files_sha256={"../../x": "0" * 64})
    assert check_files(model) == ["the install record: '../../x' is not a file name inside the model's folder"]


def test_inside_accepts_a_nested_file_s17_2(tmp_path: Path) -> None:
    assert (
        models.inside(tmp_path, "speech_tokenizer/model.safetensors", "here")
        == tmp_path / "speech_tokenizer" / "model.safetensors"
    )


# ---------------------------------------------------------------- TLS only (section 17.8)
@pytest.mark.parametrize(
    "endpoint", ["http://mirror.example.invalid", "ftp://mirror.example.invalid", "mirror.example"]
)
def test_an_endpoint_that_is_not_https_is_refused_s17_8(endpoint: str) -> None:
    with pytest.raises(ValueError, match="not an https:// URL"):
        HubClient(endpoint=endpoint)


def test_an_http_hf_endpoint_stops_the_install_with_advice_s17_8(
    config_path: Path, platform: StandInPlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_ENDPOINT", "http://mirror.example.invalid")
    admin, _, _ = make_admin(config_path, platform)
    fake = dataclasses.replace(sources(), hub=HubClient)
    with pytest.raises(AdminError, match="HF_ENDPOINT"):
        run_install(admin, args(models_only=True), sources=fake)


def test_a_redirect_is_followed_only_to_https_s17_8() -> None:
    handler = install._HttpsOnlyRedirects()  # pyright: ignore[reportPrivateUsage]
    request = urllib.request.Request("https://huggingface.co/a/b/resolve/c/model.bin")
    with pytest.raises(urllib.error.HTTPError, match="not https"):
        handler.redirect_request(request, None, 302, "Found", {}, "http://cdn.example.invalid/model.bin")
    followed = handler.redirect_request(request, None, 302, "Found", {}, "https://cdn.example.invalid/model.bin")
    assert followed is not None and followed.full_url == "https://cdn.example.invalid/model.bin"


# ---------------------------------------------------------------- nothing half-written in a snapshot
def test_a_file_being_fetched_is_never_inside_the_snapshot_folder_s4(tmp_path: Path) -> None:
    """A hard kill cannot run ``finally``: whatever the fetch had written must not be a file an engine profile
    would hash (every file in the snapshot)."""
    seen: list[Path] = []

    class Dying(ListingHub):
        def download(self, repo: str, revision: str, path: str, dest: Path) -> None:
            dest.write_bytes(b"half")
            seen.append(dest)
            raise KeyboardInterrupt  # stands in for the kill

    models_root = tmp_path / "models"
    with pytest.raises(KeyboardInterrupt):
        install_model(models_root, REPO, REV, Dying("model.safetensors"))
    (temp,) = seen
    assert temp.parent == models_root / ".staging"
    assert not temp.is_relative_to(snapshot_dir(models_root, REPO, REV))


def test_what_a_killed_install_left_is_removed_by_the_next_s4(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    staging = models_root / ".staging"
    staging.mkdir(parents=True)
    (staging / "tmp123.partial").write_bytes(b"half")
    folder = snapshot_dir(models_root, REPO, REV)
    folder.mkdir(parents=True)
    (folder / "model.safetensors.partial").write_bytes(b"half, from an earlier version")
    install_model(models_root, REPO, REV, FakeHub())
    assert list(staging.iterdir()) == []
    assert sorted(p.name for p in folder.iterdir()) == ["config.json", "model.safetensors"]


# ---------------------------------------------------------------- errors an operator can act on
class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, link: str | None = None) -> None:
        super().__init__(body)
        self.headers = {"Link": link} if link else {}


class FakeOpener:
    """Answers each URL from ``pages`` (a listing) or ``files`` (a download)."""

    def __init__(self, pages: dict[str, tuple[Any, str | None]], files: dict[str, Any] | None = None) -> None:
        self.pages = pages
        self.files = files or {}
        self.opened: list[str] = []

    def open(self, fullurl: str, data: None = None, timeout: float = 0.0) -> Any:
        self.opened.append(fullurl)
        if fullurl in self.pages:
            body, link = self.pages[fullurl]
            return FakeResponse(json.dumps(body).encode("utf-8"), link)
        return self.files[fullurl]


LISTING = "https://hub.example/api/models/org/name/tree/" + REV + "?recursive=true"
PAGE_2 = LISTING + "&cursor=2"


def test_the_listing_takes_lfs_files_by_sha256_and_small_files_by_git_sha1_s17_8() -> None:
    small = b'{"a": 1}'
    page_1 = [
        {"type": "directory", "path": "speech_tokenizer", "oid": "d" * 40, "size": 0},
        {"type": "file", "path": "config.json", "oid": git_sha1(small), "size": len(small)},
    ]
    page_2 = [
        {
            "type": "file",
            "path": "speech_tokenizer/model.safetensors",
            "oid": "e" * 40,
            "size": 1234,
            "lfs": {"oid": "f" * 64, "size": 1234, "pointerSize": 134},
        }
    ]
    opener = FakeOpener({LISTING: (page_1, f'<{PAGE_2}>; rel="next"'), PAGE_2: (page_2, None)})
    files = HubClient(endpoint="https://hub.example", opener=opener).list_files("org/name", REV)
    assert files == [
        RemoteFile("config.json", len(small), None, git_sha1(small)),
        RemoteFile("speech_tokenizer/model.safetensors", 1234, "f" * 64, "e" * 40),
    ]
    assert opener.opened == [LISTING, PAGE_2]


@pytest.mark.parametrize(
    "entry",
    [
        {"type": "file", "path": "a.json", "oid": "0" * 40},
        {"type": "file", "path": "a.json", "size": "big", "oid": "0"},
    ],
)
def test_a_malformed_listing_is_a_readable_error_s17_8(entry: dict[str, Any]) -> None:
    opener = FakeOpener({LISTING: ([entry], None)})
    with pytest.raises(ValueError, match="listing of org/name@"):
        HubClient(endpoint="https://hub.example", opener=opener).list_files("org/name", REV)


def test_a_truncated_download_is_a_readable_error_s17_8(config_path: Path, platform: StandInPlatform) -> None:
    class Truncating(FakeHub):
        def download(self, repo: str, revision: str, path: str, dest: Path) -> None:
            raise http.client.IncompleteRead(b"half", 100)

    admin, _, _ = make_admin(config_path, platform)
    with pytest.raises(AdminError, match="could not be installed: IncompleteRead"):
        run_install(admin, args(models_only=True), sources=sources(Truncating()))


def test_a_file_a_running_daemon_holds_says_to_stop_it_first_s17_8(
    config_path: Path, platform: StandInPlatform, monkeypatch: pytest.MonkeyPatch
) -> None:
    def held(src: object, dst: object) -> None:
        raise PermissionError(13, "The process cannot access the file because it is being used by another process")

    monkeypatch.setattr(install.os, "replace", held)
    admin, _, _ = make_admin(config_path, platform)
    with pytest.raises(AdminError, match="narration-admin daemon stop` first"):
        run_install(admin, args(models_only=True), sources=sources())


@pytest.mark.parametrize(
    ("error", "said"),
    [
        (subprocess.TimeoutExpired(["uv"], 3600), "did not finish"),
        (FileNotFoundError(2, "No such file"), "could not be run"),
    ],
)
def test_uv_that_times_out_or_cannot_run_is_a_readable_error_s17_8(
    config_path: Path, error: Exception, said: str
) -> None:
    def failing(argv: object, cwd: object) -> subprocess.CompletedProcess[str]:
        raise error

    worker_projects(config_path.parent)
    config = load_config(config_path)
    with pytest.raises(AdminError, match=said):
        sync_worker(config, "qwen3", uv="uv", check=lambda project: install.SyncCheck(False, "x"), run=failing)


def test_after_a_model_is_repaired_a_running_daemon_is_asked_to_stop_s4_1(
    config_path: Path, platform: StandInPlatform
) -> None:
    config = load_config(config_path)
    with NarrationStore(config.server.store_root, platform) as store:
        admin, _, _ = make_admin(config_path, platform)
        assert run_install(admin, args(models_only=True), sources=sources()) == EXIT_OK
        store.put_daemon_status(status("busy"))
        admin, _, _ = make_admin(config_path, platform)
        assert run_install(admin, args(models_only=True), sources=sources()) == EXIT_OK
        assert store.pending_commands() == (), "nothing was repaired"
        (snapshot_dir(config.server.models_root, REPO, REV) / "config.json").write_bytes(b"{}")
        admin, out, _ = make_admin(config_path, platform)
        assert run_install(admin, args(models_only=True), sources=sources(uv=FakeUv())) == EXIT_OK
        assert [c.kind for c in store.pending_commands()] == ["stop"]
        assert "asked to stop" in out.getvalue()


def test_the_manifest_written_after_a_repair_is_readable_s15(config_path: Path, platform: StandInPlatform) -> None:
    admin, _, _ = make_admin(config_path, platform)
    run_install(admin, args(models_only=True), sources=sources())
    root = load_config(config_path).server.models_root
    records = read_manifest(root)
    write_manifest(root, records)
    assert read_manifest(root) == records
