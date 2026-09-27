"""``narration-admin install``: pinned models checked against the Hub's hashes, recorded for ``doctor`` and
``verify``; worker venvs synced from their locks; a running daemon asked to stop after a repair (design
sections 4, 17.8; plan.md §9's decision for WP37).

The Hub is a fake that serves invented files, and uv is a fake runner: nothing here touches the network.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import ssl
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from narration.admin import install
from narration.admin.cli import EXIT_FAILED, EXIT_OK, Admin, AdminError
from narration.admin.doctor import SyncCheck
from narration.admin.install import HubClient, RemoteFile, Sources, TlsFailure, run_install, select_files
from narration.admin.models import read_manifest, snapshot_dir
from narration.config import load_config
from narration.contracts import names
from narration.platform.testing import StandInPlatform
from narration.store import NarrationStore

from .test_daemon_commands import status
from .test_doctor import GOOD_GPU, worker_projects

REPO = names.MODEL_ALIGNER
REV = "c" * 40
FILES = {"config.json": b'{"a": 1}', "model.safetensors": b"weights" * 100, "pytorch_model.bin": b"pickled"}


def git_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


class FakeHub:
    """Serves ``files`` for every pin; ``corrupt`` names files it sends wrong bytes for."""

    def __init__(self, files: dict[str, bytes] | None = None, *, corrupt: Sequence[str] = ()) -> None:
        self.files = dict(FILES if files is None else files)
        self.corrupt = set(corrupt)
        self.downloads: list[str] = []

    def list_files(self, repo: str, revision: str) -> list[RemoteFile]:
        return [
            RemoteFile(
                path,
                len(data),
                hashlib.sha256(data).hexdigest() if path.endswith((".safetensors", ".bin")) else None,
                git_sha1(data),
            )
            for path, data in self.files.items()
        ]

    def download(self, repo: str, revision: str, path: str, dest: Path) -> None:
        self.downloads.append(path)
        dest.write_bytes(b"not it" if path in self.corrupt else self.files[path])


class FakeUv:
    """Records uv's command lines and answers with ``returncode`` and ``stderr``."""

    def __init__(self, returncode: int = 0, stderr: str = "") -> None:
        self.calls: list[list[str]] = []
        self.returncode = returncode
        self.stderr = stderr

    def __call__(self, argv: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        return subprocess.CompletedProcess(list(argv), self.returncode, "", self.stderr)


def sources(
    hub: FakeHub | None = None,
    *,
    pins: dict[str, str] | None = None,
    synced: bool = True,
    uv: FakeUv | None = None,
    uv_path: str | None = "uv",
) -> Sources:
    fake_hub = hub or FakeHub()
    return Sources(
        hub=lambda: fake_hub,
        pins=lambda: {REPO: REV} if pins is None else pins,
        uv=lambda: uv_path,
        venv_check=lambda project: SyncCheck(synced, "checked"),
        uv_run=uv or FakeUv(),
        gpu=lambda index: GOOD_GPU,
        supported=lambda: True,
    )


def args(**given: object) -> argparse.Namespace:
    values: dict[str, object] = {"dry_run": False, "from_cache": None, "models_only": False, "workers_only": False}
    values.update(given)
    return argparse.Namespace(**values)


@pytest.fixture
def service(config_path: Path) -> Path:
    worker_projects(config_path.parent)
    return config_path.parent


def make_admin(config_path: Path, platform: StandInPlatform) -> tuple[Admin, io.StringIO, io.StringIO]:
    out, err = io.StringIO(), io.StringIO()
    return Admin(config_path=config_path, out=out, err=err, platform=lambda: platform, environ={}), out, err


def test_install_puts_each_pinned_file_in_its_snapshot_and_records_its_hash_s17_8(
    service: Path, config_path: Path, platform: StandInPlatform
) -> None:
    admin, out, _ = make_admin(config_path, platform)
    hub = FakeHub()
    assert run_install(admin, args(), sources=sources(hub)) == EXIT_OK
    folder = snapshot_dir(service / "models", REPO, REV)
    assert (folder / "model.safetensors").read_bytes() == FILES["model.safetensors"]
    assert not (folder / "pytorch_model.bin").exists(), "pickled weights are skipped where safetensors exist"
    (record,) = read_manifest(service / "models").values()
    assert record.files_sha256 == {
        p: hashlib.sha256(FILES[p]).hexdigest() for p in ("config.json", "model.safetensors")
    }
    assert not list(folder.rglob("*.partial"))
    assert "engine pin" in out.getvalue()


def test_a_file_that_does_not_match_the_hubs_hash_is_refused_and_not_left_behind_s17_8(
    service: Path, config_path: Path, platform: StandInPlatform
) -> None:
    admin, _, _ = make_admin(config_path, platform)
    with pytest.raises(AdminError, match="does not match the hash Hugging Face publishes"):
        run_install(admin, args(), sources=sources(FakeHub(corrupt=["model.safetensors"])))
    folder = snapshot_dir(service / "models", REPO, REV)
    assert not (folder / "model.safetensors").exists() and not list(folder.rglob("*.partial"))


def test_a_small_file_is_checked_by_its_git_blob_hash_s17_8(
    service: Path, config_path: Path, platform: StandInPlatform
) -> None:
    admin, _, _ = make_admin(config_path, platform)
    with pytest.raises(AdminError, match=r"config\.json does not match"):
        run_install(admin, args(), sources=sources(FakeHub(corrupt=["config.json"])))


def test_files_already_there_are_kept_if_they_match_and_replaced_if_not_s17_8(
    service: Path, config_path: Path, platform: StandInPlatform
) -> None:
    admin, out, _ = make_admin(config_path, platform)
    assert run_install(admin, args(models_only=True), sources=sources()) == EXIT_OK
    hub = FakeHub()
    assert run_install(admin, args(models_only=True), sources=sources(hub)) == EXIT_OK
    assert hub.downloads == [], "nothing is fetched again"
    (snapshot_dir(service / "models", REPO, REV) / "config.json").write_bytes(b"{}")
    assert run_install(admin, args(models_only=True), sources=sources(hub)) == EXIT_OK
    assert hub.downloads == ["config.json"] and "did not match was replaced" in out.getvalue()


def test_files_are_copied_from_a_local_hub_cache_and_checked_too_s17_8(
    service: Path, config_path: Path, platform: StandInPlatform, tmp_path: Path
) -> None:
    cache = tmp_path / "hub-cache"
    cached = snapshot_dir(cache, REPO, REV)
    cached.mkdir(parents=True)
    for path, data in FILES.items():
        (cached / path).write_bytes(data)
    admin, _, _ = make_admin(config_path, platform)
    hub = FakeHub()
    assert run_install(admin, args(models_only=True, from_cache=cache), sources=sources(hub)) == EXIT_OK
    assert hub.downloads == []
    assert (cached / "config.json").read_bytes() == FILES["config.json"], "the cache is only read"
    (cached / "model.safetensors").write_bytes(b"a damaged cache")
    (snapshot_dir(service / "models", REPO, REV) / "model.safetensors").unlink()
    with pytest.raises(AdminError, match="copied from the cache"):
        run_install(admin, args(models_only=True, from_cache=cache), sources=sources(hub))


def test_a_dry_run_changes_nothing_s17_8(service: Path, config_path: Path, platform: StandInPlatform) -> None:
    admin, out, _ = make_admin(config_path, platform)
    uv = FakeUv()
    assert run_install(admin, args(dry_run=True), sources=sources(synced=False, uv=uv)) == EXIT_OK
    assert not (service / "models").exists() and uv.calls == []
    text = out.getvalue()
    assert "2 to fetch" in text and "would be synced" in text and "nothing was changed" in text


def test_without_the_pins_the_models_are_skipped_and_the_install_is_incomplete_s17_8(
    service: Path, config_path: Path, platform: StandInPlatform
) -> None:
    admin, out, err = make_admin(config_path, platform)
    uv = FakeUv()
    fake = dataclasses.replace(sources(uv=uv, synced=False), pins=lambda: None)
    assert run_install(admin, args(), sources=fake) == EXIT_FAILED
    assert "no model pins" in err.getvalue() and "Models: skipped" in out.getvalue()
    assert len(uv.calls) == 2, "the worker venvs are still synced"


def test_worker_venvs_are_synced_from_their_locks_only_when_out_of_date_s17_8(
    service: Path, config_path: Path, platform: StandInPlatform
) -> None:
    admin, _, _ = make_admin(config_path, platform)
    uv = FakeUv()
    assert run_install(admin, args(workers_only=True), sources=sources(synced=True, uv=uv)) == EXIT_OK
    assert uv.calls == []
    assert run_install(admin, args(workers_only=True), sources=sources(synced=False, uv=uv)) == EXIT_OK
    assert [c[1:3] for c in uv.calls] == [["sync", "--locked"], ["sync", "--locked"]]
    assert [Path(c[-1]).name for c in uv.calls] == ["qwen3tts", "qa"]


def test_a_uv_failure_on_a_certificate_gives_the_tls_advice_s17_8(
    service: Path, config_path: Path, platform: StandInPlatform
) -> None:
    admin, _, _ = make_admin(config_path, platform)
    uv = FakeUv(1, "error: invalid peer certificate: UnknownIssuer")
    with pytest.raises(AdminError, match="UV_NATIVE_TLS=1") as raised:
        run_install(admin, args(workers_only=True), sources=sources(synced=False, uv=uv))
    assert "Never turn certificate verification off" in raised.value.message


def test_without_uv_the_workers_are_not_synced_s17_8(
    service: Path, config_path: Path, platform: StandInPlatform
) -> None:
    admin, out, _ = make_admin(config_path, platform)
    assert run_install(admin, args(workers_only=True), sources=sources(uv_path=None)) == EXIT_FAILED
    assert "uv was not found" in out.getvalue()


def test_a_tls_failure_says_what_to_set_and_never_to_turn_verification_off_s17_8(
    service: Path, config_path: Path, platform: StandInPlatform
) -> None:
    class Refusing(FakeHub):
        def list_files(self, repo: str, revision: str) -> list[RemoteFile]:
            raise TlsFailure("the certificate of huggingface.co could not be verified: unknown issuer")

    admin, _, _ = make_admin(config_path, platform)
    with pytest.raises(AdminError) as raised:
        run_install(admin, args(models_only=True), sources=sources(Refusing()))
    assert "SSL_CERT_FILE" in raised.value.message and "Never turn certificate verification off" in raised.value.message


def test_after_a_worker_is_repaired_a_running_daemon_is_asked_to_stop_s4_1(
    service: Path, config_path: Path, platform: StandInPlatform
) -> None:
    config = load_config(config_path)
    with NarrationStore(config.server.store_root, platform) as store:
        admin, out, _ = make_admin(config_path, platform)
        assert run_install(admin, args(workers_only=True), sources=sources(synced=False)) == EXIT_OK
        assert store.pending_commands() == (), "no daemon runs, so no stop is posted"
        store.put_daemon_status(status("busy"))
        admin, out, _ = make_admin(config_path, platform)
        assert run_install(admin, args(workers_only=True), sources=sources(synced=True)) == EXIT_OK
        assert store.pending_commands() == (), "nothing was repaired"
        admin, out, _ = make_admin(config_path, platform)
        assert run_install(admin, args(workers_only=True), sources=sources(synced=False)) == EXIT_OK
        assert [c.kind for c in store.pending_commands()] == ["stop"]
        assert "asked to stop after its segment in flight" in out.getvalue()


def test_select_files_skips_docs_other_frameworks_and_pickles_beside_safetensors_s17_8() -> None:
    def remote(path: str) -> RemoteFile:
        return RemoteFile(path, 1, None, "0" * 40)

    listed = [
        remote(p)
        for p in (
            "README.md",
            ".gitattributes",
            "tf_model.h5",
            "onnx/model.onnx",
            "flax/x.msgpack",
            "model.fp32.safetensors",
            "model.safetensors",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
            "config.json",
        )
    ]
    assert [f.path for f in select_files(listed)] == ["model.safetensors", "config.json"]
    only_bin = [remote("pytorch_model.bin"), remote("config.json")]
    assert [f.path for f in select_files(only_bin)] == ["pytorch_model.bin", "config.json"]


def test_downloads_always_verify_certificates_s17_8() -> None:
    context = HubClient(endpoint="https://example.invalid").context
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname is True


def test_a_ca_bundle_that_cannot_be_read_stops_the_install_s17_8(
    service: Path, config_path: Path, platform: StandInPlatform, tmp_path: Path
) -> None:
    def broken_hub() -> FakeHub:
        install.tls_context({"REQUESTS_CA_BUNDLE": str(tmp_path / "missing.pem")})
        return FakeHub()

    fake = dataclasses.replace(sources(), hub=broken_hub)
    admin, _, _ = make_admin(config_path, platform)
    with pytest.raises(AdminError, match="REQUESTS_CA_BUNDLE"):
        run_install(admin, args(models_only=True), sources=fake)


def test_the_hubs_next_page_link_is_followed_s17_8() -> None:
    header = '<https://huggingface.co/api/models/a/b/tree/c?cursor=x>; rel="next"'
    assert install._next_link(header) == "https://huggingface.co/api/models/a/b/tree/c?cursor=x"  # pyright: ignore[reportPrivateUsage]
    assert install._next_link(None) is None  # pyright: ignore[reportPrivateUsage]
