"""``narration-admin install``: the pinned models, verified, and the worker venvs (design sections 4, 17.7 and
17.8; plan.md §1.4).

::

    narration-admin install [--dry-run] [--from-cache <hub cache>] [--models-only | --workers-only]

1. **First, what this machine lacks**: the platform and the GPU, as ``doctor`` checks them. A missing GPU
   does not stop the install, but the machine cannot run jobs until it has one.
2. **The models.** For each model the engine pins (repo and revision, WP32's ``narration.engine.models``),
   the files of that revision are listed from Hugging Face, and each one is put in the snapshot folder
   ``<models_root>/models--<org>--<name>/snapshots/<revision>/`` (section 4). It is copied from a local Hugging
   Face cache (``--from-cache``, read-only) when it is there, else downloaded, to a temporary file in
   ``<models_root>/.staging/`` (never inside a snapshot folder). **Every file is checked against the hash
   Hugging Face publishes for that revision** (sha256 for large files, the git blob sha1 for small ones)
   before it is renamed into place; one that does not match is refused and nothing half-written is left in
   the snapshot. A file name in the listing must be a plain relative path inside the snapshot folder, or
   nothing is written. What a killed install left in ``.staging`` is removed by the next one. A file already
   there is kept if it matches, and replaced if it does not. What was installed is recorded, with each file's
   sha256, in ``<models_root>/manifest.json`` (``narration.admin.models``), which
   ``doctor`` and ``verify`` check offline. Documentation and other frameworks' weights are skipped, and
   pickled weights are skipped where safetensors exist.
3. **The worker venvs.** Each worker's uv project (``[workers.<role>] project``) is synced from its
   committed ``uv.lock`` (``uv sync --locked``), unless ``uv sync --check`` says it already matches.
4. **The daemon.** A daemon that runs keeps a worker it found broken broken for its lifetime
   (``BACKEND_NOT_INSTALLED``), so when a worker venv or a model was repaired, a running daemon is asked to
   stop (after the segment in flight). The stop follows every rule of ``daemon stop`` (section 4.1): jobs
   queued before it wait for the next start (``daemon start``, or the next ``submit_job`` while ``[daemon]
   autostart`` is on), and the daemon that start brings up runs the whole queue. As with ``daemon stop``,
   nothing is posted when no daemon runs.

**TLS** (section 17.8). Downloads verify certificates with the system's store, plus the bundle
``SSL_CERT_FILE`` or ``REQUESTS_CA_BUNDLE`` names; uv is told the same way (``UV_NATIVE_TLS=1`` uses the
system's store). Verification is never turned off: when it fails, the install stops and says what to set.
``HF_ENDPOINT`` names a Hugging Face mirror, as it does for Hugging Face's own tools; it must be ``https``,
and a redirect to anything but ``https`` is refused.

``--dry-run`` lists what would be downloaded and synced, and changes nothing.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import http.client
import json
import os
import shutil
import ssl
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

from narration.config import Config
from narration.contracts.errors import NarrationError
from narration.contracts.names import WorkerRole
from narration.platform import is_supported
from narration.workers.launch import WORKER_PROJECT_DIRS, venv_python, worker_project

from .cli import EXIT_FAILED, EXIT_OK, PROGRAM, Admin, AdminError, Subparsers
from .daemon import request_stop
from .doctor import WORKER_ROLES, SyncCheck, check_gpu, check_platform, read_gpu, uv_sync_check
from .models import InstalledModel, inside, pinned_revisions, read_manifest, snapshot_dir, write_manifest

HUB: Final = "https://huggingface.co"
CHUNK: Final = 8 * 1024 * 1024
TIMEOUT_S: Final = 120.0
STAGING_DIR: Final = ".staging"
"""Where files are written while they are fetched and checked, under the models root: never inside a
snapshot folder, whose every file an engine profile hashes."""
PARTIAL_SUFFIX: Final = ".partial"
UV_SYNC_TIMEOUT_S: Final = 3600.0
SKIP_NAMES: Final = frozenset({".gitattributes", "README.md"})
SKIP_SUFFIXES: Final = (".h5", ".msgpack", ".ot", ".onnx", ".tflite", ".mlmodel")
SKIP_PREFIXES: Final = ("onnx/", "tf/", "flax/", "coreml/")
TLS_ADVICE: Final = (
    "If this network inspects TLS, point SSL_CERT_FILE (or REQUESTS_CA_BUNDLE) at its root certificate bundle, "
    "or set UV_NATIVE_TLS=1 for uv, and run the install again. Never turn certificate verification off; if it "
    "still fails, stop and ask whoever runs this network (design section 17.8)."
)


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """A file of a pinned revision, as the Hub lists it: its size and the hash to check it against."""

    path: str
    size: int
    sha256: str | None
    """The LFS sha256 (large files); None for a small file, which is checked by ``git_sha1``."""
    git_sha1: str


class Hub(Protocol):
    """Where the models come from: list a revision's files, and fetch one."""

    def list_files(self, repo: str, revision: str) -> list[RemoteFile]:
        """Every file of ``repo`` at ``revision``."""
        ...

    def download(self, repo: str, revision: str, path: str, dest: Path) -> None:
        """Write the file ``path`` of ``repo`` at ``revision`` to ``dest``."""
        ...


class TlsFailure(Exception):
    """A certificate could not be verified."""


def tls_context(environ: Mapping[str, str] | None = None) -> ssl.SSLContext:
    """A verifying TLS context: the system's certificates (``SSL_CERT_FILE`` and ``SSL_CERT_DIR`` apply), plus
    the bundle ``REQUESTS_CA_BUNDLE`` names. Verification is never turned off."""
    env = os.environ if environ is None else environ
    context = ssl.create_default_context()
    bundle = env.get("REQUESTS_CA_BUNDLE")
    if bundle:
        context.load_verify_locations(cafile=bundle)
    return context


class _HttpsOnlyRedirects(urllib.request.HTTPRedirectHandler):
    """Follows a redirect only to another ``https`` URL (the Hub sends downloads to its CDN), so a redirect can
    never take a download off TLS."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise urllib.error.HTTPError(
                newurl, code, f"refused a redirect to a URL that is not https: {newurl}", headers, fp
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def https_endpoint(endpoint: str) -> str:
    """``endpoint`` without a trailing ``/``; ``ValueError`` unless it is an ``https://`` URL (section 17.8: a
    plain-HTTP mirror would serve the hashes and the files unverified)."""
    parts = urllib.parse.urlsplit(endpoint)
    if parts.scheme != "https" or not parts.netloc:
        raise ValueError(
            f"the Hugging Face endpoint {endpoint!r} (HF_ENDPOINT) is not an https:// URL. Downloads are only made "
            "over verified TLS (design section 17.8): set HF_ENDPOINT to the mirror's https:// address, or unset it"
        )
    return endpoint.rstrip("/")


class Opener(Protocol):
    """What ``HubClient`` opens URLs with (``urllib.request.OpenerDirector``; tests pass a fake)."""

    def open(self, fullurl: str, data: None = None, timeout: float = ...) -> Any:
        """Open ``fullurl``."""
        ...


class HubClient:
    """The Hugging Face Hub over HTTPS (``urllib``), verifying certificates; ``HF_ENDPOINT`` names a mirror,
    which must be ``https`` too. Redirects are followed only to ``https`` URLs."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        context: ssl.SSLContext | None = None,
        opener: Opener | None = None,
    ) -> None:
        self.endpoint = https_endpoint(endpoint or os.environ.get("HF_ENDPOINT") or HUB)
        self.context = context or tls_context()
        self.opener: Opener = opener or urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=self.context), _HttpsOnlyRedirects()
        )

    def _open(self, url: str) -> Any:
        try:
            return self.opener.open(url, timeout=TIMEOUT_S)
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, ssl.SSLCertVerificationError):
                raise TlsFailure(
                    f"the certificate of {urllib.parse.urlsplit(url).netloc} could not be verified: {exc.reason}"
                ) from exc
            raise

    def list_files(self, repo: str, revision: str) -> list[RemoteFile]:
        """Every file of ``repo`` at ``revision``, following the listing's pages. ``ValueError`` for a listing
        that is not the Hub's shape."""
        url: str | None = f"{self.endpoint}/api/models/{repo}/tree/{revision}?recursive=true"
        files: list[RemoteFile] = []
        while url:
            with self._open(url) as response:
                entries = json.load(response)
                url = _next_link(response.headers.get("Link"))
            if not isinstance(entries, list):
                raise ValueError(f"the Hub's listing of {repo}@{revision} is not a list")
            for entry in entries:
                if not isinstance(entry, dict) or entry.get("type") != "file":
                    continue
                try:
                    lfs = entry.get("lfs") or {}
                    sha256 = lfs.get("oid") if isinstance(lfs, dict) else None
                    files.append(
                        RemoteFile(
                            str(entry["path"]),
                            int(entry["size"]),
                            str(sha256) if sha256 is not None else None,
                            str(entry["oid"]),
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"the Hub's listing of {repo}@{revision} has a file entry without a usable {exc}: {entry!r}"
                    ) from exc
        return files

    def download(self, repo: str, revision: str, path: str, dest: Path) -> None:
        url = f"{self.endpoint}/{repo}/resolve/{revision}/{urllib.parse.quote(path)}"
        with self._open(url) as response, dest.open("wb") as out:
            shutil.copyfileobj(response, out, CHUNK)


def _next_link(header: str | None) -> str | None:
    """The ``rel="next"`` URL of an HTTP ``Link`` header (the Hub pages long listings)."""
    for part in (header or "").split(","):
        url, _, params = part.partition(";")
        if 'rel="next"' in params:
            return url.strip().strip("<>")
    return None


def select_files(files: Sequence[RemoteFile]) -> list[RemoteFile]:
    """The files the service loads: no documentation, no other frameworks' weights, and no pickled weights
    where safetensors exist (as ``tools/dev_models.py`` picks them)."""
    kept = [
        f
        for f in files
        if f.path not in SKIP_NAMES
        and not f.path.endswith(SKIP_SUFFIXES)
        and not f.path.startswith(SKIP_PREFIXES)
        and ".fp32" not in f.path
    ]
    if any(f.path.endswith(".safetensors") for f in kept):
        kept = [f for f in kept if not (f.path.endswith(".bin") or f.path.endswith(".bin.index.json"))]
    return kept


def remove_stale_partials(models_root: Path, folder: Path) -> None:
    """Remove what an install that was killed left: temporary files in ``<models_root>/.staging/``, and any
    ``*.partial`` file an earlier version left inside the snapshot folder."""
    staging = models_root / STAGING_DIR
    stale = [*staging.glob(f"*{PARTIAL_SUFFIX}"), *(folder.rglob(f"*{PARTIAL_SUFFIX}") if folder.is_dir() else [])]
    for path in stale:
        with contextlib.suppress(OSError):
            path.unlink()


def file_hashes(path: Path) -> tuple[str, str]:
    """The sha256 and the git blob sha1 of a file."""
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())  # git's object hash, not for security
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            sha256.update(chunk)
            sha1.update(chunk)
    return sha256.hexdigest(), sha1.hexdigest()


def matches(remote: RemoteFile, path: Path) -> str | None:
    """The file's sha256 if it is ``remote`` (size and published hash), else None."""
    if not path.is_file() or path.stat().st_size != remote.size:
        return None
    sha256, sha1 = file_hashes(path)
    ok = sha256 == remote.sha256 if remote.sha256 else sha1 == remote.git_sha1
    return sha256 if ok else None


@dataclass(slots=True)
class ModelResult:
    """What installing one model did."""

    repo: str
    revision: str
    present: int = 0
    fetched: list[tuple[str, int, str]] = field(default_factory=list[tuple[str, int, str]])
    """``(path, size, how)`` of each file copied or downloaded (or, in a dry run, to be)."""
    repaired: bool = False
    """A file that was there did not match and was replaced."""
    record: InstalledModel | None = None


def install_model(
    models_root: Path,
    repo: str,
    revision: str,
    hub: Hub,
    *,
    cache: Path | None = None,
    dry_run: bool = False,
    progress: Callable[[str], None] = lambda line: None,
) -> ModelResult:
    """Put ``repo@revision``'s files in its snapshot folder, each checked against the Hub's hash (see the
    module docstring). Raises ``ValueError`` for a file that does not match after it was fetched."""
    result = ModelResult(repo=repo, revision=revision)
    folder = snapshot_dir(models_root, repo, revision)
    source = snapshot_dir(cache, repo, revision) if cache is not None else None
    staging = models_root / STAGING_DIR
    if not dry_run:
        remove_stale_partials(models_root, folder)
    record: dict[str, str] = {}
    for remote in select_files(hub.list_files(repo, revision)):
        dest = inside(folder, remote.path, f"{repo}@{revision[:12]}")
        sha256 = matches(remote, dest)
        if sha256 is not None:
            result.present += 1
            record[remote.path] = sha256
            continue
        result.repaired = result.repaired or dest.exists()
        cached_path = inside(source, remote.path, "the cache") if source is not None else None
        cached = cached_path if cached_path is not None and cached_path.is_file() else None
        how = "copy" if cached is not None else "download"
        result.fetched.append((remote.path, remote.size, how))
        if dry_run:
            continue
        progress(f"  {how}: {repo} {remote.path} ({remote.size / 1024**2:,.1f} MB)")
        dest.parent.mkdir(parents=True, exist_ok=True)
        staging.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(dir=staging, suffix=PARTIAL_SUFFIX)
        os.close(handle)
        tmp = Path(name)
        try:
            if cached is not None:
                shutil.copyfile(cached, tmp)
            else:
                hub.download(repo, revision, remote.path, tmp)
            sha256 = matches(remote, tmp)
            if sha256 is None:
                raise ValueError(
                    f"{repo}@{revision[:12]}: {remote.path} does not match the hash Hugging Face publishes for it "
                    f"({'copied from the cache' if cached is not None else 'downloaded'}); it was not installed"
                )
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
        record[remote.path] = sha256
    if not dry_run:
        result.record = InstalledModel(repo=repo, revision=revision, snapshot_dir=folder, files_sha256=record)
    return result


@dataclass(frozen=True, slots=True)
class WorkerResult:
    """What syncing one worker's venv did: ``synced`` (it was out of date and uv synced it, or would in a dry
    run), and uv's last words."""

    role: str
    project: Path
    synced: bool
    detail: str


UvRun = Callable[[Sequence[str], Path], "subprocess.CompletedProcess[str]"]


def run_uv(argv: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run uv with this environment, without ``VIRTUAL_ENV`` (the server's venv) and, unless set, with
    ``UV_PYTHON_DOWNLOADS=never`` (the worker uses the Python the service runs on)."""
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env.setdefault("UV_PYTHON_DOWNLOADS", "never")
    return subprocess.run(
        list(argv),
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=UV_SYNC_TIMEOUT_S,
    )


def find_uv() -> str | None:
    """The uv that runs this service (``UV``, set by ``uv run``), else the one on ``PATH``."""
    return os.environ.get("UV") or shutil.which("uv")


def sync_worker(
    config: Config,
    role: WorkerRole,
    *,
    uv: str,
    dry_run: bool = False,
    check: Callable[[Path], SyncCheck] | None = None,
    run: UvRun = run_uv,
) -> WorkerResult:
    """Sync ``role``'s venv from its ``uv.lock`` unless it already matches. ``AdminError`` says what to do
    when the project is missing or uv fails."""
    project = worker_project(config, role)
    if project is None or not (project / "pyproject.toml").is_file():
        raise AdminError(
            f"the {role} worker's uv project is not at {project}: set [workers.{role}] project to the service's "
            f"workers/{WORKER_PROJECT_DIRS[role]} folder, then run the install again."
        )
    state = (check or (lambda p: uv_sync_check(p, uv=uv)))(project)
    if state.synced is True and venv_python(project).is_file():
        return WorkerResult(role, project, False, "its venv matches its uv.lock")
    if dry_run:
        return WorkerResult(role, project, True, "its venv would be synced from its uv.lock")
    try:
        done = run([uv, "sync", "--locked", "--project", str(project)], project)
    except subprocess.TimeoutExpired as exc:
        raise AdminError(
            f"uv did not finish syncing the {role} worker's venv in {project} within {exc.timeout:.0f} s. Check the "
            "network (or the uv cache's drive), then run the install again."
        ) from exc
    except OSError as exc:
        raise AdminError(
            f"uv ({uv}) could not be run: {exc}. Install uv or put it on PATH, then run the install again."
        ) from exc
    if done.returncode != 0:
        tail = [line for line in (done.stderr + "\n" + done.stdout).splitlines() if line.strip()][-5:]
        said = " | ".join(line.strip() for line in tail)
        advice = f" {TLS_ADVICE}" if "certificate" in said.lower() else ""
        raise AdminError(f"uv could not sync the {role} worker's venv in {project} (uv: {said}).{advice}")
    return WorkerResult(role, project, True, "its venv was synced from its uv.lock")


# ---------------------------------------------------------------- the command
def register(subparsers: Subparsers) -> None:
    """Add ``install`` (``narration.admin.cli``)."""
    parser = subparsers.add_parser(
        "install",
        help="download the pinned models (verified) and sync the worker venvs (section 17.8)",
        description=(
            "Put every pinned model in the models root, each file checked against the hash Hugging Face "
            "publishes for its revision, and sync each worker's venv from its uv.lock. A running daemon is "
            "asked to stop when something it uses was repaired."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="list what would be downloaded and synced; change nothing"
    )
    parser.add_argument(
        "--from-cache",
        type=Path,
        default=None,
        metavar="DIR",
        help="a local Hugging Face hub cache to copy files from (read-only), checked like downloads",
    )
    only = parser.add_mutually_exclusive_group()
    only.add_argument("--models-only", action="store_true", help="install the models, not the worker venvs")
    only.add_argument("--workers-only", action="store_true", help="sync the worker venvs, not the models")
    parser.set_defaults(handler=run_install)


@dataclass(frozen=True, slots=True, kw_only=True)
class Sources:
    """Where ``install`` gets things; tests replace them."""

    hub: Callable[[], Hub] = HubClient
    pins: Callable[[], dict[str, str] | None] = pinned_revisions
    uv: Callable[[], str | None] = find_uv
    venv_check: Callable[[Path], SyncCheck] | None = None
    uv_run: UvRun = run_uv
    gpu: Callable[[int], Any] = read_gpu
    supported: Callable[[], bool] = is_supported


def run_install(admin: Admin, args: argparse.Namespace, *, sources: Sources | None = None) -> int:
    """``install``: see the module docstring."""
    sources = sources or Sources()
    config = admin.config()
    admin.say("Before installing:")
    for finding in (check_platform(sources.supported()), check_gpu(config, sources.gpu)):
        admin.say(f"  {finding.level}: {finding.summary}")
        if finding.next_step:
            admin.say(f"    -> {finding.next_step}")
    incomplete: list[str] = []
    repaired: list[str] = []
    if not args.workers_only:
        incomplete += _install_models(admin, config, args, sources, repaired)
    if not args.models_only:
        incomplete += _sync_workers(admin, config, args, sources, repaired)
    if repaired and not args.dry_run:
        _stop_daemon_after_repair(admin, repaired)
    if incomplete:
        admin.warn(f"{PROGRAM}: the install is not complete: " + "; ".join(incomplete) + ".")
        return EXIT_FAILED
    if args.dry_run:
        admin.say("\nA dry run: nothing was changed.")
    else:
        admin.say(f"\nInstalled. Next: `{PROGRAM} engine pin`, then `{PROGRAM} doctor`.")
    return EXIT_OK


def _install_models(
    admin: Admin, config: Config, args: argparse.Namespace, sources: Sources, repaired: list[str]
) -> list[str]:
    try:
        pins = sources.pins()
    except ImportError as exc:
        raise AdminError(
            f"the engine's model pins are in this build but cannot be loaded ({exc}). Sync the server's venv "
            "(uv sync --locked), then run the install again."
        ) from exc
    if pins is None:
        admin.say(
            "\nModels: skipped. This build does not include the engine's model pins (narration.engine.models), so "
            "which revisions to install is not known."
        )
        return ["the models were not installed (this build has no model pins)"]
    root = config.server.models_root
    admin.say(f"\nModels, into {root}:")
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
    try:
        records = read_manifest(root)
    except ValueError as exc:
        admin.say(f"  the install record cannot be read ({exc}); it is written again")
        records = {}
    try:
        hub = sources.hub()
    except ValueError as exc:
        raise AdminError(f"{exc}.") from exc
    except (OSError, ssl.SSLError) as exc:
        raise AdminError(
            f"the certificates to verify downloads with cannot be loaded ({exc}): check the file REQUESTS_CA_BUNDLE "
            "or SSL_CERT_FILE names."
        ) from exc
    total = 0
    for repo, revision in sorted(pins.items()):
        try:
            result = install_model(
                root, repo, revision, hub, cache=args.from_cache, dry_run=args.dry_run, progress=admin.say
            )
        except TlsFailure as exc:
            raise AdminError(f"{exc}. {TLS_ADVICE}") from exc
        except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as exc:
            held = (
                f" A running daemon may hold that file open: run `{PROGRAM} daemon stop` first."
                if isinstance(exc, PermissionError)
                else ""
            )
            raise AdminError(
                f"{repo}@{revision[:12]} could not be installed: {exc or type(exc).__name__}. Nothing half-written was "
                f"left in its folder.{held} Then run the install again (files already checked are kept)."
            ) from exc
        size = sum(size for _, size, _ in result.fetched)
        total += size
        verb = "to fetch" if args.dry_run else "fetched"
        admin.say(
            f"  {repo}@{revision[:12]}: {result.present} file(s) present"
            + (f", {len(result.fetched)} {verb} ({size / 1024**3:.2f} GB)" if result.fetched else "")
            + (" (a file that did not match was replaced)" if result.repaired and not args.dry_run else "")
        )
        if result.record is not None:
            records[result.record.key] = result.record
            write_manifest(root, records)
        if result.repaired and not args.dry_run:
            repaired.append(f"{repo}@{revision[:12]}")
    if args.dry_run and total:
        admin.say(f"  in all, {total / 1024**3:.2f} GB would be fetched")
    return []


def _sync_workers(
    admin: Admin, config: Config, args: argparse.Namespace, sources: Sources, repaired: list[str]
) -> list[str]:
    admin.say("\nWorker venvs:")
    uv = sources.uv()
    if uv is None:
        admin.say("  uv was not found: install uv (https://docs.astral.sh/uv/), then run the install again")
        return ["the worker venvs were not synced (uv was not found)"]
    for role in WORKER_ROLES:
        result = sync_worker(config, role, uv=uv, dry_run=args.dry_run, check=sources.venv_check, run=sources.uv_run)
        admin.say(f"  {role}: {result.project}: {result.detail}")
        if result.synced and not args.dry_run:
            repaired.append(f"the {role} worker's venv")
    return []


def _stop_daemon_after_repair(admin: Admin, repaired: list[str]) -> None:
    if not admin.store_exists():
        return
    try:
        status = request_stop(admin.store(), now=False)
    except NarrationError as exc:
        admin.warn(
            f"{PROGRAM}: could not ask the daemon to stop ({exc.message}); stop it with `{PROGRAM} daemon stop`."
        )
        return
    if status is not None:
        admin.say(
            f"\nThe daemon (pid {status.pid}) was asked to stop after its segment in flight, since it may hold "
            f"a worker it found broken ({', '.join(repaired)} changed). Jobs queued before this stop wait for "
            f"the next start: `{PROGRAM} daemon start`, or the next submit_job while [daemon] autostart is on. "
            "That daemon runs the whole queue."
        )
