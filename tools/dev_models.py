"""Set up a developer's models root: pinned Hugging Face snapshots, hash-verified (plan.md Q6).

For each pinned (repo, revision) this tool fills ``<models_root>/models--<org>--<name>/snapshots/<revision>/``,
the layout the service loads from (a snapshot directory named by its commit SHA, design section 4):

* files already in a local Hugging Face cache (``--hf-cache``, read-only) are **copied**;
* files not there are **downloaded** from huggingface.co at the pinned revision;
* every file is checked against the hash the Hub publishes for that revision: sha256 for LFS files, the
  git blob sha1 for small files. A file that does not match is refused, and nothing half-written is left:
  each file is written to a temporary name and then renamed.

A ``manifest.json`` in the models root records what was installed, with every file's hash.

This is a developer tool for this repository's own test models. The operator's install is
``narration-admin install`` (design section 17.8). Standard library only.

Usage::

    python tools/dev_models.py --models-root <models_root> [--hf-cache <hf_hub_cache>] [--only REPO ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HUB = "https://huggingface.co"
CHUNK = 8 * 1024 * 1024


@dataclass(frozen=True)
class Pin:
    repo: str
    revision: str
    files: tuple[str, ...] | None  # None: every file of the revision except docs and other frameworks' weights


PINS: tuple[Pin, ...] = (
    Pin("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "fd4b254389122332181a7c3db7f27e918eec64e3", None),
    Pin("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "5ecdb67327fd37bb2e042aab12ff7391903235d3", None),
    Pin("openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1", None),
    # WavLM: main has only pytorch_model.bin; refs/pr/8 adds safetensors. WP22 pins one of them.
    Pin("microsoft/wavlm-base-plus-sv", "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e", None),
    Pin("microsoft/wavlm-base-plus-sv", "1bfd64eca136543feb28c5ffaf05381c6af33121", ("model.safetensors",)),
    Pin("facebook/wav2vec2-large-960h-lv60-self", "54074b1c16f4de6a5ad59affb4caa8f2ea03a119", None),
    Pin("Qwen/Qwen3-ForcedAligner-0.6B", "c7cbfc2048c462b0d63a45797104fc9db3ad62b7", None),
)

SKIP_NAMES = frozenset({".gitattributes", "README.md"})
SKIP_SUFFIXES = (".h5", ".msgpack", ".ot", ".onnx", ".tflite", ".mlmodel")
SKIP_PREFIXES = ("onnx/", "tf/", "flax/", "coreml/")


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    sha256: str | None  # LFS files
    git_sha1: str  # every file


def fetch_json(url: str) -> object:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def remote_files(pin: Pin) -> list[RemoteFile]:
    url = f"{HUB}/api/models/{pin.repo}/tree/{pin.revision}?recursive=true"
    entries = fetch_json(url)
    assert isinstance(entries, list)
    files = []
    for e in entries:
        if e.get("type") != "file":
            continue
        path = e["path"]
        if pin.files is not None:
            if path not in pin.files:
                continue
        elif path in SKIP_NAMES or path.endswith(SKIP_SUFFIXES) or path.startswith(SKIP_PREFIXES) or ".fp32" in path:
            continue
        lfs = e.get("lfs") or {}
        files.append(RemoteFile(path, int(e["size"]), lfs.get("oid"), e["oid"]))
    # Where safetensors weights exist, the pickled duplicates are not needed (and not loaded).
    if any(f.path.endswith(".safetensors") for f in files) and pin.files is None:
        files = [f for f in files if not (f.path.endswith(".bin") or f.path.endswith(".bin.index.json"))]
    return files


def file_hashes(path: Path) -> tuple[str, str]:
    """Return (sha256, git blob sha1) of a file."""
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            sha256.update(chunk)
            sha1.update(chunk)
    return sha256.hexdigest(), sha1.hexdigest()


def matches(remote: RemoteFile, path: Path) -> tuple[bool, str]:
    if not path.is_file() or path.stat().st_size != remote.size:
        return False, ""
    sha256, sha1 = file_hashes(path)
    ok = sha256 == remote.sha256 if remote.sha256 else sha1 == remote.git_sha1
    return ok, sha256


def copy_or_download(remote: RemoteFile, pin: Pin, source: Path | None, dest: Path) -> str:
    tmp = dest.with_name(dest.name + ".partial")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source is not None and source.is_file():
        how = "copied"
        shutil.copyfile(source, tmp)
    else:
        how = "downloaded"
        quoted = urllib.parse.quote(remote.path)
        url = f"{HUB}/{pin.repo}/resolve/{pin.revision}/{quoted}"
        with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out, CHUNK)
    ok, sha256 = matches(remote, tmp)
    if not ok:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"{pin.repo}@{pin.revision[:8]}: {remote.path} does not match the Hub's hash ({how})")
    os.replace(tmp, dest)
    print(f"  {how}: {remote.path} ({remote.size:,} bytes)", flush=True)
    return sha256


def snapshot_dir(root: Path, pin: Pin) -> Path:
    return root / ("models--" + pin.repo.replace("/", "--")) / "snapshots" / pin.revision


def install(pin: Pin, models_root: Path, hf_cache: Path | None) -> dict[str, object]:
    print(f"{pin.repo} @ {pin.revision}", flush=True)
    dest_dir = snapshot_dir(models_root, pin)
    source_dir = snapshot_dir(hf_cache, pin) if hf_cache else None
    record: dict[str, str] = {}
    for remote in remote_files(pin):
        dest = dest_dir / remote.path
        ok, sha256 = matches(remote, dest)
        if ok:
            print(f"  present: {remote.path}", flush=True)
        else:
            source = source_dir / remote.path if source_dir else None
            sha256 = copy_or_download(remote, pin, source, dest)
        record[remote.path] = sha256
    return {
        "repo": pin.repo,
        "revision": pin.revision,
        "snapshot_dir": dest_dir.relative_to(models_root).as_posix(),
        "files_sha256": record,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--models-root",
        type=Path,
        default=os.environ.get("NARRATION_MODELS_ROOT"),
        help="destination (default: $NARRATION_MODELS_ROOT)",
    )
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=os.environ.get("HF_HUB_CACHE"),
        help="a local Hugging Face hub cache to copy from, read-only (default: $HF_HUB_CACHE)",
    )
    parser.add_argument("--only", nargs="*", default=None, help="install only these repos")
    args = parser.parse_args(argv)
    if args.models_root is None:
        parser.error("--models-root or NARRATION_MODELS_ROOT is required")
    models_root: Path = args.models_root.resolve()
    models_root.mkdir(parents=True, exist_ok=True)

    manifest_path = models_root / "manifest.json"
    manifest: dict[str, object] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for pin in PINS:
        if args.only and pin.repo not in args.only:
            continue
        manifest[f"{pin.repo}@{pin.revision}"] = install(pin, models_root, args.hf_cache)
        tmp = manifest_path.with_suffix(".json.partial")
        tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, manifest_path)
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
