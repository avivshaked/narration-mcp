"""``narration-admin render``: render one text in a voice from the terminal (design sections 7.1, 7.3 to 7.5).

::

    narration-admin render --voice <clip.wav> --transcript <text> --text <text> [--out <take.wav>]

A thin client of the service's own backend (``narration.backend``), with no second path: it sends one
``submit_job`` request (the voice, one segment, the takes), waits for the job with ``get_job``'s long-poll,
reads ``get_results``, and prints each take (its verdict, its delivery file and its cue times). With
``--out`` it copies the suggested take's delivery WAV there. So the request passes the tool's own checks:
the text pipeline, the clip's path, size and sha256 (section 17.3), the synthetic-voices rule (section
17.4), the voice's measurement (``measure_voice`` first) and the limits. The job runs in the daemon, which
the submission starts when none runs (``[daemon] autostart``).

The same command again returns the same job while it runs (section 7.3), so a wait that ends early (``--wait``,
or Ctrl+C) loses nothing: run it again to wait again. A finished job's work is cached, so it renders
nothing twice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

import anyio

from narration.backend import DaemonLauncher, NarrationBackend, backend_for
from narration.contracts.names import TERMINAL_JOB_STATUSES
from narration.mcp.validation import build_validators

from .cli import EXIT_FAILED, EXIT_OK, EXIT_USAGE, PROGRAM, Admin, AdminError, Subparsers

WAIT_S: Final = 1800.0
"""How long ``render`` waits for the job by default: it may wait for the GPU behind other jobs."""
POLL_S: Final = 30.0
"""Each ``get_job`` long-poll's ``wait_s`` (the tool takes at most 55)."""
SEGMENT_ID: Final = "render"
TOOL: Final = "submit_job"
"""The tool whose request ``render`` sends, and whose input schema checks it."""
MAX_TEXT_BYTES: Final = 64 * 1024
"""A text or transcript file is read up to this size; the service's own limits are far below it."""


def register(subparsers: Subparsers) -> None:
    """Add ``render`` (``narration.admin.cli``)."""
    parser = subparsers.add_parser(
        "render",
        help="render one text in a voice, from the terminal",
        description=(
            "Render one text in a voice through the service's own submit_job, wait for the job, and print "
            "each take: its QA verdict, its delivery file and its cue times. The voice must be synthetic "
            "(designed by this service, or in [voices] allow_sha256) and measured (measure_voice) under the "
            "pinned engine. The same command again returns the same job while it runs."
        ),
    )
    parser.add_argument("--voice", required=True, metavar="WAV", help="the voice clip (an absolute path to a WAV)")
    parser.add_argument(
        "--sha256",
        metavar="HEX",
        help="the clip's sha256 (default: computed from the file now)",
    )
    transcript = parser.add_mutually_exclusive_group(required=True)
    transcript.add_argument("--transcript", help="the exact words spoken in the clip")
    transcript.add_argument("--transcript-file", type=Path, metavar="PATH", help="the transcript, from a UTF-8 file")
    text = parser.add_mutually_exclusive_group(required=True)
    text.add_argument("--text", help="the text to speak, as it should be heard")
    text.add_argument("--text-file", type=Path, metavar="PATH", help="the text, from a UTF-8 file")
    parser.add_argument(
        "--takes", type=int, choices=(1, 2, 3), default=None, help="distinct takes (default: [defaults] takes)"
    )
    parser.add_argument("--segment-id", default=SEGMENT_ID, help=f"the segment's name (default {SEGMENT_ID})")
    parser.add_argument("--dry-run", action="store_true", help="plan only: what would be rendered, and estimates")
    parser.add_argument(
        "--wait",
        type=float,
        default=WAIT_S,
        metavar="S",
        help=f"seconds to wait for the job (default {WAIT_S:g}; 0: submit and do not wait)",
    )
    parser.add_argument("--out", type=Path, metavar="WAV", help="copy the suggested take's delivery WAV here")
    parser.add_argument("--force", action="store_true", help="with --out, replace a file that is there")
    parser.add_argument("--json", action="store_true", help="print get_results' JSON instead of a summary")
    parser.set_defaults(handler=render)


# ---------------------------------------------------------------- the request


def _read_text(path: Path, what: str) -> str:
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_TEXT_BYTES + 1)
    except OSError as exc:
        raise AdminError(f"cannot read the {what} file {path}: {exc.strerror or exc}", exit_code=EXIT_USAGE) from exc
    if len(data) > MAX_TEXT_BYTES:
        raise AdminError(f"the {what} file {path} is over {MAX_TEXT_BYTES} bytes", exit_code=EXIT_USAGE)
    try:
        return data.decode("utf-8-sig").strip()
    except UnicodeDecodeError as exc:
        raise AdminError(f"the {what} file {path} is not UTF-8", exit_code=EXIT_USAGE) from exc


def _sha256(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError as exc:
        raise AdminError(f"cannot read the voice clip {path}: {exc.strerror or exc}", exit_code=EXIT_USAGE) from exc


def build_request(args: argparse.Namespace, check_path: Callable[[str], Path]) -> dict[str, Any]:
    """The ``submit_job`` request ``render``'s arguments make: the voice, one segment, the options. It is
    checked against the tool's input schema (``narration.mcp.validation``), so it is refused exactly as the
    tool would refuse it (a ``NarrationError``) before anything is queued.

    Without ``--sha256``, the clip's sha256 is computed here, from the file ``check_path`` (section 17.3,
    ``Platform.check_readable_path``) returns: a path the service would refuse is never opened.
    """
    voice = Path(args.voice).expanduser()
    if not voice.is_absolute():
        voice = Path(os.path.abspath(voice))
    transcript = args.transcript if args.transcript is not None else _read_text(args.transcript_file, "transcript")
    text = args.text if args.text is not None else _read_text(args.text_file, "text")
    sha = args.sha256.lower() if args.sha256 else _sha256(check_path(str(voice)))
    options: dict[str, Any] = {}
    if args.takes is not None:
        options["takes"] = args.takes
    if args.dry_run:
        options["dry_run"] = True
    request: dict[str, Any] = {
        "voice": {"path": str(voice), "sha256": sha, "transcript": transcript},
        "segments": [{"segment_id": args.segment_id, "text": text}],
        "label": f"{PROGRAM} render",
    }
    if options:
        request["options"] = options
    build_validators()[TOOL].validate(request)
    return request


# ---------------------------------------------------------------- the command


def launcher_for(admin: Admin) -> DaemonLauncher | None:
    """The daemon launcher ``render`` uses: None for the backend's own (tests replace it)."""
    return None


def render(admin: Admin, args: argparse.Namespace) -> int:
    """Submit, wait, read the results, and copy the suggested take (see the module docstring)."""
    if args.out is not None and args.out.exists() and not args.force:
        raise AdminError(f"{args.out} is there already; pass --force to replace it", exit_code=EXIT_USAGE)
    request = build_request(args, admin.platform().check_readable_path)
    backend = backend_for(admin.config(), admin.store(), admin.platform(), launcher=launcher_for(admin))
    submitted = backend.submit_job_sync(request)
    for warning in submitted.get("warnings", []):
        admin.warn(f"{warning['severity']}: {warning['code']}: {warning['message']}")
    if args.dry_run:
        _print_plan(admin, submitted, as_json=args.json)
        return EXIT_OK
    job_id = str(submitted["job_id"])
    admin.warn(f"job {job_id}: {submitted['status']}")
    if args.wait <= 0:
        admin.say(job_id)
        return EXIT_OK
    job = _wait(admin, backend, job_id, args.wait)
    if job["status"] not in TERMINAL_JOB_STATUSES:
        raise AdminError(
            f"job {job_id} is still {job['status']} after {args.wait:g} s. It keeps running in the daemon; run "
            "the same command again to wait for it (the identical request returns the same job).",
        )
    results = backend.get_results_sync({"job_id": job_id})
    if args.json:
        admin.say(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        _print_results(admin, results)
    if job["status"] != "completed":
        error = job.get("error") or {}
        admin.warn(f"job {job_id} {job['status']}: {error.get('code', '')} {error.get('message', '')}".rstrip())
        return EXIT_FAILED
    if args.out is not None:
        return _copy_out(admin, results, args.out)
    return EXIT_OK


def _wait(admin: Admin, backend: NarrationBackend, job_id: str, wait_s: float) -> dict[str, Any]:
    """``get_job``, long-polled until the job has finished or ``wait_s`` has passed; progress to stderr."""
    said: list[str | None] = [None]

    async def progress(done: float, total: float | None, message: str | None) -> None:
        if message and message != said[0]:
            said[0] = message
            share = f" ({done / total:.0%})" if total else ""
            admin.warn(f"  {message}{share}")

    async def main() -> dict[str, Any]:
        deadline = anyio.current_time() + wait_s
        job: dict[str, Any] = {}
        while True:
            remaining = deadline - anyio.current_time()
            args = {"job_id": job_id, "wait_s": max(0.0, min(POLL_S, remaining))}
            job = await backend.get_job(args, progress)
            if job["status"] in TERMINAL_JOB_STATUSES or remaining <= 0:
                return job

    return anyio.run(main)


def _print_plan(admin: Admin, submitted: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        admin.say(json.dumps(submitted, ensure_ascii=False, indent=2))
        return
    plan = submitted["plan"]
    admin.say(
        f"would render {plan['renders_needed']}, post-process {plan['deliveries_needed']} and score "
        f"{plan['analyses_needed']} (about {plan['est_audio_s']:.1f} s of audio, {plan['est_wall_s']:.0f} s)"
    )
    for text in submitted.get("text", []):
        admin.say(f"{text['segment_id']}: {text['spoken_chars']} spoken characters")
        for cue in text["cues"]:
            admin.say(f"  engine text: {cue['engine']}")


def _print_results(admin: Admin, results: Mapping[str, Any]) -> None:
    job = results["job"]
    admin.say(f"job {job['job_id']}: {job['status']}" + (f" ({job['outcome']})" if job.get("outcome") else ""))
    for segment in results.get("segments", []):
        admin.say(f"{segment['segment_id']}: {segment['status']}, suggested take {segment['suggested_take_id']}")
        for take in segment["takes"]:
            qa = take.get("qa") or {}
            delivery = take["delivery"]
            admin.say(
                f"  take {take['take_id']} (attempt {take['attempt']}): {qa.get('verdict', 'not scored')}, "
                f"{delivery['duration_s']:.2f} s"
            )
            admin.say(f"    {delivery['path']}")
            admin.say(f"    sha256 {delivery['sha256']}")
            for cue in take["cues"]:
                start, end = cue["start_s"], cue["end_s"]
                where = f"{start:.2f}-{end:.2f} s" if start is not None and end is not None else "not placed"
                admin.say(f"    cue {cue['index']}: {where}")
            for flag in [*qa.get("flags", []), *take.get("flags", [])]:
                admin.say(f"    {flag['severity']}: {flag['code']}: {flag['message']}")
    if results.get("report_md"):
        admin.say(f"report: {results['report_md']}")


def _copy_out(admin: Admin, results: Mapping[str, Any], out: Path) -> int:
    """Copy the suggested take's delivery WAV to ``out`` (a temporary name, then a rename)."""
    for segment in results.get("segments", []):
        take = next((t for t in segment["takes"] if t["take_id"] == segment["suggested_take_id"]), None)
        if take is None:
            admin.warn(f"{segment['segment_id']}: no take is suggested (none passed QA); nothing was copied")
            return EXIT_FAILED
        source = Path(take["delivery"]["path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.name}.{secrets.token_hex(4)}.tmp")
        try:
            shutil.copyfile(source, tmp)
            os.replace(tmp, out)
        except OSError as exc:
            raise AdminError(f"cannot copy {source} to {out}: {exc.strerror or exc}") from exc
        finally:
            tmp.unlink(missing_ok=True)
        admin.say(f"wrote {out}")
        return EXIT_OK
    admin.warn("the job has no segment to copy")
    return EXIT_FAILED


__all__ = ["POLL_S", "WAIT_S", "build_request", "register", "render"]
