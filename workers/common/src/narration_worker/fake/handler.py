"""The ``fake`` role: every op of both real workers, with deterministic synthetic outputs (plan.md WP16).

It lets the daemon, the job engine and the front-end be built and tested with no model and no GPU, and it
keeps the contract the real workers keep (``handler.py``): model ops need ``load`` first, ``load`` checks
the snapshot directories it is given, paths stay in the store.

- ``load`` checks what the real workers check, so a daemon that sends a load a real worker would refuse is
  refused here too (``op_load``). A Qwen load (one that names ``model``) is checked as the ``qwen3`` worker
  checks it, including its ceiling, ``settings.generation.max_new_tokens``. A QA load (``models``) and the
  bare ``{"device": …}`` of the fake's own tests are loads the ``qwen3`` worker never sees; for them the
  ceiling is 8192 unless they carry ``settings``. The reply's ``vram_mb`` is null: the fake holds nothing on
  a GPU, and the protocol allows null, which is what ``qwen3`` replies for a CPU load.
- ``synthesize`` and ``design`` write a real float32 mono WAV at 24 kHz whose length follows the text
  (``audio.py``) with the workers' shared, byte-reproducible writer (``narration_worker.wav``), and record
  what they said (``registry.py``). They count tokens as the real worker does
  (``protocol.AudioReply``): a take of F frames at 12.5 per second takes F + 1 talker steps, the last one
  the end token. Under the call's ``max_new_tokens`` (2 to the loaded ceiling), a cap of F + 1 or more
  leaves the take whole (``new_tokens`` F), and a cap C of F or less cuts it to C - 1 frames with
  ``hit_token_cap: true``. The reply echoes the cap it applied as ``max_new_tokens``. A take that ends under
  its cap is the same under any cap. A ``token_cap`` fault renders the call as if it had the fault's cap,
  reply included.
- ``transcribe`` hears the text back, with word times, from the take or from any post-processed copy of it.
- ``embed`` gives a 512-dimension unit vector: the voice's direction plus a small per-take part, so takes of
  one voice (and the clip they clone) are about 0.99 similar and different voices are not.
- ``f0`` reports the tone's pitch, ``align`` places the tokens over the words' times, ``profile`` computes
  plausible numbers and writes two small PNGs.

The same request gives the same bytes and the same reply, in any process. Planted faults (``faults.py``)
come only from the file named by ``NARRATION_FAKE_SPEC`` in the worker's environment.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import os
import re
import struct
import subprocess
import sys
import time
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from narration_worker.determinism import parse_determinism
from narration_worker.errors import OpError
from narration_worker.handler import (
    MIN_MAX_NEW_TOKENS,
    Request,
    WorkerContext,
    WorkerHandler,
    require_bool,
    require_int,
    require_max_new_tokens,
    require_number,
    require_one_of,
    require_str,
    require_str_list,
)
from narration_worker.protocol import Controls, WorkerErrorCode
from narration_worker.wav import write_float32_mono

from .audio import (
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    SEGMENT,
    SEGMENT_S,
    Heard,
    decode,
    layout,
    new_tokens,
    render,
    spoken_tokens,
    symbol_hz,
    symbol_of,
    truncate,
)
from .faults import CONTENT_KINDS, REQUEST_KINDS, Facts, Fault, FaultSpec, SpecError, SpecSource, Tally
from .registry import RECORD_SCHEMA, Registry
from .wav import Audio, WavError, read_wav

log = logging.getLogger(__name__)

FAKE_REVISION: Final = hashlib.sha1(b"narration-worker fake 1").hexdigest()
"""A 40-hex revision for the fake's "models", so the server can treat them like pinned snapshots."""
FAKE_ASR_MODEL: Final = "narration-worker/fake-asr"
FAKE_SV_MODEL: Final = "narration-worker/fake-sv"
FAKE_ALIGNER_MODEL: Final = "narration-worker/fake-ctc"
EMBEDDING_DIM: Final = 512
"""WavLM-base-plus-sv's x-vector size."""
DEFAULT_MAX_NEW_TOKENS: Final = 8192
"""The ceiling of a load that is not a Qwen load and carries no ``settings`` (a QA load, or the bare
``{"device": …}`` of the fake's own tests): the pinned Qwen snapshots' value (plan.md 1.3 item 1). A Qwen load
must give its ceiling, as the ``qwen3`` worker requires."""
CEILING_FIELD: Final = "settings.generation.max_new_tokens"
"""The ``details.field`` of a refused ceiling, as the ``qwen3`` worker names it."""
REVISION_PATTERN: Final = re.compile(r"[0-9a-f]{40}")
"""A snapshot's revision: a 40-hex commit SHA, which names its folder (design section 4)."""
DEVICE_PATTERN: Final = re.compile(r"cpu|cuda(:\d+)?")
"""The devices a real worker's ``load`` accepts."""
ALIGN_FRAME_S: Final = 0.02
"""wav2vec2's frame: 320 samples at 16 kHz."""
F0_HOP_S: Final = 0.01
PROFILE_KEYS: Final = (
    "duration_s",
    "pitch_median_hz",
    "pitch_p10_hz",
    "pitch_p90_hz",
    "pitch_range_st",
    "speaking_rate_wpm",
    "pause_ratio",
    "loudness_lufs",
    "spectral_centroid_hz",
    "hnr_db",
    "cpps_db",
)
"""The voice profile's measurements (the server's ``models.ProfileMeasurements``)."""
PROFILE_PICTURES: Final = ("spectrogram", "pitch")

_TAKE_VARIATION: Final = 0.12
_REAL_TIME_FACTOR: Final = 0.25


class FakeHandler(WorkerHandler):
    """The fake worker. See the module docstring."""

    role = "fake"
    uses_torch = False
    controls = Controls(pace=False, context=False, instruct=False)

    def __init__(self, context: WorkerContext) -> None:
        super().__init__(context)
        self.registry = Registry(context.store_root)
        self.tally = Tally(context.store_root)
        self.spec_source = SpecSource()
        self.spec_source.current()  # a broken spec crashes the worker at start-up (SpecError)
        self.loaded = False
        self.max_new_tokens = DEFAULT_MAX_NEW_TOKENS
        self.voices: dict[str, str] = {}

    # ------------------------------------------------------------------ planted faults
    def before_request(self, op: str, request: Request) -> bytes | None:
        spec = self._spec()
        faults = spec.for_op(op, REQUEST_KINDS)
        if not faults:
            return None
        facts = self._facts(op, request) if spec.needs_facts(op) else Facts()
        raw: bytes | None = None
        for fault in faults:
            if fault.matches(facts) and self.tally.claim(fault):
                raw = self._fire(fault, op) or raw
        return raw

    def _spec(self) -> FaultSpec:
        try:
            return self.spec_source.current()
        except SpecError as exc:
            raise OpError("INTERNAL", f"the fake worker's spec is invalid: {exc}") from exc

    def _fire(self, fault: Fault, op: str) -> bytes | None:
        params = fault.params
        log.warning("planted fault %s on %s", fault.kind, op)
        if fault.kind == "crash":
            code = int(params.get("exit_code", 3))
            log.error("planted crash: exiting with code %d in the middle of %s", code, op)
            for handler in logging.getLogger().handlers:
                handler.flush()
            sys.stderr.flush()
            os._exit(code)
        if fault.kind == "hang":
            while True:
                time.sleep(60)
        if fault.kind == "delay":
            time.sleep(float(params["seconds"]))
            return None
        if fault.kind == "gpu_oom":
            raise OpError("GPU_OOM", "CUDA out of memory (planted by the fake worker)", {"planted": True})
        if fault.kind == "error":
            code = cast(WorkerErrorCode, params["code"])
            raise OpError(code, str(params.get("message", "planted by the fake worker")), {"planted": True})
        if fault.kind == "alignment_error":
            raise OpError(
                "ALIGNMENT_ERROR", "forced alignment failed (planted by the fake worker)", {"reason": "planted"}
            )
        if fault.kind == "protocol_break":
            return b"the fake worker planted this line: it is not a protocol message\n"
        self._make_noise()
        return None

    @staticmethod
    def _make_noise() -> None:
        print("fake worker: planted noise on sys.stdout", flush=True)
        if sys.__stdout__ is not None:
            sys.__stdout__.write("fake worker: planted noise on sys.__stdout__\n")
            sys.__stdout__.flush()
        os.write(1, b"fake worker: planted noise on file descriptor 1\n")
        subprocess.run(
            [sys.executable, "-c", "print('fake worker: planted noise from a child process')"],
            stdin=subprocess.DEVNULL,
            check=False,
            timeout=60,
        )

    def _facts(self, op: str, request: Request) -> Facts:
        if op == "synthesize":
            return Facts(_str(request, "engine_text"), _int(request, "seed"), _str(request, "voice_hash"))
        if op == "design":
            return Facts(_str(request, "design_text"), _int(request, "seed"))
        if op == "prepare_voice":
            return Facts(_str(request, "ref_text"), None, _str(request, "voice_hash"))
        if "wav" in request and self.loaded:
            try:
                record, _ = self._recognise(self._read(self.input_file(request, "wav")))
            except OpError:
                return Facts()
            if record is not None:
                return Facts(record["text"], record["seed"], record["voice_hash"])
        return Facts()

    # ------------------------------------------------------------------ load / unload
    def op_load(self, request: Request) -> dict[str, Any]:
        """Check a ``load`` as the real workers check it, then "load" (nothing is read, and no VRAM is used).

        In the ``qwen3`` worker's order (``narration_qwen3tts.worker``), each refusal with its code and
        ``details.field``:

        1. Each snapshot reference (a Qwen load's ``model``, a QA load's ``models`` by use) is an object with
           ``repo``, ``revision`` and an absolute ``snapshot_dir`` (``INVALID_REQUEST``). Every snapshot folder
           exists before anything else is checked (``BACKEND_NOT_INSTALLED``); then each revision is a 40-hex
           SHA that names its folder (section 4).
        2. ``device`` is ``cpu``, ``cuda`` or ``cuda:<n>``.
        3. A Qwen load carries ``dtype``, ``attn_implementation`` and ``determinism``, and ``determinism`` is
           valid wherever it is given.
        4. A Qwen load, and any load that carries ``settings``, gives the ceiling of every call's cap,
           ``settings.generation.max_new_tokens``: an integer of at least 2 (qwen-tts's ``min_new_tokens``),
           never defaulted (section 10.1, DC-4). Any other load's ceiling is ``DEFAULT_MAX_NEW_TOKENS``.

        Not checked, as ``qwen3`` checks them: the values of ``dtype`` and ``attn_implementation``, the other
        nine sampling values and ``non_streaming_mode``, and the snapshot's files. A refused load changes
        nothing; a load that passes clears the prepared voices.
        """
        refs = _snapshot_refs(request)
        for name, ref in refs:
            if not Path(ref["snapshot_dir"]).is_dir():
                raise OpError(
                    "BACKEND_NOT_INSTALLED",
                    f"no snapshot of {ref['repo']} at {ref['snapshot_dir']}; install the models (narration-admin "
                    "install)",
                    {"field": name, "repo": ref["repo"], "snapshot_dir": ref["snapshot_dir"]},
                )
        for name, ref in refs:
            _check_revision(name, ref)
        device = require_str(request, "device")
        if DEVICE_PATTERN.fullmatch(device) is None:
            raise OpError("INVALID_REQUEST", "device must be cpu, cuda or cuda:<n>", {"field": "device"})
        qwen = "model" in request
        if qwen:
            require_str(request, "dtype")
            require_str(request, "attn_implementation")
            if "determinism" not in request:
                raise OpError(
                    "INVALID_REQUEST",
                    "a Qwen load needs the determinism switches (section 10.1)",
                    {"field": "determinism"},
                )
        if "determinism" in request:
            parse_determinism(request["determinism"])
        cap = _ceiling(request) if qwen or "settings" in request else DEFAULT_MAX_NEW_TOKENS
        self.max_new_tokens = cap
        self.loaded = True
        self.voices.clear()
        return {"load_s": 0.0, "vram_mb": None}

    def op_unload(self, request: Request) -> dict[str, Any]:
        self.shutdown()
        return {}

    def shutdown(self) -> None:
        self.loaded = False
        self.voices.clear()

    def _need_loaded(self, op: str) -> None:
        if not self.loaded:
            raise OpError("NOT_LOADED", f"{op} needs a loaded model: send load first")

    # ------------------------------------------------------------------ speaking
    def op_prepare_voice(self, request: Request) -> dict[str, Any]:
        self._need_loaded("prepare_voice")
        voice_hash = require_str(request, "voice_hash")
        require_str(request, "ref_text", allow_empty=True)
        require_bool(request, "x_vector_only_mode")
        path = self.input_file(request, "ref_wav")
        record, _ = self._recognise(self._read(path))
        self.voices[voice_hash] = record["voice_key"] if record else "file:" + _sha256(path)[:16]
        return {}

    def op_synthesize(self, request: Request) -> dict[str, Any]:
        self._need_loaded("synthesize")
        voice_hash = require_str(request, "voice_hash")
        text = require_str(request, "engine_text")
        language = require_str(request, "language")
        seed = require_int(request, "seed", minimum=0, maximum=0xFFFFFFFF)
        cap = require_max_new_tokens(request, self.max_new_tokens)
        out = self.output_file(request, "out_path")
        voice_key = self.voices.get(voice_hash)
        if voice_key is None:
            raise OpError(
                "VOICE_NOT_PREPARED", "send prepare_voice for this voice_hash first", {"voice_hash": voice_hash}
            )
        return self._speak("synthesize", text, language, seed, cap, voice_hash, voice_key, out, "engine_text")

    def op_design(self, request: Request) -> dict[str, Any]:
        self._need_loaded("design")
        description = require_str(request, "description")
        text = require_str(request, "design_text")
        language = require_str(request, "language")
        seed = require_int(request, "seed", minimum=0, maximum=0xFFFFFFFF)
        cap = require_max_new_tokens(request, self.max_new_tokens)
        out = self.output_file(request, "out_path")
        voice_key = "design:" + hashlib.sha256(f"{description}\x00{seed}".encode()).hexdigest()[:16]
        return self._speak("design", text, language, seed, cap, None, voice_key, out, "design_text")

    def _speak(
        self,
        op: str,
        text: str,
        language: str,
        seed: int,
        cap: int,
        voice_hash: str | None,
        voice_key: str,
        out: Path,
        field: str,
    ) -> dict[str, Any]:
        """Render a take of ``text`` under ``cap``, the call's ``max_new_tokens`` (see the module docstring for
        how the cap counts). A ``token_cap`` fault models the call having had the fault's cap, if that is
        lower: the take is cut there and the reply echoes it, as a real worker's would."""
        tokens = spoken_tokens(text)
        if not tokens:
            raise OpError("INVALID_REQUEST", f"{field} has no words to speak", {"field": field})
        # one entry per word the take says: (the word, is it one of the text's words?, was it planted?)
        entries: list[tuple[str, bool, bool]] = [(token, True, False) for token in tokens]
        heard_text: str | None = None
        cap_fault: Fault | None = None
        planted: list[dict[str, Any]] = []
        for fault in self._content_faults(op, Facts(text, seed, voice_hash)):
            planted.append({"kind": fault.kind, **fault.params})
            if fault.kind == "say":
                said = spoken_tokens(str(fault.params["text"]))
                if not said:
                    raise OpError("INTERNAL", "the fake spec's say.text has no words to speak")
                entries = [(w, True, i >= len(tokens) or w != tokens[i]) for i, w in enumerate(said)]
                heard_text = fault.params.get("heard")
            elif fault.kind == "wrong_word":
                entries = _replace_word(entries, fault.params)
            elif fault.kind == "head_insertion":
                entries = [(w, False, True) for w in fault.params.get("words", ["so"])] + entries
            elif fault.kind == "end_insertion":
                entries = entries + [(w, False, True) for w in fault.params.get("words", ["okay"])]
            else:
                cap_fault = fault
        bursts, total = layout([said for said, _, _ in entries])
        full = [b.segments for b in bursts]
        natural = new_tokens(total)  # F frames: F + 1 talker steps, the last one the end token
        if cap_fault is not None:  # the call is rendered as if its cap were the fault's, when that is lower
            cap = min(cap, int(cap_fault.params.get("max_new_tokens", max(2, natural * 6 // 10))))
        hit = cap <= natural  # the end token would come at step F + 1, past the cap
        frames = cap - 1 if hit else natural
        if hit:
            bursts, total = truncate(bursts, frames * SAMPLES_PER_FRAME)
        identity = {
            "op": op,
            "voice_key": voice_key,
            "voice_hash": voice_hash,
            "text": text,
            "language": language,
            "seed": seed,
            "cap": cap if hit else None,
            "planted": planted,
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")).digest()
        preferred_id = int.from_bytes(digest[:4], "big")
        whole = [i for i in range(len(bursts)) if bursts[i].segments == full[i]]
        if heard_text is None:
            words = [_heard_word(entries[i][0], i, i, 0, 1, entries[i][2]) for i in whole]
        else:
            words = _map_heard(heard_text.split(), whole, {w.lower() for w, _, _ in entries})
        record = {
            "schema": RECORD_SCHEMA,
            "kind": op,
            "voice_key": voice_key,
            "voice_hash": voice_hash,
            "text": text,
            "language": language,
            "seed": seed,
            "sample_rate": SAMPLE_RATE,
            "samples": total,
            "bursts": [[b.start, b.segments] for b in bursts],
            "words": words,
            "text_bursts": [i for i in range(len(bursts)) if entries[i][1]],
            "planted": planted,
            "hit_token_cap": hit,
        }
        utterance_id = self.registry.claim(preferred_id, record)  # the next free id if another utterance has it
        write_float32_mono(out, render(bursts, total, utterance_id), SAMPLE_RATE)
        return {
            "sample_rate": SAMPLE_RATE,
            "samples": total,
            "gen_s": round(total / SAMPLE_RATE * _REAL_TIME_FACTOR, 3),
            "hit_token_cap": hit,
            "max_new_tokens": cap,
            "new_tokens": frames,
        }

    def _content_faults(self, op: str, facts: Facts) -> list[Fault]:
        spec = self._spec()
        return [f for f in spec.for_op(op, CONTENT_KINDS) if f.matches(facts) and self.tally.claim(f)]

    # ------------------------------------------------------------------ hearing
    def op_transcribe(self, request: Request) -> dict[str, Any]:
        self._need_loaded("transcribe")
        path = self.input_file(request, "wav")
        require_str(request, "language")
        with_times = require_bool(request, "word_timestamps")
        require_bool(request, "long_form")
        record, heard = self._recognise(self._read(path))
        if record is None:
            canned = self._spec().transcripts.get(_sha256(path))
            if canned is None:
                raise OpError(
                    "UNSUPPORTED_AUDIO",
                    "the fake worker transcribes only audio it rendered, or a file its spec's transcripts list",
                    {"field": "wav"},
                )
            words = [{"text": w, "start_s": None, "end_s": None, "probability": None} for w in canned.split()]
            return {"text": canned, "words": words, "model": FAKE_ASR_MODEL, "revision": FAKE_REVISION}
        times = _burst_times(record, heard)
        words = []
        for word in record["words"]:
            start, end = times[word["burst"]][0], times[word["last_burst"]][1]
            width = (end - start) / word["parts"]
            start, end = start + width * word["part"], start + width * (word["part"] + 1)
            words.append(
                {
                    "text": word["text"],
                    "start_s": round(start, 3) if with_times else None,
                    "end_s": round(end, 3) if with_times else None,
                    "probability": 0.55 if word["planted"] else 0.99,
                }
            )
        text = " ".join(w["text"] for w in words)
        return {"text": text, "words": words, "model": FAKE_ASR_MODEL, "revision": FAKE_REVISION}

    def op_embed(self, request: Request) -> dict[str, Any]:
        self._need_loaded("embed")
        path = self.input_file(request, "wav")
        require_one_of(request, "device", ("cuda", "cpu"))
        record, _ = self._recognise(self._read(path))
        if record is not None:
            voice, take = record["voice_key"], "utterance:" + record["id"]
        else:
            digest = _sha256(path)
            voice, take = "file:" + digest[:16], "file:" + digest
        base = _unit_vector(voice)
        variation = _unit_vector(take)
        embedding = _normalise([a + _TAKE_VARIATION * b for a, b in zip(base, variation, strict=True)])
        return {"embedding": embedding, "dim": EMBEDDING_DIM, "model": FAKE_SV_MODEL, "revision": FAKE_REVISION}

    def op_f0(self, request: Request) -> dict[str, Any]:
        self._need_loaded("f0")
        path = self.input_file(request, "wav")
        fmin = require_number(request, "fmin_hz")
        fmax = require_number(request, "fmax_hz")
        if not 0 < fmin < fmax:
            raise OpError("INVALID_REQUEST", "fmin_hz must be positive and below fmax_hz", {"field": "fmin_hz"})
        audio = self._read(path)
        track = _pitch_track(audio, decode(audio), _sha256(path))
        f0 = [hz if hz is not None and fmin <= hz <= fmax else None for hz in track]
        probability = [0.95 if hz is not None else 0.02 for hz in f0]
        return {"hop_s": F0_HOP_S, "f0_hz": f0, "voiced_probability": probability, "method": "fake-pitch"}

    def op_align(self, request: Request) -> dict[str, Any]:
        self._need_loaded("align")
        path = self.input_file(request, "wav")
        tokens = require_str_list(request, "tokens")
        audio = self._read(path)
        record, heard = self._recognise(audio)
        frames = ctc_frames(audio.duration_s)
        repeats = sum(1 for a, b in itertools.pairwise(tokens) if a == b)
        if not tokens or frames < len(tokens) + repeats:
            raise OpError(
                "ALIGNMENT_ERROR",
                "the audio is too short for the tokens (frames < tokens + repeats)" if tokens else "no tokens to align",
                {
                    "reason": "too_short" if tokens else "no_tokens",
                    "frames": frames,
                    "tokens": len(tokens),
                    "repeats": repeats,
                },
            )
        if record is not None:
            times = _burst_times(record, heard)
            words = [times[i] for i in record["text_bursts"]]
        else:
            words = [
                (start / heard.sample_rate, (start + n * heard.sample_rate * SEGMENT_S) / heard.sample_rate)
                for start, n in heard.bursts
            ]
        if not words:
            words = [(0.0, audio.duration_s)]
        spans = place_tokens(tokens, words, frames)
        return {
            "frame_s": ALIGN_FRAME_S,
            "num_frames": frames,
            "spans": spans,
            "model": FAKE_ALIGNER_MODEL,
            "revision": FAKE_REVISION,
            "device": "cpu",
        }

    def op_profile(self, request: Request) -> dict[str, Any]:
        self._need_loaded("profile")
        path = self.input_file(request, "wav")
        out_dir = self.output_dir(request, "out_dir")
        transcript = request.get("transcript")
        if transcript is not None and not isinstance(transcript, str):
            raise OpError("INVALID_REQUEST", "transcript must be a string or null", {"field": "transcript"})
        audio = self._read(path)
        heard = decode(audio)
        track = _pitch_track(audio, heard, _sha256(path))
        measurements = _profile_numbers(audio, heard, track, transcript)
        pictures: dict[str, str] = {}
        for name in PROFILE_PICTURES:
            picture = out_dir / f"{name}.png"
            _write_png(picture, track if name == "pitch" else None)
            pictures[name] = str(picture)
        method = {"f0": "fake-pitch", "hnr": "fake", "cpps": "fake", "loudness": "fake-mean-square"}
        return {"measurements": measurements, "pictures": pictures, "method": method}

    # ------------------------------------------------------------------ helpers
    def _read(self, path: Path) -> Audio:
        try:
            return read_wav(path)
        except WavError as exc:
            raise OpError("UNSUPPORTED_AUDIO", str(exc), {"path": str(path)}) from exc

    def _recognise(self, audio: Audio) -> tuple[dict[str, Any] | None, Heard]:
        """The registry record of an utterance the fake rendered, or None; and what was heard."""
        heard = decode(audio)
        if heard.utterance_id is None:
            return None, heard
        record = self.registry.get(heard.utterance_id)
        if record is None:
            return None, heard
        if [n for _, n in record["bursts"]] != [n for _, n in heard.bursts]:
            log.warning("utterance %08x: the audio's bursts do not match the record", heard.utterance_id)
            return None, heard
        return record, heard


# ---------------------------------------------------------------------- pure helpers


def _str(request: Request, name: str) -> str | None:
    value = request.get(name)
    return value if isinstance(value, str) else None


def _int(request: Request, name: str) -> int | None:
    value = request.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _snapshot_refs(request: Request) -> list[tuple[str, dict[str, str]]]:
    """A load's snapshot references as (field, ref): ``model``, then ``models.<use>``. Each must be an object
    with ``repo``, ``revision`` and an absolute ``snapshot_dir`` (``INVALID_REQUEST``, as ``qwen3`` checks
    ``model``)."""
    refs: list[tuple[str, object]] = []
    if "model" in request:
        refs.append(("model", request["model"]))
    if "models" in request:
        models = request["models"]
        if not isinstance(models, dict):
            raise OpError("INVALID_REQUEST", "models must be an object", {"field": "models"})
        refs.extend((f"models.{use}", ref) for use, ref in cast(dict[str, object], models).items())
    checked: list[tuple[str, dict[str, str]]] = []
    for name, ref in refs:
        if not isinstance(ref, dict) or not all(
            isinstance(ref.get(k), str) for k in ("repo", "revision", "snapshot_dir")
        ):
            raise OpError("INVALID_REQUEST", f"{name} must have repo, revision and snapshot_dir", {"field": name})
        ref = cast(dict[str, str], ref)
        if not Path(ref["snapshot_dir"]).is_absolute():
            field = f"{name}.snapshot_dir"
            raise OpError("INVALID_REQUEST", f"{field} must be an absolute path", {"field": field})
        checked.append((name, ref))
    return checked


def _check_revision(name: str, ref: Mapping[str, str]) -> None:
    """A snapshot's revision is a 40-hex SHA and names its folder (design section 4), as ``qwen3`` checks."""
    revision = ref["revision"]
    if REVISION_PATTERN.fullmatch(revision) is None:
        field = f"{name}.revision"
        raise OpError("INVALID_REQUEST", f"{field} must be a 40-hex commit SHA", {"field": field})
    folder = Path(ref["snapshot_dir"]).name
    if folder != revision:
        raise OpError(
            "INVALID_REQUEST",
            "the snapshot folder must be named by its revision (section 4)",
            {"field": f"{name}.snapshot_dir", "revision": revision, "folder": folder},
        )


def _ceiling(request: Request) -> int:
    """``settings.generation.max_new_tokens``, required as the ``qwen3`` worker requires it: an integer of at
    least ``MIN_MAX_NEW_TOKENS``, else ``INVALID_REQUEST`` with ``details.field`` naming the missing or bad
    member (``settings``, ``settings.generation`` or ``CEILING_FIELD``)."""
    if "settings" not in request:
        raise OpError(
            "INVALID_REQUEST", "every audio-changing setting must be passed (section 10.1)", {"field": "settings"}
        )
    settings = request["settings"]
    if not isinstance(settings, dict):
        raise OpError(
            "INVALID_REQUEST",
            "settings must be an object with non_streaming_mode and generation",
            {"field": "settings"},
        )
    generation = cast(dict[str, object], settings).get("generation")
    if not isinstance(generation, dict):
        raise OpError(
            "INVALID_REQUEST",
            "settings.generation must be an object with every sampling value (section 10.1)",
            {"field": "settings.generation"},
        )
    generation = cast(dict[str, object], generation)
    if "max_new_tokens" not in generation:
        raise OpError(
            "INVALID_REQUEST",
            f"{CEILING_FIELD} must be passed explicitly (section 10.1): it is the ceiling of every call's cap",
            {"field": CEILING_FIELD},
        )
    ceiling = generation["max_new_tokens"]
    if not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling < MIN_MAX_NEW_TOKENS:
        raise OpError(
            "INVALID_REQUEST",
            f"{CEILING_FIELD} must be an integer of at least {MIN_MAX_NEW_TOKENS} (qwen-tts's min_new_tokens)",
            {"field": CEILING_FIELD},
        )
    return ceiling


def _replace_word(entries: list[tuple[str, bool, bool]], params: Mapping[str, Any]) -> list[tuple[str, bool, bool]]:
    """Replace one of the text's words (``word``: an index into them; the middle one by default)."""
    positions = [i for i, (_, is_text, _) in enumerate(entries) if is_text]
    index = int(params.get("word", len(positions) // 2))
    index = min(max(index + len(positions) if index < 0 else index, 0), len(positions) - 1)
    target = positions[index]
    original = entries[target][0]
    core_end = max((i + 1 for i, ch in enumerate(original) if ch.isalnum()), default=len(original))
    replaced = (str(params.get("replacement", "wrong")) + original[core_end:], True, True)
    return [replaced if i == target else entry for i, entry in enumerate(entries)]


def _heard_word(text: str, burst: int, last_burst: int, part: int, parts: int, planted: bool) -> dict[str, Any]:
    """A transcript word: the bursts it spans, and which of ``parts`` equal shares of them it takes."""
    return {"text": text, "burst": burst, "last_burst": last_burst, "part": part, "parts": parts, "planted": planted}


def _map_heard(heard: Sequence[str], bursts: Sequence[int], said: set[str]) -> list[dict[str, Any]]:
    """Spread a given transcript's words over the take's bursts in order: one each when the counts match,
    several bursts per word when fewer words are heard, equal shares of a burst when more are."""
    count, total = len(heard), len(bursts)
    if not count or not total:
        return []
    spans = [(k * total // count, max(k * total // count, (k + 1) * total // count - 1)) for k in range(count)]
    words = []
    for k, token in enumerate(heard):
        sharing = [j for j, span in enumerate(spans) if span == spans[k]]
        first, last = spans[k]
        words.append(
            _heard_word(token, bursts[first], bursts[last], sharing.index(k), len(sharing), token.lower() not in said)
        )
    return words


def _burst_times(record: dict[str, Any], heard: Heard) -> list[tuple[float, float]]:
    """Each recorded burst's (start, end) in the heard file's seconds (the record's layout, shifted)."""
    rate = record["sample_rate"]
    offset = heard.bursts[0][0] / heard.sample_rate - record["bursts"][0][0] / rate
    return [(start / rate + offset, (start + n * SEGMENT) / rate + offset) for start, n in record["bursts"]]


def _unit_vector(key: str) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < EMBEDDING_DIM:
        digest = hashlib.sha256(f"{key}\x00{counter}".encode()).digest()
        values.extend((b - 127.5) / 127.5 for b in digest)
        counter += 1
    return _normalise(values[:EMBEDDING_DIM])


def _normalise(values: Sequence[float]) -> list[float]:
    norm = math.sqrt(math.fsum(v * v for v in values))
    return [v / norm for v in values]


def _pitch_track(audio: Audio, heard: Heard, file_sha256: str) -> list[float | None]:
    """The pitch every ``F0_HOP_S``: each segment's tone for the fake's own audio, a nominal pitch in the
    tone bursts of anything else, and None in silence."""
    frames = max(1, math.ceil(audio.duration_s / F0_HOP_S))
    track: list[float | None] = [None] * frames
    nominal = 100.0 + int(file_sha256[:4], 16) % 150
    rate = heard.sample_rate
    seg = rate * SEGMENT_S
    index = 0
    for start, count in heard.bursts:
        for k in range(count):
            hz = symbol_hz(symbol_of(heard.utterance_id, index + k)) if heard.utterance_id is not None else nominal
            first = math.ceil((start + k * seg) / rate / F0_HOP_S)
            last = min(frames, math.ceil((start + (k + 1) * seg) / rate / F0_HOP_S))
            for frame in range(first, last):
                track[frame] = hz
        index += count
    return track


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    return sorted_values[round(q * (len(sorted_values) - 1))]


def _profile_numbers(
    audio: Audio, heard: Heard, track: Sequence[float | None], transcript: str | None
) -> dict[str, float | None]:
    voiced = sorted(v for v in track if v is not None)
    median = _quantile(voiced, 0.5) if voiced else None
    p10 = _quantile(voiced, 0.1) if voiced else None
    p90 = _quantile(voiced, 0.9) if voiced else None
    flags = [v is not None for v in track]
    if any(flags):
        first = flags.index(True)
        last = len(flags) - 1 - flags[::-1].index(True)
        span = flags[first : last + 1]
        pause_ratio = round(span.count(False) / len(span), 4)
    else:
        pause_ratio = 1.0
    rate = heard.sample_rate
    energy = [v * v for start, n in heard.bursts for v in audio.samples[start : start + round(n * rate * SEGMENT_S)]]
    mean_square = math.fsum(energy) / len(energy) if energy else 0.0
    duration = audio.duration_s
    words = len(transcript.split()) if transcript else 0
    return {
        "duration_s": round(duration, 4),
        "pitch_median_hz": median,
        "pitch_p10_hz": p10,
        "pitch_p90_hz": p90,
        "pitch_range_st": round(12 * math.log2(p90 / p10), 3) if p10 and p90 else None,
        "speaking_rate_wpm": round(words / duration * 60, 2) if words and duration > 0 else None,
        "pause_ratio": pause_ratio,
        "loudness_lufs": round(10 * math.log10(mean_square) - 0.691, 2) if mean_square > 0 else None,
        "spectral_centroid_hz": round((median or 150.0) * 1.4, 1),
        "hnr_db": 30.0 if voiced else None,
        "cpps_db": 18.0 if voiced else None,
    }


def ctc_frames(duration_s: float) -> int:
    """wav2vec2's frame count for a clip at 16 kHz: a 400-sample window every 320 samples."""
    samples = round(duration_s * 16_000)
    return 0 if samples < 400 else (samples - 400) // 320 + 1


def place_tokens(tokens: Sequence[str], words: Sequence[tuple[float, float]], frames: int) -> list[dict[str, Any]]:
    """Token spans over the words' times, as ``merge_tokens`` would give them: in order, at least one frame
    each, with a blank frame between two equal tokens, inside ``frames``.

    Letters share their word's time evenly; ``|`` takes the pause between words. When the text's words and
    the audio's words differ in number, the words' times are shared out by their letters instead. The
    caller has checked ``frames >= tokens + repeats``, so the fallback of one frame per token always fits.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    for i, token in enumerate(tokens):
        if token == "|":
            if current:
                groups.append(current)
                current = []
        else:
            current.append(i)
    if current:
        groups.append(current)
    if len(groups) == len(words):
        group_times = list(words)
    else:
        start, end = words[0][0], words[-1][1]
        letters = sum(len(g) for g in groups) or 1
        group_times = []
        done = 0
        for group in groups:
            a = start + (end - start) * done / letters
            done += len(group)
            group_times.append((a, start + (end - start) * done / letters))
    targets: list[tuple[float, float]] = []
    seen = 0
    for i, token in enumerate(tokens):
        if token != "|":
            gi = next(g for g, group in enumerate(groups) if i in group)
            a, b = group_times[gi]
            k = groups[gi].index(i)
            size = len(groups[gi])
            targets.append((a + (b - a) * k / size, a + (b - a) * (k + 1) / size))
            seen = gi + 1
        else:
            before = (
                group_times[seen - 1][1]
                if seen > 0
                else max(0.0, (group_times[0][0] if group_times else 0.0) - ALIGN_FRAME_S)
            )
            after = group_times[seen][0] if seen < len(group_times) else before + ALIGN_FRAME_S
            targets.append((before, max(after, before)))
    spans: list[dict[str, Any]] = []
    prev_end = 0
    for i, (token, (a, b)) in enumerate(zip(tokens, targets, strict=True)):
        start = max(math.floor(a / ALIGN_FRAME_S), prev_end + (1 if i and token == tokens[i - 1] else 0))
        end = max(round(b / ALIGN_FRAME_S), start + 1)
        spans.append({"token_index": i, "start_frame": start, "end_frame": end, "score": 0.9})
        prev_end = end
    if spans and spans[-1]["end_frame"] > frames:
        spans = []
        pos = 0
        for i, token in enumerate(tokens):
            if i and token == tokens[i - 1]:
                pos += 1
            spans.append({"token_index": i, "start_frame": pos, "end_frame": pos + 1, "score": 0.9})
            pos += 1
    return spans


def _write_png(path: Path, track: Sequence[float | None] | None) -> None:
    """A small grey PNG: the pitch track drawn as columns, or a plain square (standard library only)."""
    width, height = 64, 32
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            value = 255
            if track:
                hz = track[x * len(track) // width]
                if hz is not None and (height - 1 - y) <= (hz - 100) * height / 450:
                    value = 64
            row.append(value)
        rows.append(bytes(row))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"".join(rows), 9))
    png += chunk(b"IEND", b"")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(png)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
