"""The worker contract tests (design Appendix A; plan.md WP16): every worker role runs these.

Subclass ``WorkerContract`` once per worker, in a class whose name starts with ``Test``, and set ``role``::

    from narration_worker.testing.contract import WorkerContract

    class TestQwen3WorkerContract(WorkerContract):
        role = "qwen3"

The default ``worker_argv`` fixture starts ``python -m narration_worker --role <role> --store <tmp store>``
with the running interpreter, so run the tests from the worker's own venv. Override fixtures to change what
is started (``worker_argv``, ``worker_env``) or to enable the tests that need a loaded model:

- ``load_request``: a ``load`` request that succeeds here. The default ``None`` skips the tests that need
  it, as for a real model on a machine without it. For a role with ``synthesize`` or ``design``, it is a
  complete Qwen load: ``model``, ``determinism`` and ``settings`` whose ``generation`` has all ten sampling
  values (``GENERATION`` is one), the ceiling ``settings.generation.max_new_tokens`` among them (the
  ceiling test fails, not skips, without them);
- ``render_requests``: for a role with ``synthesize`` or ``design``, the requests to send after that
  ``load`` so that one of those calls succeeds, that call last (for example a ``prepare_voice`` and then a
  ``synthesize``). The test sets the last call's ``max_new_tokens`` itself. A role with those ops that gives
  a ``load_request`` must give these too: the call-cap tests fail, not skip, without them.

Nothing here needs a model or a GPU unless those fixtures load one.

What the contract says, beyond the message shapes:

- ``hello`` names the role, the protocol version, exactly the role's ops, and a complete fingerprint;
- every reply echoes its request's id; a line without an integer id gets ``"id": null``;
- malformed input and unknown ops get ``INVALID_REQUEST`` and the worker keeps serving;
- model ops reply ``NOT_LOADED`` before ``load`` (and after ``unload``), before reading any file;
- ``load`` with a snapshot directory that does not exist replies ``BACKEND_NOT_INSTALLED``;
- a role that takes ``models`` (the QA models by use) checks each snapshot reference as
  ``narration_worker.snapshots`` does (design section 4): its shape and absolute folder, then every folder's
  presence before anything else, then a 40-hex revision naming its folder, each with its code and
  ``details.field``;
- a rendering role's ``load`` needs its ceiling, ``settings.generation.max_new_tokens`` (design section 10.1,
  DC-4): a load whose ``settings``, ``settings.generation`` or ceiling is missing or malformed, or whose
  ceiling is below 2, is ``INVALID_REQUEST`` with ``details.field`` naming that member, never defaulted;
- ``synthesize`` and ``design`` need their own ``max_new_tokens`` (design section 10.1, DC-4): missing, not
  an integer, below 2 or above the loaded ceiling is ``INVALID_REQUEST`` for ``max_new_tokens``, before
  ``VOICE_NOT_PREPARED``;
- the reply echoes the cap and counts as ``protocol.AudioReply`` says: a cap below the take's length cuts it
  and reports ``hit_token_cap: true`` with ``new_tokens`` = cap - 1; an end token exactly at the cap (a
  cap of ``new_tokens`` + 1) is not a hit and changes nothing; the loaded ceiling does not cut a short take;
- ``shutdown`` replies, then the process exits with code 0; so does the end of input, without a reply;
- stdout carries protocol replies and nothing else.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

from narration_worker.protocol import COMMON_OPS, FAKE_OPS, OPS_BY_ROLE, PROTOCOL_VERSION, Fingerprint, WorkerRole
from narration_worker.qwen_settings import CEILING_FIELD, GENERATION_KEYS

from .client import WorkerProcess, check_reply

CALL_CAP_OPS: tuple[str, ...] = ("synthesize", "design")
QA_MODEL_OPS: tuple[str, ...] = ("transcribe", "embed", "align")
"""The ops of a role whose ``load`` takes ``models`` by use (the QA models)."""
"""The ops that take their own ``max_new_tokens`` (design section 10.1, DC-4)."""
DEFAULT_CEILING = 8192
"""The loaded ceiling when a ``load`` request gives no ``settings.generation.max_new_tokens``: the pinned Qwen
snapshots' value (a real Qwen worker's ``load`` always passes it)."""
SAMPLE_CAP = 128
"""The ``max_new_tokens`` of the sample requests: the daemon's floor, so within any ceiling."""
MISSING: Any = object()
"""A request member left out (as opposed to one sent as null)."""

DETERMINISM: dict[str, Any] = {
    "tf32": False,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "deterministic_algorithms": "warn_only",
}
GENERATION: dict[str, Any] = {
    "do_sample": True,
    "top_k": 50,
    "top_p": 1.0,
    "temperature": 0.9,
    "repetition_penalty": 1.05,
    "subtalker_dosample": True,
    "subtalker_top_k": 50,
    "subtalker_top_p": 1.0,
    "subtalker_temperature": 0.9,
    "max_new_tokens": DEFAULT_CEILING,
}
"""A complete ``settings.generation`` for a test's Qwen ``load``: all ten sampling values
(``qwen_settings.GENERATION_KEYS``), as the pinned snapshots set them (plan.md section 1.3 item 1)."""


def sample_requests(store_root: Path) -> dict[str, dict[str, Any]]:
    """A well-formed request body for each model op, naming files that need not exist."""
    scratch = store_root / "scratch" / "contract"
    wav = str(scratch / "in.wav")
    out = str(scratch / "out.wav")
    voice = "sha256:" + "0" * 64
    return {
        "prepare_voice": {"voice_hash": voice, "ref_wav": wav, "ref_text": "Hello there.", "x_vector_only_mode": False},
        "synthesize": {
            "voice_hash": voice,
            "engine_text": "Hello there.",
            "language": "English",
            "seed": 1,
            "max_new_tokens": SAMPLE_CAP,
            "out_path": out,
        },
        "design": {
            "description": "A calm voice.",
            "design_text": "Hello there.",
            "language": "English",
            "seed": 1,
            "max_new_tokens": SAMPLE_CAP,
            "out_path": out,
        },
        "transcribe": {"wav": wav, "language": "English", "word_timestamps": True, "long_form": True},
        "embed": {"wav": wav, "device": "cpu"},
        "f0": {"wav": wav, "fmin_hz": 50.0, "fmax_hz": 400.0},
        "align": {"wav": wav, "tokens": ["H", "E", "L", "L", "O"]},
        "profile": {"wav": wav, "out_dir": str(scratch / "profile"), "transcript": None},
    }


def loaded_ceiling(load_request: dict[str, Any]) -> int:
    """The ``max_new_tokens`` ceiling a ``load`` request sets (``DEFAULT_CEILING`` when it names none)."""
    settings = load_request.get("settings")
    generation = settings.get("generation", {}) if isinstance(settings, dict) else {}
    ceiling = generation.get("max_new_tokens", DEFAULT_CEILING) if isinstance(generation, dict) else DEFAULT_CEILING
    return int(ceiling)


def missing_snapshot_load(store_root: Path) -> dict[str, Any]:
    """A complete ``load`` request whose every snapshot directory is missing."""
    ref = {"repo": "example/missing", "revision": "0" * 40, "snapshot_dir": str(store_root / "no-such-snapshot")}
    return {
        "device": "cpu",
        "model": ref,
        "models": {"asr": ref, "sv": ref, "aligner": ref},
        "engine_profile_id": "contract-test",
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "determinism": dict(DETERMINISM),
        "settings": {"non_streaming_mode": False, "generation": dict(GENERATION)},
    }


class WorkerContract:
    """The contract tests; subclass per worker (see the module docstring)."""

    role: ClassVar[WorkerRole]
    timeout_s: ClassVar[float] = 120.0
    """How long any single reply may take (a real worker imports torch when it starts)."""
    small_cap: ClassVar[int] = 2
    """The ``max_new_tokens`` the call-cap test expects to cut the last of ``render_requests`` (at least 2)."""

    # ------------------------------------------------------------------ fixtures to override
    @pytest.fixture
    def store_root(self, tmp_path: Path) -> Path:
        root = tmp_path / "store"
        (root / "scratch").mkdir(parents=True)
        return root

    @pytest.fixture
    def worker_argv(self, store_root: Path) -> list[str]:
        return [
            sys.executable,
            "-m",
            "narration_worker",
            "--role",
            self.role,
            "--store",
            str(store_root),
            "--cpu-threads",
            "2",
        ]

    @pytest.fixture
    def worker_env(self) -> dict[str, str] | None:
        return None

    @pytest.fixture
    def load_request(self, store_root: Path) -> dict[str, Any] | None:
        return None

    @pytest.fixture
    def render_requests(self, store_root: Path) -> list[tuple[str, dict[str, Any]]] | None:
        return None

    @pytest.fixture
    def worker(self, worker_argv: list[str], worker_env: dict[str, str] | None) -> Iterator[WorkerProcess]:
        process = WorkerProcess(worker_argv, env=worker_env)
        try:
            process.start()
            yield process
        finally:
            process.stop()

    # ------------------------------------------------------------------ the contract
    def test_hello_names_role_protocol_ops_and_fingerprint_appA(self, worker: WorkerProcess) -> None:
        reply = worker.request("hello", timeout_s=self.timeout_s)
        assert reply["ok"] is True, reply
        assert reply["role"] == self.role
        assert reply["protocol"] == PROTOCOL_VERSION
        ops = reply["capabilities"]["ops"]
        assert sorted(ops) == sorted(OPS_BY_ROLE[self.role]) and len(ops) == len(set(ops))
        controls = reply["capabilities"].get("controls")
        if controls is not None:
            assert set(controls) == {"pace", "context", "instruct"}
            assert all(isinstance(v, bool) for v in controls.values())
        fingerprint = reply["fingerprint"]
        assert set(fingerprint) == set(Fingerprint.__annotations__)
        assert isinstance(fingerprint["python"], str) and fingerprint["python"].startswith("3.")
        assert isinstance(fingerprint["platform"], str) and fingerprint["platform"]
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in fingerprint["packages"].items())
        assert "narration-worker" in fingerprint["packages"]
        for key in ("cuda", "cudnn", "gpu", "driver"):
            assert fingerprint[key] is None or isinstance(fingerprint[key], str), key
        assert fingerprint["cpu_threads"] == 2
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in fingerprint["env"].items())
        assert fingerprint["env"].get("CUBLAS_WORKSPACE_CONFIG"), "CUBLAS_WORKSPACE_CONFIG must be set (s10.1)"

    def test_every_reply_echoes_its_request_id_appA(self, worker: WorkerProcess) -> None:
        for rid in (0, 7, 2**40, -3):
            worker.send({"id": rid, "op": "hello"})
            reply = worker.receive(self.timeout_s)
            assert (reply["id"], reply["ok"]) == (rid, True)

    def test_an_unknown_op_is_invalid_request_appA(self, worker: WorkerProcess) -> None:
        foreign = sorted(set(FAKE_OPS) - set(OPS_BY_ROLE[self.role]))
        for op in ["no_such_op", *foreign]:
            reply = worker.request(op, timeout_s=self.timeout_s)
            assert reply["ok"] is False and reply["error"]["code"] == "INVALID_REQUEST", (op, reply)

    @pytest.mark.parametrize(
        ("line", "expected_id"),
        [
            (b"this is not json", None),
            (b"[1, 2, 3]", None),
            (b"\xff\xfe\xfd", None),
            (b'{"op": "hello"}', None),
            (b'{"id": "1", "op": "hello"}', None),
            (b'{"id": true, "op": "hello"}', None),
            (b'{"id": 1, "op": "hello", "x": NaN}', None),
            (b'{"id": 5}', 5),
            (b'{"id": 6, "op": 7}', 6),
        ],
    )
    def test_malformed_input_is_invalid_request_and_the_worker_keeps_serving_appA(
        self, worker: WorkerProcess, line: bytes, expected_id: int | None
    ) -> None:
        worker.send_raw(line + b"\n")
        reply = worker.receive(self.timeout_s)
        assert reply["id"] == expected_id
        assert reply["ok"] is False and reply["error"]["code"] == "INVALID_REQUEST"
        worker.send_raw(b"\n")  # a blank line gets no reply
        assert worker.request("hello", timeout_s=self.timeout_s)["ok"] is True

    def test_model_ops_before_load_reply_not_loaded_appA(self, worker: WorkerProcess, store_root: Path) -> None:
        samples = sample_requests(store_root)
        for op in OPS_BY_ROLE[self.role]:
            if op in COMMON_OPS:
                continue
            reply = worker.request(op, timeout_s=self.timeout_s, **samples[op])
            assert reply["ok"] is False and reply["error"]["code"] == "NOT_LOADED", (op, reply)

    def test_load_with_a_missing_snapshot_is_backend_not_installed_s14(
        self, worker: WorkerProcess, store_root: Path
    ) -> None:
        reply = worker.request("load", timeout_s=self.timeout_s, **missing_snapshot_load(store_root))
        assert reply["ok"] is False and reply["error"]["code"] == "BACKEND_NOT_INSTALLED", reply

    def test_load_checks_snapshot_references_s4(self, worker: WorkerProcess, store_root: Path) -> None:
        """Every worker checks a ``models`` reference the same way (``narration_worker.snapshots``), in this
        order: its shape and an absolute folder; every folder's presence, before anything else; then a 40-hex
        revision that names its folder. Each refusal has its code and ``details.field``."""
        if not any(op in OPS_BY_ROLE[self.role] for op in QA_MODEL_OPS):
            pytest.skip(f"the {self.role} role's load takes no models")
        folder = store_root / "snapshots" / ("0" * 40)
        folder.mkdir(parents=True)
        ref = {"repo": "example/asr", "revision": "0" * 40, "snapshot_dir": str(folder)}
        cases: list[tuple[object, str, str]] = [
            ([ref], "INVALID_REQUEST", "models"),
            ({"asr": {"repo": "example/asr"}}, "INVALID_REQUEST", "models.asr"),
            ({"asr": ref | {"snapshot_dir": "relative"}}, "INVALID_REQUEST", "models.asr.snapshot_dir"),
            (
                {"asr": ref | {"revision": "x", "snapshot_dir": str(store_root / "none")}},
                "BACKEND_NOT_INSTALLED",
                "models.asr",
            ),
            (
                {"asr": ref | {"revision": "x"}, "sv": ref | {"snapshot_dir": str(store_root / "none")}},
                "BACKEND_NOT_INSTALLED",
                "models.sv",
            ),
            ({"asr": ref | {"revision": "1" * 40}}, "INVALID_REQUEST", "models.asr.snapshot_dir"),
            ({"asr": ref | {"revision": "main"}}, "INVALID_REQUEST", "models.asr.revision"),
        ]
        for models, code, field in cases:
            reply = worker.request("load", timeout_s=self.timeout_s, device="cpu", models=models)
            assert reply["ok"] is False and reply["error"]["code"] == code, (models, reply)
            assert reply["error"].get("details", {}).get("field") == field, (models, reply)

    def test_load_then_unload_appA(
        self, worker: WorkerProcess, store_root: Path, load_request: dict[str, Any] | None
    ) -> None:
        if load_request is None:
            pytest.skip(f"no loadable {self.role} models here: override the load_request fixture to run this")
        loaded = worker.request("load", timeout_s=self.timeout_s, **load_request)
        assert loaded["ok"] is True, loaded
        assert isinstance(loaded["load_s"], int | float) and loaded["load_s"] >= 0
        assert loaded["vram_mb"] is None or isinstance(loaded["vram_mb"], int)
        assert worker.request("unload", timeout_s=self.timeout_s) == {"id": loaded["id"] + 1, "ok": True}
        assert worker.request("unload", timeout_s=self.timeout_s)["ok"] is True
        model_ops = [op for op in OPS_BY_ROLE[self.role] if op not in COMMON_OPS]
        reply = worker.request(model_ops[0], timeout_s=self.timeout_s, **sample_requests(store_root)[model_ops[0]])
        assert reply["ok"] is False and reply["error"]["code"] == "NOT_LOADED"

    def test_a_load_without_a_valid_ceiling_is_invalid_request_s10_1(
        self, worker: WorkerProcess, load_request: dict[str, Any] | None
    ) -> None:
        """``load``'s ``settings.generation.max_new_tokens`` is the ceiling of every call's cap (DC-4). It is an
        audio-changing setting, so it is always passed, never defaulted (section 10.1): a load that leaves it
        out, or sets it below 2 (qwen-tts's ``min_new_tokens``), is refused, naming the member."""
        if not any(op in OPS_BY_ROLE[self.role] for op in CALL_CAP_OPS):
            pytest.skip(f"the {self.role} role has no synthesize or design")
        if load_request is None:
            pytest.skip(f"no loadable {self.role} models here: override the load_request fixture to run this")
        settings = load_request.get("settings")
        generation = settings.get("generation") if isinstance(settings, dict) else None
        if (
            "model" not in load_request
            or "determinism" not in load_request
            or not isinstance(generation, dict)
            or not set(GENERATION_KEYS) <= set(generation)
        ):
            pytest.fail(
                f"the {self.role} role renders, so its load_request must be a complete Qwen load: model, "
                "determinism, and settings whose generation has all ten sampling values (GENERATION_KEYS), "
                "the ceiling settings.generation.max_new_tokens among them"
            )
        assert isinstance(settings, dict)
        ceiling = CEILING_FIELD
        cases: list[tuple[str, object]] = [
            ("settings", MISSING),
            ("settings", []),
            ("settings.generation", {k: v for k, v in settings.items() if k != "generation"}),
            ("settings.generation", {**settings, "generation": "every value"}),
            (ceiling, {**settings, "generation": {k: v for k, v in generation.items() if k != "max_new_tokens"}}),
        ]
        for bad in (1, 0, -1, None, 1.5, True, "8192"):
            cases.append((ceiling, {**settings, "generation": {**generation, "max_new_tokens": bad}}))
        for field, value in cases:
            body = {k: v for k, v in load_request.items() if k != "settings"}
            if value is not MISSING:
                body["settings"] = value
            reply = worker.request("load", timeout_s=self.timeout_s, **body)
            assert reply["ok"] is False and reply["error"]["code"] == "INVALID_REQUEST", (field, value, reply)
            assert reply["error"].get("details", {}).get("field") == field, (field, value, reply)

    def test_a_call_without_a_valid_max_new_tokens_is_invalid_request_s10_1(
        self, worker: WorkerProcess, store_root: Path, load_request: dict[str, Any] | None
    ) -> None:
        ops = [op for op in CALL_CAP_OPS if op in OPS_BY_ROLE[self.role]]
        if not ops:
            pytest.skip(f"the {self.role} role has no synthesize or design")
        if load_request is None:
            pytest.skip(f"no loadable {self.role} models here: override the load_request fixture to run this")
        assert worker.request("load", timeout_s=self.timeout_s, **load_request)["ok"] is True
        ceiling = loaded_ceiling(load_request)
        samples = sample_requests(store_root)
        missing = object()
        for op in ops:
            body = {k: v for k, v in samples[op].items() if k != "max_new_tokens"}
            for bad in (missing, None, 1, 0, -1, ceiling + 1, 1.5, True, "128"):
                payload = body if bad is missing else {**body, "max_new_tokens": bad}
                reply = worker.request(op, timeout_s=self.timeout_s, **payload)
                assert reply["ok"] is False and reply["error"]["code"] == "INVALID_REQUEST", (op, bad, reply)
                assert reply["error"].get("details", {}).get("field") == "max_new_tokens", (op, bad, reply)

    def render_at_ceiling(
        self,
        worker: WorkerProcess,
        load_request: dict[str, Any] | None,
        render_requests: list[tuple[str, dict[str, Any]]] | None,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Load, send the set-up requests, and render the last call at the ceiling: (op, body, reply).

        Skips when the role has neither ``synthesize`` nor ``design`` or no ``load_request`` is given; fails
        when those are there but ``render_requests`` is not, so a real worker cannot skip the cap tests.
        """
        if not any(op in OPS_BY_ROLE[self.role] for op in CALL_CAP_OPS):
            pytest.skip(f"the {self.role} role has no synthesize or design")
        if load_request is None:
            pytest.skip(f"no loadable {self.role} models here: override the load_request fixture to run this")
        if render_requests is None:
            pytest.fail(
                f"override render_requests: the {self.role} role renders, and its call cap tests (DC-4) need "
                "the requests that make one synthesize or design call succeed after load_request"
            )
        *setup, (op, body) = render_requests
        assert op in CALL_CAP_OPS, f"the last of render_requests must be one of {CALL_CAP_OPS}, not {op}"
        assert worker.request("load", timeout_s=self.timeout_s, **load_request)["ok"] is True
        for setup_op, setup_body in setup:
            reply = worker.request(setup_op, timeout_s=self.timeout_s, **setup_body)
            assert reply["ok"] is True, (setup_op, reply)
        ceiling = loaded_ceiling(load_request)
        full = worker.request(op, timeout_s=self.timeout_s, **{**body, "max_new_tokens": ceiling})
        assert full["ok"] is True and full["hit_token_cap"] is False, full
        assert full["max_new_tokens"] == ceiling, full
        return op, body, full

    def test_a_call_cap_below_the_take_cuts_it_and_reports_hit_token_cap_s10_1(
        self,
        worker: WorkerProcess,
        load_request: dict[str, Any] | None,
        render_requests: list[tuple[str, dict[str, Any]]] | None,
    ) -> None:
        op, body, full = self.render_at_ceiling(worker, load_request, render_requests)
        cut = worker.request(op, timeout_s=self.timeout_s, **{**body, "max_new_tokens": self.small_cap})
        assert cut["ok"] is True and cut["hit_token_cap"] is True, cut
        assert cut["max_new_tokens"] == self.small_cap, cut
        assert cut.get("new_tokens", self.small_cap - 1) == self.small_cap - 1, cut
        assert cut["samples"] < full["samples"]

    def test_an_end_token_exactly_at_the_cap_is_not_a_hit_s10_1(
        self,
        worker: WorkerProcess,
        load_request: dict[str, Any] | None,
        render_requests: list[tuple[str, dict[str, Any]]] | None,
    ) -> None:
        """A render of F frames takes F + 1 steps, the last one the end token: a cap of F + 1 is not a hit and
        changes nothing; a cap of F is a hit with F - 1 frames (``protocol.AudioReply``)."""
        op, body, full = self.render_at_ceiling(worker, load_request, render_requests)
        assert "new_tokens" in full, "the worker must report new_tokens (the decoded frames) for this contract"
        frames = full["new_tokens"]
        assert frames >= 2, f"render_requests' last call must render at least 2 frames, not {frames}"
        exact = worker.request(op, timeout_s=self.timeout_s, **{**body, "max_new_tokens": frames + 1})
        assert exact["ok"] is True and exact["hit_token_cap"] is False, exact
        assert (exact["new_tokens"], exact["samples"]) == (frames, full["samples"]), exact
        under = worker.request(op, timeout_s=self.timeout_s, **{**body, "max_new_tokens": frames})
        assert under["ok"] is True and under["hit_token_cap"] is True, under
        assert under["new_tokens"] == frames - 1, under

    def test_shutdown_replies_then_exits_zero_appA(self, worker: WorkerProcess) -> None:
        reply = worker.request("shutdown", timeout_s=self.timeout_s)
        assert reply["ok"] is True
        assert worker.wait(self.timeout_s) == 0

    def test_end_of_input_exits_zero_appA(self, worker: WorkerProcess) -> None:
        assert worker.request("hello", timeout_s=self.timeout_s)["ok"] is True
        worker.close_stdin()
        assert worker.wait(self.timeout_s) == 0
        assert len(worker.lines) == 1

    def test_stdout_is_protocol_only_appA(self, worker: WorkerProcess, store_root: Path) -> None:
        worker.request("hello", timeout_s=self.timeout_s)
        worker.request("no_such_op", timeout_s=self.timeout_s)
        worker.send_raw(b"not json\n")
        worker.receive(self.timeout_s)
        worker.request("load", timeout_s=self.timeout_s, **missing_snapshot_load(store_root))
        worker.request("unload", timeout_s=self.timeout_s)
        worker.request("shutdown", timeout_s=self.timeout_s)
        assert worker.wait(self.timeout_s) == 0
        assert len(worker.lines) == 6, worker.lines
        for line in worker.lines:
            check_reply(line)
