"""The daemon's worker client (design Appendix A, sections 4 and 4.1; plan.md WP16 acceptance).

Every client comes from ``make_client``, which closes it at teardown (killing its process if need be), and
every request has a timeout.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from narration.config import Config, WorkerProject, WorkersConfig
from narration.contracts.errors import WorkerCrashed, WorkerFailure, WorkerTimeout
from narration.contracts.interfaces import WorkerClient
from narration.contracts.worker import FAKE_OPS, PROTOCOL_VERSION
from narration.workers import SubprocessWorkerClient, WorkerCommand, venv_python, worker_command, worker_env

from .conftest import ClientFactory

TIMEOUT = 30.0
VOICE = "sha256:" + "cd" * 32


def _voice(client: SubprocessWorkerClient, store: Path) -> None:
    clip = store / "scratch" / "clip.wav"
    client.request(
        "design",
        {
            "description": "A calm voice.",
            "design_text": "Some of them thrive.",
            "language": "English",
            "seed": 3,
            "out_path": str(clip),
        },
        timeout_s=TIMEOUT,
    )
    client.request(
        "prepare_voice",
        {"voice_hash": VOICE, "ref_wav": str(clip), "ref_text": "Some of them thrive.", "x_vector_only_mode": False},
        timeout_s=TIMEOUT,
    )


def _synthesize(
    client: SubprocessWorkerClient, store: Path, name: str, text: str = "Small things live."
) -> dict[str, Any]:
    return client.request(
        "synthesize",
        {
            "voice_hash": VOICE,
            "engine_text": text,
            "language": "English",
            "seed": 11,
            "out_path": str(store / "scratch" / f"{name}.wav"),
        },
        timeout_s=TIMEOUT,
    )


def _ready(client: SubprocessWorkerClient, store: Path) -> None:
    client.start()
    client.request("load", {"device": "cpu"}, timeout_s=TIMEOUT)
    _voice(client, store)


# ---------------------------------------------------------------------- round trips


def test_the_client_is_a_worker_client_appA(make_client: ClientFactory) -> None:
    assert isinstance(make_client(), WorkerClient)


def test_round_trip_through_the_fake_appA(make_client: ClientFactory, store: Path) -> None:
    client = make_client()
    hello = client.start()
    assert hello["role"] == "fake" and hello["protocol"] == PROTOCOL_VERSION
    assert sorted(hello["capabilities"]["ops"]) == sorted(FAKE_OPS)
    assert client.pid is not None and client.is_alive()
    loaded = client.request("load", {"device": "cpu"}, timeout_s=TIMEOUT)
    assert loaded["ok"] is True and loaded["vram_mb"] is None
    _voice(client, store)
    audio = _synthesize(client, store, "take")
    assert audio["sample_rate"] == 24_000 and audio["samples"] > 0 and audio["hit_token_cap"] is False
    heard = client.request(
        "transcribe",
        {"wav": str(store / "scratch" / "take.wav"), "language": "English", "word_timestamps": True, "long_form": True},
        timeout_s=TIMEOUT,
    )
    assert heard["text"] == "Small things live."
    client.close()
    assert client.exit_code == 0 and not client.is_alive()


def test_requests_from_many_threads_get_their_own_replies_appA(make_client: ClientFactory, store: Path) -> None:
    client = make_client()
    _ready(client, store)
    texts = [f"Word number {n} is here." for n in range(6)]
    results: dict[int, str] = {}
    errors: list[BaseException] = []

    def work(n: int) -> None:
        try:
            _synthesize(client, store, f"t{n}", texts[n])
            reply = client.request(
                "transcribe",
                {
                    "wav": str(store / "scratch" / f"t{n}.wav"),
                    "language": "English",
                    "word_timestamps": False,
                    "long_form": True,
                },
                timeout_s=TIMEOUT,
            )
            results[n] = reply["text"]
        except BaseException as exc:  # reported below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=TIMEOUT)
    assert not errors
    assert results == dict(enumerate(texts))


def test_ok_false_raises_worker_failure_and_the_worker_serves_on_appA(make_client: ClientFactory, store: Path) -> None:
    client = make_client({"faults": [{"kind": "gpu_oom", "times": 1}]})
    _ready(client, store)
    with pytest.raises(WorkerFailure) as caught:
        _synthesize(client, store, "first")
    assert caught.value.code == "GPU_OOM"
    assert caught.value.details == {"planted": True}
    assert client.is_alive()
    assert _synthesize(client, store, "retry")["ok"] is True


def test_a_payload_may_not_set_id_or_op_appA(make_client: ClientFactory) -> None:
    client = make_client()
    client.start()
    with pytest.raises(ValueError):
        client.request("hello", {"id": 5}, timeout_s=TIMEOUT)


# ---------------------------------------------------------------------- crashes, timeouts, protocol breaks


def test_a_crash_is_reported_at_once_with_exit_code_and_stderr_tail_appA(
    make_client: ClientFactory, store: Path
) -> None:
    client = make_client({"faults": [{"kind": "crash", "exit_code": 3}]})
    _ready(client, store)
    started = time.monotonic()
    with pytest.raises(WorkerCrashed) as caught:
        client.request(
            "synthesize",
            {
                "voice_hash": VOICE,
                "engine_text": "Hello.",
                "language": "English",
                "seed": 1,
                "out_path": str(store / "scratch" / "x.wav"),
            },
            timeout_s=120.0,
        )
    assert time.monotonic() - started < 10.0, "a crash must be reported, not waited out"
    assert caught.value.exit_code == 3
    assert "planted crash" in caught.value.stderr_tail
    assert not client.is_alive()
    with pytest.raises(WorkerCrashed):
        client.request("hello", {}, timeout_s=TIMEOUT)


def test_a_timeout_stops_the_worker_appA(make_client: ClientFactory, store: Path) -> None:
    client = make_client({"faults": [{"kind": "hang", "op": "unload"}]})
    client.start()
    started = time.monotonic()
    with pytest.raises(WorkerTimeout):
        client.request("unload", {}, timeout_s=1.0)
    assert 0.9 < time.monotonic() - started < 10.0
    assert not client.is_alive()
    assert client.exit_code is not None, "the hung worker must have been killed and reaped"
    with pytest.raises(WorkerCrashed):
        client.request("hello", {}, timeout_s=TIMEOUT)


def test_a_line_that_is_not_protocol_is_a_crash_appA(make_client: ClientFactory) -> None:
    client = make_client({"faults": [{"kind": "protocol_break", "op": "unload"}]})
    client.start()
    with pytest.raises(WorkerCrashed) as caught:
        client.request("unload", {}, timeout_s=TIMEOUT)
    assert "not a protocol message" in str(caught.value)
    assert not client.is_alive()


def test_noise_on_stdout_does_not_reach_the_client_appA(make_client: ClientFactory, store: Path) -> None:
    client = make_client({"faults": [{"kind": "stdout_noise"}]})
    _ready(client, store)
    assert _synthesize(client, store, "noisy")["ok"] is True
    assert client.is_alive()
    assert "planted noise from a child process" in client.stderr_tail()


def test_the_stderr_tail_is_bounded_appA(make_client: ClientFactory, store: Path) -> None:
    client = make_client({"faults": [{"kind": "stdout_noise"}]}, stderr_tail_bytes=200)
    _ready(client, store)
    _synthesize(client, store, "noisy")
    assert 0 < len(client.stderr_tail().encode()) <= 200


def test_a_worker_that_fails_to_start_is_reported_with_its_stderr_appA(make_client: ClientFactory) -> None:
    client = make_client({"faults": [{"kind": "no_such_kind"}]})
    with pytest.raises(WorkerCrashed) as caught:
        client.start()
    assert caught.value.exit_code == 2
    assert "no_such_kind" in caught.value.stderr_tail or "kind" in caught.value.stderr_tail
    assert not client.is_alive()


def test_a_crash_during_hello_fails_start_appA(make_client: ClientFactory) -> None:
    client = make_client({"faults": [{"kind": "crash", "op": "hello"}]})
    with pytest.raises(WorkerCrashed):
        client.start()
    assert client.exit_code == 3


def test_close_sends_shutdown_and_the_worker_exits_zero_appA(make_client: ClientFactory) -> None:
    client = make_client()
    client.start()
    client.close(timeout_s=TIMEOUT)
    assert client.exit_code == 0
    client.close()  # a second close is harmless
    with pytest.raises(WorkerCrashed, match="closed"):
        client.request("hello", {}, timeout_s=TIMEOUT)


def test_close_kills_a_worker_that_hangs_on_shutdown_appA(make_client: ClientFactory) -> None:
    client = make_client({"faults": [{"kind": "hang", "op": "shutdown"}]})
    client.start()
    started = time.monotonic()
    client.close(timeout_s=1.0)
    assert time.monotonic() - started < 15.0
    assert client.exit_code not in (None, 0)


def test_a_role_or_protocol_mismatch_is_backend_not_installed_appA(config: Config) -> None:
    command = worker_command(config, "fake")
    liar = WorkerCommand(role="qa", argv=command.argv, env=command.env)
    client = SubprocessWorkerClient(liar)
    try:
        with pytest.raises(WorkerFailure) as caught:
            client.start()
        assert caught.value.code == "BACKEND_NOT_INSTALLED"
        assert not client.is_alive()
    finally:
        client.close(timeout_s=5.0)


def test_a_missing_worker_python_is_backend_not_installed_s14(config: Config, tmp_path: Path) -> None:
    missing = tmp_path / "no-venv" / "python.exe"
    with pytest.raises(WorkerFailure) as caught:
        worker_command(config, "fake", python=missing)
    assert caught.value.code == "BACKEND_NOT_INSTALLED"
    client = SubprocessWorkerClient(WorkerCommand(role="fake", argv=(str(missing), "-m", "narration_worker"), env={}))
    with pytest.raises(WorkerFailure) as started:
        client.start()
    assert started.value.code == "BACKEND_NOT_INSTALLED"


# ---------------------------------------------------------------------- the daemon's hooks


def test_on_spawn_gets_the_pid_before_hello_s4_1(make_client: ClientFactory) -> None:
    seen: list[int] = []
    client = make_client(on_spawn=seen.append)
    client.start()
    assert seen == [client.pid]


def test_a_failing_spawn_hook_stops_the_worker_s4_1(make_client: ClientFactory) -> None:
    def refuse(pid: int) -> None:
        raise PermissionError(f"cannot add {pid} to the group")

    client = make_client(on_spawn=refuse)
    with pytest.raises(PermissionError):
        client.start()
    assert not client.is_alive()
    assert client.exit_code is not None


# ---------------------------------------------------------------------- command and environment


def test_the_command_is_the_process_identity_marker_s4_1(config: Config, store: Path) -> None:
    command = worker_command(config, "fake")
    assert command.argv[0] == sys.executable
    assert command.argv[1:7] == ("-m", "narration_worker", "--role", "fake", "--store", str(store.resolve()))
    assert command.argv[7:] == ("--cpu-threads", "8")


def test_the_environment_is_offline_deterministic_and_thread_capped_s4_1() -> None:
    workers = WorkersConfig(cpu_threads=3, env={"HF_HUB_OFFLINE": "0", "EXTRA": "yes"})
    base = {"PATH": "p", "PYTHONPATH": "leak", "VIRTUAL_ENV": "server-venv", "PYTHONHOME": "h"}
    env = worker_env(workers, base=base)
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert env["EXTRA"] == "yes" and env["PATH"] == "p"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert env[name] == "3"
    assert not {"PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME"} & set(env)
    kept = worker_env(WorkersConfig(env={"CUBLAS_WORKSPACE_CONFIG": ":16:8"}), base={})
    assert kept["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_the_worker_sees_the_cap_and_the_pins_in_its_fingerprint_s10_1(store: Path) -> None:
    config = Config.for_tests(store)
    config = Config(server=config.server, workers=WorkersConfig(cpu_threads=3))
    client = SubprocessWorkerClient(worker_command(config, "fake"))
    try:
        fingerprint = client.start()["fingerprint"]
    finally:
        client.close(timeout_s=5.0)
    assert fingerprint["cpu_threads"] == 3
    assert fingerprint["env"]["OMP_NUM_THREADS"] == "3"
    assert fingerprint["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert fingerprint["env"]["HF_HUB_OFFLINE"] == "1"


def test_real_roles_run_their_own_venv_python_s4(store: Path, tmp_path: Path) -> None:
    project = tmp_path / "qa-project"
    python = venv_python(project)
    config = Config(server=Config.for_tests(store).server, workers=WorkersConfig(qa=WorkerProject(project=project)))
    with pytest.raises(WorkerFailure) as caught:
        worker_command(config, "qa")
    assert caught.value.code == "BACKEND_NOT_INSTALLED"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    assert worker_command(config, "qa").argv[0] == str(python)
    with pytest.raises(WorkerFailure):
        worker_command(Config.for_tests(store), "qwen3")  # no project and no service root
