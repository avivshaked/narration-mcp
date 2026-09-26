"""Delivery is deterministic on one machine with the same pins (design sections 10.1 and 13), and its work
runs on the calling thread alone (section 4.1)."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from narration.config import DeliveryConfig
from narration.post import DeliveryPipeline, deliver

from .signals import speech_like, write_raw

_PROCESS = (
    "import sys\n"
    "from pathlib import Path\n"
    "from narration.config import DeliveryConfig\n"
    "from narration.post import DeliveryPipeline\n"
    "DeliveryPipeline().process(Path(sys.argv[1]), Path(sys.argv[2]), DeliveryConfig())\n"
)

# numpy's import starts OpenBLAS's threads, so counting threads proves nothing: this measures the CPU time
# each thread spends while `deliver` runs, and prints the calling thread's and the sum of all others'.
_THREAD_TIMES = (
    "import threading\n"
    "import numpy as np, psutil\n"
    "from narration.config import DeliveryConfig\n"
    "from narration.post import deliver\n"
    "t = np.arange(24000 * 60) / 24000\n"
    "x = 0.1 * np.sin(2 * np.pi * 150 * t) * (0.3 + np.sin(2 * np.pi * 2 * t) ** 2)\n"
    "deliver(x[:48000], 24000, DeliveryConfig())\n"
    "proc, main = psutil.Process(), threading.get_native_id()\n"
    "def cpu():\n"
    "    return {th.id: th.user_time + th.system_time for th in proc.threads()}\n"
    "before = cpu()\n"
    "deliver(x, 24000, DeliveryConfig())\n"
    "after = cpu()\n"
    "others = sum(v - before.get(k, 0.0) for k, v in after.items() if k != main)\n"
    "print(after[main] - before.get(main, 0.0), others)\n"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_delivery_gives_the_same_bytes_twice_s13(tmp_path: Path) -> None:
    raw = tmp_path / "raw.wav"
    write_raw(str(raw), speech_like(24000, bursts=(1.4, 2.1), gaps=(0.5,)), 24000)
    first, second = tmp_path / "a.wav", tmp_path / "b.wav"
    out_a = DeliveryPipeline().process(raw, first, DeliveryConfig())
    out_b = DeliveryPipeline().process(raw, second, DeliveryConfig())
    assert first.read_bytes() == second.read_bytes()
    assert out_a == out_b


def test_delivery_in_memory_gives_the_same_bytes_as_the_file_s13(tmp_path: Path) -> None:
    x = speech_like(24000)
    raw, out = tmp_path / "raw.wav", tmp_path / "out.wav"
    write_raw(str(raw), x, 24000)
    DeliveryPipeline().process(raw, out, DeliveryConfig())
    # The raw file is float32, so the in-memory input is rounded the same way first.
    assert deliver(x.astype("float32").astype("float64"), 24000, DeliveryConfig()).wav == out.read_bytes()


def test_delivery_gives_the_same_bytes_in_another_process_s10_1(tmp_path: Path) -> None:
    raw = tmp_path / "raw.wav"
    write_raw(str(raw), speech_like(24000, level=0.2, bursts=(1.0, 1.0), gaps=(0.3,)), 24000)
    here, there = tmp_path / "here.wav", tmp_path / "there.wav"
    DeliveryPipeline().process(raw, here, DeliveryConfig())
    subprocess.run([sys.executable, "-c", _PROCESS, str(raw), str(there)], check=True, timeout=120)
    assert _sha256(here) == _sha256(there)


def test_delivery_work_stays_on_the_calling_thread_s4_1() -> None:
    done = subprocess.run(
        [sys.executable, "-c", _THREAD_TIMES],
        check=True,
        timeout=120,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    main, others = (float(v) for v in done.stdout.split())
    assert main > 0.2  # a minute of audio is real work for the calling thread
    # Other threads may be credited a scheduler tick (about 16 ms on Windows), never the work itself.
    assert others <= max(0.05 * main, 0.035)
