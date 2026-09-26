"""Delivery is deterministic (design section 13) and stays on one CPU thread (section 4.1)."""

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

_THREADS = (
    "import threading, time\n"
    "import numpy as np, psutil\n"
    "from narration.config import DeliveryConfig\n"
    "from narration.post import deliver\n"
    "t = np.arange(24000 * 20) / 24000\n"
    "x = 0.1 * np.sin(2 * np.pi * 150 * t) * (0.3 + np.sin(2 * np.pi * 2 * t) ** 2)\n"
    "proc, counts, stop = psutil.Process(), [], threading.Event()\n"
    "def sample():\n"
    "    while not stop.is_set():\n"
    "        counts.append(proc.num_threads())\n"
    "        time.sleep(0.001)\n"
    "sampler = threading.Thread(target=sample)\n"
    "sampler.start()\n"
    "time.sleep(0.1)\n"
    "before = max(counts)\n"
    "deliver(x, 24000, DeliveryConfig())\n"
    "stop.set()\n"
    "sampler.join()\n"
    "print(before, max(counts))\n"
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


def test_delivery_gives_the_same_bytes_across_processes_s13(tmp_path: Path) -> None:
    raw = tmp_path / "raw.wav"
    write_raw(str(raw), speech_like(24000, level=0.2, bursts=(1.0, 1.0), gaps=(0.3,)), 24000)
    here, there = tmp_path / "here.wav", tmp_path / "there.wav"
    DeliveryPipeline().process(raw, here, DeliveryConfig())
    subprocess.run([sys.executable, "-c", _PROCESS, str(raw), str(there)], check=True, timeout=120)
    assert _sha256(here) == _sha256(there)


def test_delivery_runs_on_one_thread_s4_1() -> None:
    done = subprocess.run(
        [sys.executable, "-c", _THREADS], check=True, timeout=120, capture_output=True, text=True, encoding="utf-8"
    )
    before, during = (int(v) for v in done.stdout.split())
    assert during == before
