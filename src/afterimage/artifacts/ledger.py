"""Device-hour accounting and a lightweight GPU sampler.

The ledger is an append-only CSV on shared storage. A single short
``write`` on a file opened with O_APPEND is atomic on POSIX for lines this
small, so concurrent workers can append without a lock.
"""

from __future__ import annotations

import csv
import io
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

LEDGER_FIELDS = (
    "job_id", "phase", "block", "device", "device_name",
    "start_unix", "end_unix", "device_seconds", "status",
)


@dataclass
class LedgerRow:
    job_id: str
    phase: str
    block: int
    device: str
    device_name: str
    start_unix: float
    end_unix: float
    device_seconds: float
    status: str


def append_row(path: Path, row: LedgerRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=LEDGER_FIELDS)
    if new:
        w.writeheader()
    w.writerow(asdict(row))
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, buf.getvalue().encode())
    finally:
        os.close(fd)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("job_id") != "job_id"]


def totals(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in read_rows(path):
        if r["status"] != "ok":
            key = f"{r['phase']}:{r['status']}"
        else:
            key = r["phase"]
        out[key] = out.get(key, 0.0) + float(r["device_seconds"]) / 3600.0
    return out


class GpuSampler:
    """Samples GPU utilization, memory and power every ``interval`` seconds.

    It is a no-op when NVML is unavailable (CPU runs).
    """

    def __init__(self, out: Path, interval: float = 10.0, device_index: int = 0):
        self.out, self.interval, self.idx = out, interval, device_index
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GpuSampler":
        try:
            import pynvml

            pynvml.nvmlInit()
            self._h = pynvml.nvmlDeviceGetHandleByIndex(self.idx)
            self._nvml = pynvml
        except Exception:
            return self
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        nv = self._nvml
        with open(self.out, "a", newline="") as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(["unix", "util_gpu", "mem_used_mb", "power_w"])
            while not self._stop.is_set():
                try:
                    u = nv.nvmlDeviceGetUtilizationRates(self._h)
                    m = nv.nvmlDeviceGetMemoryInfo(self._h)
                    p = nv.nvmlDeviceGetPowerUsage(self._h) / 1000.0
                    w.writerow([f"{time.time():.1f}", u.gpu, m.used // 2**20, f"{p:.1f}"])
                    f.flush()
                except Exception:
                    pass
                self._stop.wait(self.interval)

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1)
