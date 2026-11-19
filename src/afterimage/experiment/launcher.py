"""Claim-and-run scheduling of independent blocks (docs/blueprint.md §11.3).

Each worker walks block indices in order. It claims a block with a
non-blocking ``flock`` on ``locks/block_XXX.lock``, runs it, appends a
ledger row and releases the lock. Workers stop once ``n_blocks`` blocks have
passed the gate (gate-excluded blocks are replaced by later indices), once
the index cap is reached, or once the device-hour budget would be exceeded.

No Ray, no scheduler daemon: blocks are coarse, independent and resumable.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import time
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path

from ..artifacts import ledger
from ..config import Config
from .layout import STATUS_COMPLETE, STATUS_EXCLUDED, STATUS_FAILED, block_status, study_dir


def n_passed(cfg: Config) -> int:
    from .layout import list_blocks

    return sum(block_status(cfg, b) in (STATUS_COMPLETE, STATUS_FAILED) for b in list_blocks(cfg))


def max_block_index(cfg: Config) -> int:
    return 2 * cfg.design.n_blocks + 8


def _budget_ok(cfg: Config, budget_hours: float) -> bool:
    if budget_hours <= 0:
        return True
    rows = [r for r in ledger.read_rows(study_dir(cfg) / "ledger.csv") if r["phase"] == cfg.meta.name]
    spent = sum(float(r["device_seconds"]) for r in rows) / 3600.0
    done = [float(r["device_seconds"]) for r in rows if r["status"] == "ok"]
    est = (sum(done) / len(done) / 3600.0) if done else 0.0
    return spent + est <= budget_hours


def worker_loop(cfg: Config, run_one: Callable[[int], dict], *, device: str, blocks: list[int] | None = None,
                budget_hours: float = 0.0, log: Callable[[str], None] = print) -> list[int]:
    """Run blocks until the study is full. Returns the blocks this worker ran."""
    sdir = study_dir(cfg)
    (sdir / "locks").mkdir(parents=True, exist_ok=True)
    explicit = blocks is not None
    candidates = blocks if explicit else list(range(max_block_index(cfg)))
    ran: list[int] = []
    for b in candidates:
        if not explicit and n_passed(cfg) >= cfg.design.n_blocks:
            break
        if block_status(cfg, b) in (STATUS_COMPLETE, STATUS_EXCLUDED, STATUS_FAILED):
            continue
        if not _budget_ok(cfg, budget_hours):
            log("device-hour budget reached; not claiming more blocks")
            break
        fd = os.open(sdir / "locks" / f"block_{b:03d}.lock", os.O_CREAT | os.O_RDWR, 0o644)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue  # another worker owns it
            if block_status(cfg, b) in (STATUS_COMPLETE, STATUS_EXCLUDED, STATUS_FAILED):
                continue
            start, status = time.time(), "ok"
            try:
                info = run_one(b)
                status = "ok" if info.get("status") in (STATUS_COMPLETE, STATUS_EXCLUDED) else info.get("status", "?")
            except Exception:
                status = "error"
                log(traceback.format_exc())
            end = time.time()
            ledger.append_row(sdir / "ledger.csv", ledger.LedgerRow(
                job_id=uuid.uuid4().hex[:12], phase=cfg.meta.name, block=b, device=device,
                device_name=_device_name(device), start_unix=start, end_unix=end,
                device_seconds=end - start, status=status))
            ran.append(b)
            if status == "error":
                raise RuntimeError(f"block {b} raised; see log above (resumable: rerun the same command)")
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
    return ran


def _device_name(device: str) -> str:
    if device.startswith("cuda"):
        try:
            import torch

            return torch.cuda.get_device_name(torch.device(device).index or 0)
        except Exception:
            return "cuda"
    return "cpu"


def launch(config_path: str, *, workers_per_device: int, devices: list[str], budget_hours: float) -> int:
    """Spawn one worker subprocess per (device, slot) and wait for all of them."""
    procs = []
    for dev in devices:
        for slot in range(workers_per_device):
            env = dict(os.environ)
            cmd = [sys.executable, "-m", "afterimage.cli", "run", "--config", config_path,
                   "--budget-gpu-hours", str(budget_hours)]
            if dev.startswith("cuda"):
                idx = torch_index(dev)
                env["CUDA_VISIBLE_DEVICES"] = str(idx)
                cmd += ["--device", "cuda:0"]
            else:
                cmd += ["--device", "cpu"]
            log = Path(study_dir_from(config_path)) / "logs" / f"worker_{dev.replace(':', '')}_{slot}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            fh = open(log, "a")
            procs.append((subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT), fh))
            time.sleep(1.0)  # stagger so workers do not race for the first lock
    rc = 0
    for p, fh in procs:
        rc |= p.wait()
        fh.close()
    return rc


def torch_index(dev: str) -> int:
    return int(dev.split(":", 1)[1]) if ":" in dev else 0


def study_dir_from(config_path: str) -> Path:
    from ..config import load_config

    return study_dir(load_config(config_path))
