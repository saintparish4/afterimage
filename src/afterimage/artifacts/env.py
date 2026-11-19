"""Capture the software, hardware and source state of a run.

Every checkpoint carries this metadata so an auditor can tell exactly what
produced it. Host names are never recorded, only a salted hash, so runs can
be grouped by machine without identifying it.
"""

from __future__ import annotations

import hashlib
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

_HOST_SALT = b"afterimage-host-v1"


def repo_root(start: Path | None = None) -> Path | None:
    p = (start or Path(__file__)).resolve()
    for cand in [p, *p.parents]:
        if (cand / ".git").exists():
            return cand
    return None


def git_info(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    if root is None:
        return {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def git_show(commit: str, relpath: str, root: Path | None = None) -> bytes:
    root = root or repo_root()
    return subprocess.run(
        ["git", "show", f"{commit}:{relpath}"], cwd=root, capture_output=True, check=True
    ).stdout


def git_commit_exists(commit: str, root: Path | None = None) -> bool:
    root = root or repo_root()
    r = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=root, capture_output=True
    )
    return r.returncode == 0


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def host_hash() -> str:
    return hashlib.sha256(_HOST_SALT + socket.gethostname().encode()).hexdigest()[:16]


def _nvml_info() -> dict[str, Any]:
    try:
        import pynvml  # provided by nvidia-ml-py
    except ImportError:
        return {}
    try:
        pynvml.nvmlInit()
        drv = pynvml.nvmlSystemGetDriverVersion()
        return {"nvidia_driver": drv.decode() if isinstance(drv, bytes) else drv}
    except Exception:  # pragma: no cover - depends on hardware
        return {}


def capture_env(device: str | None = None) -> dict[str, Any]:
    import numpy
    import pyarrow
    import safetensors
    import scipy
    import statsmodels

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "host_hash": host_hash(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "pyarrow": pyarrow.__version__,
        "safetensors": safetensors.__version__,
        "git": git_info(),
        "env": {k: os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES")},
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        info["torch_threads"] = torch.get_num_threads()
        info["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
        info["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
        info["tf32_cudnn"] = torch.backends.cudnn.allow_tf32
        if device and device.startswith("cuda") and torch.cuda.is_available():
            idx = torch.device(device).index or 0
            info["gpu_name"] = torch.cuda.get_device_name(idx)
            info["gpu_capability"] = list(torch.cuda.get_device_capability(idx))
    except ImportError:
        info["torch"] = None
    info.update(_nvml_info())
    return info
