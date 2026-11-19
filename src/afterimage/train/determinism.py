"""Determinism switches (docs/blueprint.md §6.5).

``configure`` must run before the first CUDA call in the process, because
cuBLAS reads ``CUBLAS_WORKSPACE_CONFIG`` when its handle is created.
"""

from __future__ import annotations

import os

import torch

from ..config import RuntimeConfig


def configure(rt: RuntimeConfig) -> dict:
    if rt.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    if rt.threads > 0:
        torch.set_num_threads(rt.threads)
    return flags()


def flags() -> dict:
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "threads": torch.get_num_threads(),
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def resolve_device(name: str) -> torch.device:
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device {name!r} requested but CUDA is not available (use --device cpu)")
    return dev
