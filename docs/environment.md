# Environment

## Locked software

| Component | Version | Source |
|---|---|---|
| Python | 3.12 (`.python-version`) | system or uv-managed |
| torch | 2.14.1 (`+cpu` or `+cu130`) | PyTorch wheel index, via the `cpu` / `cu130` extras |
| numpy, scipy, statsmodels, pyarrow, safetensors | exact versions in `uv.lock` | PyPI |
| pytest, hypothesis | exact versions in `uv.lock` (dev group) | PyPI |
| Container base | `python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df` | Docker Hub, digest looked up 2026-11-19 |
| uv (in the container) | 0.11.29 | `ghcr.io/astral-sh/uv:0.11.29` |

Install with `uv sync --frozen --extra cpu` (or `--extra cu130` on a GPU
machine). The two extras pin the same torch release so CPU and GPU runs
share one code path. The CUDA wheels bundle the CUDA runtime, so a GPU host
needs only an NVIDIA driver recent enough for CUDA 13.0, plus the container
toolkit if you use Docker.

Every run records its own environment in `block.json` and `determinism.json`
(`artifacts/env.py:capture_env`): library versions, torch CUDA/cuDNN
versions, GPU name and driver, CPU model, thread count, determinism flags,
git commit and dirty flag, and a salted host hash.

## Development machine (where this code was written and tested)

| Item | Value |
|---|---|
| OS | Linux 6.18 (WSL2) |
| CPU | AMD Ryzen 5 5625U, 8 threads |
| GPU | none |
| Python | 3.12.3 |
| torch | 2.14.1+cpu |
| numpy / scipy / statsmodels / pyarrow | 2.5.5 / 1.18.1 / 0.15.0 / 26.0.1 |

All tests, including the exhaustive and end-to-end slow tests, pass on this
machine. Bitwise replay and resume pass on its CPU (`afterimage replay`).
They have **not** been checked on any GPU yet.

## Confirmatory device

Not chosen yet. Before the confirmatory run, record here the GPU model,
driver, image digest and the `afterimage replay --device cuda` result. All
arms of a block run in one process on one device. Bitwise equality is
expected only within one GPU model and software stack (blueprint §11.9).
