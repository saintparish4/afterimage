# Measured throughput

Everything here was measured on the development machine described in
[`environment.md`](environment.md): an AMD Ryzen 5 5625U (8 threads), with
no GPU. **No GPU has been measured yet.** Replace the GPU estimates in
blueprint §11.1 with `afterimage replay --device cuda` output before
planning the pilot.

## Data

| Step | Config | Result |
|---|---|---|
| `data build` (4M-row pool + dev splits, 8 workers) | `pilot.toml` | 116 s wall, 27 MB on disk |
| `data audit` (exact byte-level disjointness of all splits) | `pilot.toml` | 43 s |
| Exhaustive oracle check, all 10,525,160 trees of depth ≤ 3, both semantics | test suite | 14 s |

## Training step (pilot model: 4 layers, d=128, 0.80M parameters, batch 256)

| Measurement | Result |
|---|---|
| `afterimage replay --steps 200 --device cpu` | 1.2 steps/s; replay and resume bit-identical |
| Profiled step, deterministic mode, 4 threads | 0.67 s (forward 0.77 s and backward 0.21 s when timed separately) |
| Profiled step, non-deterministic mode | 0.80 s (deterministic mode costs nothing measurable on CPU) |

The step is about 4·10¹⁰ FLOPs, so at FP32 it is compute-bound, not
launch-bound as blueprint v0.1 assumed. One pilot block (≈ 164k steps) would
take about 40 hours on this CPU, which makes the pilot a GPU job.

## Smoke study (`configs/smoke.toml`: 2 layers, d=32, 3 blocks × 6 arms)

| Measurement | Result |
|---|---|
| `afterimage run` | 0.088 device-hours (about 5 min), ≈ 47 steps/s |
| Full CLI pipeline (build, audit, replay, run, analyze, calibrate, status, audit) | 6.4 min wall |
