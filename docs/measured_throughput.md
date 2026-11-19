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

## Exploratory learnability probe (not the gate, not the pilot)

The pilot model and optimizer (`configs/pilot.toml`), trained on CPU for
1,500 steps, which is 5% of the planned T0 = 30,000. Block-0 seed, arm C,
accuracy on the first 2,000 rows of each split:

| step | depth 1 | depth 2 | id_dev depth 3 | id_dev depth 4 | id_dev affected | ood_dev all | ood_dev depth 4 |
|---|---|---|---|---|---|---|---|
| 0 | 0.56 | 0.59 | 0.55 | 0.54 | 0.51 | 0.53 | 0.51 |
| 500 | 1.00 | 0.89 | 0.74 | 0.65 | 0.60 | 0.69 | 0.64 |
| 1000 | 1.00 | 0.95 | 0.80 | 0.70 | 0.66 | 0.72 | 0.67 |
| 1500 | 1.00 | 0.98 | 0.82 | 0.69 | 0.67 | 0.71 | 0.64 |

The model learns quickly at depths 1–2. Depth 4, and rows whose answer
depends on the X node ("affected"), are learned much more slowly. That is
consistent with XOR being parity-like (blueprint §1.6). Whether the
pre-registered gate (≥ 99% at every ID depth, held-out accuracy in
[0.60, 0.97]) can be reached by T0 = 30,000 is **not** established by
this run. Deciding it is M2's job on a GPU, with the escalation ladder in
blueprint §5.3 if needed. Wall time on CPU was about 2.2 s/step at first,
because of CPU contention, and about 0.7 s/step once the CPU was free.
