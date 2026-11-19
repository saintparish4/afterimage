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

## RunPod RTX 4090 (pod `8fpo9efvptklkf`, EUR-IS-2, driver 595.112.07, host CUDA 13.2, 12 vCPUs)

| Measurement | Result |
|---|---|
| `data build` (pilot) | 63 s. Pool SHA-256 identical to the local build (`e0a1c9f36a51…`), so the data is bitwise reproducible across machines |
| `data audit` | all 23 checks pass |
| `afterimage replay --steps 500` | **replay and resume bit-identical on the GPU**; 34.8 steps/s (for-loop AdamW), 43.9 steps/s (foreach AdamW) |
| Step profile, deterministic, foreach | 20.6 ms per step (forward+backward 17.4 ms, optimizer 1.7 ms). CPU-dispatch bound, not compute bound |
| 4 concurrent workers on one GPU | ≈ 20–22 ms per step each, ≈ 190 steps/s aggregate |
| 8 concurrent workers on one GPU | ≈ 60 ms per step each (the 12 vCPUs are saturated), ≈ 133 steps/s aggregate, ≈ 3.2 GB GPU memory per process |
| One block, 4 workers | base phase (30k steps) ≈ 11 min; a full block ≈ 1 h |

## First pilot gate result (blocks 0-3, T0 = 30,000)

All four base models **failed** the pre-registered learnability gate.
The pilot was stopped there, and no arms were run. The failures:

| block | id_dev acc | id_dev depth-4 acc | X-containing acc | ood_dev acc (band 0.60-0.97) | "loss spikes" |
|---|---|---|---|---|---|
| 000 | 0.9899 ✗ | 0.9814 ✗ | 0.9873 ✗ | 0.9708 ✗ | 135 ✗ |
| 001 | 0.9973 | 0.9947 | 0.9961 | 0.9919 ✗ | 93 ✗ |
| 002 | 0.9915 ✗ | 0.9837 ✗ | 0.9880 ✗ | 0.9741 ✗ | 10 ✗ |
| 003 | 0.9977 | 0.9954 | 0.9969 | 0.9895 ✗ | 41 ✗ |

Depths 1–3 are at 99.85–100% in every block. Three findings:

1. **The held-out compositions are at the ceiling** (97–99%). The model
   generalizes to unseen operator adjacencies almost perfectly, which is
   above the band meant to leave room for an effect.
2. **The loss-spike check is miscalibrated.** Median loss after warm-up
   is only 0.005–0.015. "3 × median" therefore flags ordinary minibatch
   noise: the largest second-half loss in any block is 0.10, losses are
   finite, and the loss trend is flat.
3. **Depth 4 is not reliably at 99% by T0** (2 of 4 blocks at 98.1–98.4%).
