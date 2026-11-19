# Pilot report (milestone M5)

**Status: exploratory.** These are pilot results on the development splits.
Under the pre-registered plan, pilot effect estimates are never pooled
with confirmatory data and support no claim about whether an afterimage
exists. The pilot's job was to fix the design, check that the machinery
works on real hardware, and calibrate the margins and sample size.

Run on RunPod, RTX 4090 (Secure Cloud), 4 worker processes per GPU,
2026-11-19. Code at commit `a84e79f` (pilot config as revised below).
Artifacts: `results/pilot/` (`dev/report.md`, `dev/analysis.json`,
`calibration.json`); raw runs, see §6.

## 1. Design revisions made during the pilot

Each revision was decided on base-model data only (the gate, before any
arm ran) and is committed with its evidence.

| Attempt | Change | Evidence |
|---|---|---|
| 1 (T0 = 30k) | all 4 bases failed the gate | held-out accuracy 0.97–0.99 (ceiling); loss-spike rule flagged noise around a ~0.005 median; depth 4 below 99% in 2 of 4 blocks |
| → | `ood_min_heldout = 3`, spike floor 0.5 nats, T0 = 50k | the same bases scored 0.91–0.98 on examples with ≥ 3 held-out triples |
| 2 | 1 of 4 bases passed | seeds that learned the familiar task best also generalized best (held-out 0.97–0.98), so the 0.97 cap rejected them |
| → | held-out cap raised to 0.99 | — |
| 3 (final) | 8 of 9 bases passed (block 2 excluded: familiar-task 0.9937, depth 4 0.9876) | — |

Determinism held throughout: the four fork states of attempt 3 are
bit-identical to attempt 2's (same seeds, separate launches, 50,000
steps each).

## 2. Machinery checks (all 8 blocks)

| Check | Result |
|---|---|
| Stage A: every arm starts from the identical fork state | 8/8 |
| Stage B: I adopts the false rule within the window | 7/8 blocks pass (I agreed with XNOR on 98.3–99.9% of affected rows in the blocks checked) |
| Placebo N − C on the primary endpoint | −0.0050 nats, 95% CI [−0.0131, 0.0032], p = 0.19 (null, as required) |
| Post-fork failures | 0 |
| Leakage audit | clean |
| GPU determinism replay | passed |

## 3. Calibration for the confirmatory run (`afterimage calibrate`)

| Quantity | Value |
|---|---|
| ε familiar-task NLL | 0.000348 nats |
| ε familiar-task NLL, affected rows | 0.000801 nats |
| ε familiar-task accuracy | 0.002 (floor) |
| δ_G, smallest held-out effect of interest | 0.00401 nats |
| σ_d, SD of the per-block primary difference | 0.0111 nats |
| n for detection / for equivalence | 63 / 68 |
| **n blocks** | **68** |

## 4. Exploratory estimates (not evidence either way)

* **The intervention takes hold, and correction removes its effect fast.**
  The held-out NLL difference I − C is 2.90 nats at the end of the window
  (h = 0). It falls to within ±0.012 nats by h = 1,250 and is +0.0006 at
  H = 20,000.
* **Primary endpoint (n = 8):** +0.00063 nats, 95% CI [−0.0087, 0.0099].
  Per-block values range from −0.016 to +0.019.
* **Secondaries:** nothing survives Holm correction. The savings probe
  (S6, I re-adopts XNOR faster than C) has unadjusted p = 0.039, and the
  dispersion estimand (S5) p = 0.052. These are exactly the kind of pilot
  signals that must not be interpreted until a pre-registered
  confirmatory run tests them.

## 5. Issue to resolve before freezing the pre-registration

**The held-out endpoint is back at the ceiling by H.** The gate checks
held-out accuracy at T0 (0.93–0.98), but generalization keeps improving
through the window and 20,000 correction steps. At H every arm scores
≈ 99.7% on `ood_dev`. NLL remains continuous, so the analysis is still
valid, but the headroom is small. That is why δ_G is only 0.004 nats and
n is 68. Options include a shorter H, held-out examples with more novel
adjacencies, or accepting n ≈ 68. This is a design decision for the
pre-registration and has not been made.

## 6. Cost and artifact status

Billed by RunPod (pod billing read, 2026-11-19): **$2.44**, of which $0.375
was pod `8fpo9efvptklkf` (attempt 1, stopped, then terminated after its
data was copied) and $2.063 was pod `ghb3vnyue1qnty` (attempts 2–3). The
ledger's device-hours (6.9 for attempt 3) count worker-hours. With 4
workers per GPU that is about 4× the GPU time.

The account balance ran out while the exploratory diagnostics were
running, and RunPod stopped the pod. Consequences:

* `results/pilot/` (the dev report, `analysis.json`, `calibration.json`)
  was copied locally before that and is complete. Every number in this
  report comes from it.
* The final attempt's raw `runs/pilot/` (checkpoints, per-example
  predictions, logs) is only on the stopped pod's `/workspace` disk. The
  first two attempts are local (`runs/pilot_attempt1/`,
  `runs/pilot_attempt2/`).
* The diagnostics (`afterimage diagnose`) did not finish and are not
  reported.
