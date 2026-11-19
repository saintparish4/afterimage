# AFTERIMAGE pre-registration

**Status: FROZEN** at the commit that introduced this version of the file.
That commit's hash is recorded in `configs/confirmatory.toml` →
`meta.prereg_commit` (a follow-up commit). The sealed confirmatory test sets
are derived from that hash (`afterimage data seal`), so they did not exist
when this plan was written.

Design details, rationale and references are in
[`docs/blueprint.md`](../docs/blueprint.md). The pilot that fixed every
value below is in [`docs/pilot_report.md`](../docs/pilot_report.md).

## 1. Question

Does a temporary episode of false supervision (target operator `X` trained
as XNOR instead of XOR for K steps) leave a persistent, causally
attributable difference in held-out compositional generalization after
corrective training has made familiar-task behavior practically equivalent
to an otherwise identical control?

## 2. Frozen design

| Item | Value |
|---|---|
| Config | `configs/confirmatory.toml` at this commit (science hash `5e5d71edf731b419…`) |
| Code | this commit |
| Dataset | data hash `9f9f20644e10a32c`, pool content digest prefix `e0a1c9f36a51` (rebuilt deterministically on each machine; `data verify` and `data audit` must pass) |
| Held-out set | depth 2–4, every example contains at least 3 of the 6 held-out triples |
| Model | 4 layers, d=128, 4 heads, d_ff=512, ≈0.80M parameters, AdamW (foreach) |
| Fork step T0 | 50,000 |
| Intervention length K | 2,000 |
| Correction horizon H | 20,000 |
| Horizons reported | 0, 1,250, 2,500, 5,000, 10,000, 20,000 |
| Arms | C, I, N, S (+ exploratory hybrids) |
| Gate | as in the config: familiar task ≥ 0.995 overall and ≥ 0.99 per depth, held-out accuracy in [0.60, 0.99], no loss above max(3 × median, 0.5) in the second half of base training |
| Blocks | **n = 16**: the first 16 blocks, by index among 0–19, that pass the gate |
| Execution | blocks 0–19 split into disjoint explicit lists across RunPod GPU pods. All arms of a block run in one process on one device. Every pod must pass `afterimage replay` before training, and the per-pod study directories are combined with `afterimage merge` |

## 3. Endpoints

* **Primary:** τ(H), the paired I − C difference in mean answer NLL (nats)
  on `ood_conf`, one value per block, at step T0+K+H.
* **Precondition (familiar-task equivalence):** paired TOST at α = 0.05 for
  I − C on `id_conf` NLL (ε = 0.000347913), on `id_conf` affected-subset
  NLL (ε = 0.000800566), and on `id_conf` accuracy (ε = 0.002).
* **Smallest effect of interest:** δ_G = 0.00401223 nats.
* **Secondary family (Holm, α = 0.05):** S1 G_X, S2 G_¬X, S3 G_inv,
  S4 I − S, S5 dispersion κ, S6 savings ρ.
* **Placebo:** N − C on the primary endpoint. A p-value below 0.01 makes
  the result invalid.

## 4. Analysis

* Primary test: paired t-test on per-block differences, with the 95% CI.
  BCa bootstrap and Wilcoxon are reported alongside. If they disagree about
  excluding zero, the more conservative result is used.
* The outcome category follows `afterimage.analysis.plan.classify` at this
  commit (A1 / A2 / A / B / C / D). The analysis runs once, with
  `afterimage analyze --confirmatory`.
* Sensitivity analyses (reported, not confirmatory): Brier score, crossed
  blocks × examples bootstrap, per-protocol (Stage B passes only),
  worst-case imputation for post-fork failures.

## 5. Power analysis, and why n = 16

| Quantity | Value |
|---|---|
| Pilot blocks | 8 |
| σ_d (SD of the per-block primary difference) | 0.0111 nats |
| n for detection at δ_G (two-sided, α = 0.05, power 0.8) | 63 |
| n for TOST equivalence at δ_G (power 0.8 at τ = 0) | 68 |
| **n used** | **16** (the blueprint's minimum; chosen for time and cost) |
| Power to detect τ = δ_G at n = 16 | 0.27 |
| Minimum detectable effect at n = 16 (power 0.8) | 0.0084 nats |
| Power to establish equivalence at n = 16 | ≈ 0 |

Consequences, stated before seeing any confirmatory data:

* This run can **detect** an afterimage of about 0.008 nats or larger. A
  true effect near δ_G is more likely than not to be missed.
* It **cannot establish practical equivalence** (category B) at δ_G.
* A null or small effect will most likely be classified **C
  (inconclusive)**. That outcome will be reported as "underpowered", not
  as "no afterimage".

Source: `results/pilot/calibration.json` (`afterimage calibrate --config
configs/pilot.toml`).

## 6. Exclusions and failures

* Blocks that fail the learnability gate at T0 (before any treatment) are
  excluded and replaced by the next block index. The analysis uses the
  first 16 gate-passing blocks by index.
* Blocks that fail the Stage B manipulation check stay in the primary
  analysis (intention to treat).
* Post-fork divergence is recorded, never dropped. More than 10% of blocks
  failing after the fork makes the result invalid.

## 7. Deviations

Any deviation from this document is reported in the results with its
reason. Analyses not listed here are labelled exploratory.
