# AFTERIMAGE pre-registration

**Status: TEMPLATE, not frozen.** Every `⟨…⟩` must be filled from the pilot
before this file is committed as the pre-registration. The hash of that
commit goes into `configs/confirmatory.toml` → `meta.prereg_commit` in a
*follow-up* commit. The sealed confirmatory test sets are derived from that
hash (`afterimage data seal`), so they cannot exist before this document.

Design details, rationale and references are in
[`docs/blueprint.md`](../docs/blueprint.md). This file records only what is
being committed to.

## 1. Question

Does a temporary episode of false supervision (target operator `X` trained
as XNOR instead of XOR for K steps) leave a persistent, causally
attributable difference in held-out compositional generalization after
corrective training has made familiar-task behavior practically equivalent
to an otherwise identical control?

## 2. Frozen design

| Item | Value |
|---|---|
| Config | `configs/confirmatory.toml` at this commit (science hash `⟨hash⟩`) |
| Code | this commit |
| Dataset | data hash `⟨data_hash⟩`, manifest SHA-256 `⟨sha⟩` |
| Model | 4 layers, d=128, 4 heads, d_ff=512, ≈0.80M parameters |
| Fork step T0 | `⟨pilot⟩` |
| Intervention length K | `⟨pilot⟩` |
| Correction horizon H | `⟨pilot⟩` |
| Horizons reported | 0, H/16, H/8, H/4, H/2, H |
| Arms | C, I, N, S (+ exploratory hybrids) |
| Blocks n | `⟨pilot⟩` (from the power analysis below) |
| Confirmatory device type | `⟨GPU model⟩`, determinism replay passed (`runs/confirmatory/determinism.json`) |

## 3. Endpoints

* **Primary:** τ(H), the paired I − C difference in mean answer NLL (nats)
  on `ood_conf`, one value per block, at step T0+K+H.
* **Precondition (familiar-task equivalence):** paired TOST at α = 0.05 for
  I − C on `id_conf` NLL (margin ε = `⟨pilot⟩`), `id_conf` affected-subset
  NLL (ε = `⟨pilot⟩`) and `id_conf` accuracy (ε = `⟨pilot⟩`).
* **Smallest effect of interest:** δ_G = `⟨pilot⟩` nats.
* **Secondary family (Holm, α = 0.05):** S1 G_X, S2 G_¬X, S3 G_inv,
  S4 I − S, S5 dispersion κ, S6 savings ρ.
* **Placebo:** N − C on the primary endpoint. A p-value below 0.01 makes
  the result invalid.

## 4. Analysis

* Primary test: paired t-test on per-block differences, with the 95% CI.
  BCa bootstrap and Wilcoxon are reported alongside. If they disagree about
  excluding zero, the more conservative result is used.
* The outcome category follows `afterimage.analysis.plan.classify` at this
  commit (A1 / A2 / A / B / C / D).
* Sensitivity analyses (reported, not confirmatory): Brier score, crossed
  blocks × examples bootstrap, per-protocol (Stage B passes only),
  worst-case imputation for post-fork failures.

## 5. Power analysis (from the pilot)

| Quantity | Value |
|---|---|
| Pilot blocks | `⟨8⟩` |
| σ_d (SD of the per-block primary difference) | `⟨pilot⟩` |
| n for detection at δ_G (two-sided, α = 0.05, power 0.8) | `⟨pilot⟩` |
| n for TOST equivalence at δ_G (power 0.8 at τ = 0) | `⟨pilot⟩` |
| **n** | `⟨pilot⟩` |

Source: `results/pilot/calibration.json` (`afterimage calibrate --config configs/pilot.toml`).

## 6. Exclusions and failures

* Blocks that fail the learnability gate at T0 (before any treatment) are
  excluded and replaced by the next block index. The analysis uses the
  first n gate-passing blocks by index.
* Blocks that fail the Stage B manipulation check stay in the primary
  analysis (intention to treat).
* Post-fork divergence is recorded, never dropped. More than 10% of blocks
  failing after the fork makes the result invalid.

## 7. Deviations

Any deviation from this document is reported in the results with its
reason. Analyses not listed here are labelled exploratory.
