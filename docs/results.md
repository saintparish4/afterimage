# AFTERIMAGE results

**Outcome of the pre-registered confirmatory study: category C,
inconclusive.** The study is valid on every pre-registered check and
familiar-task equivalence was established. The primary 95% CI includes
zero. With n = 16 the study had power 0.27 at the smallest effect of
interest, so this is an underpowered null, not evidence that no
afterimage exists.

* Pre-registration: [`prereg/preregistration.md`](../prereg/preregistration.md), frozen at commit `94c23f7`.
* Analysis output (run once, `afterimage analyze --confirmatory`, at commit `5437a1c`, clean tree):
  [`results/confirmatory_report.md`](results/confirmatory_report.md),
  [`results/confirmatory_analysis.json`](results/confirmatory_analysis.json).
* Pilot: [`pilot_report.md`](pilot_report.md).

## 1. What was run

* Design: blueprint v0.2 as revised by the pilot. A 0.80M-parameter
  Transformer, base T0 = 50,000 steps, false-rule window K = 2,000,
  correction H = 20,000. Arms C, I, N, S, plus two exploratory hybrids.
* Blocks 0–19 were seeded from the confirmatory master seed. Blocks 7, 12
  and 17 failed the learnability gate before treatment and were excluded.
  The analysis used the first 16 passing blocks: 0–6, 8–11, 13–16, 18.
  Block 19 (passed, not needed) and a contingency block 20 were stopped
  unfinished and are not analysed.
* Hardware: RunPod Secure Cloud, one RTX 4090, one RTX PRO 4500, one RTX
  PRO 4000 and one RTX 3090. Each pod ran disjoint blocks, and all arms
  of a block ran in one process on one GPU. Every pod passed the
  bit-identical replay and resume check before training
  ([`results/confirmatory_determinism.json`](results/confirmatory_determinism.json)).
  Every pod rebuilt the dataset with identical hashes, and derived
  byte-identical sealed test sets from the pre-registration commit.

## 2. Validity (all pre-registered checks pass)

| Check | Result |
|---|---|
| Stage A: every arm starts from the identical fork | 16/16 |
| Post-fork failures | 0 |
| Placebo N − C on the primary endpoint | +0.0027 nats, 95% CI [−0.00004, +0.0053], p = 0.053 (threshold for invalidity: p < 0.01) |
| Determinism replay | passed on all four GPU types |
| Leakage audit | clean |
| Single design (science hash) | yes, `5e5d71edf731b419…` |

## 3. Familiar-task equivalence (precondition): established

| Metric | I − C | 90% CI | Margin | TOST p |
|---|---|---|---|---|
| NLL, all rows | +0.000057 | [−0.00010, +0.00022] | 0.00035 | 0.003 |
| NLL, affected rows | +0.00023 | [−0.00012, +0.00059] | 0.00080 | 0.007 |
| Accuracy | −0.000003 | [−0.000023, +0.000017] | 0.002 | < 10⁻²⁵ |

After 20,000 corrective steps, the intervention models are equivalent to
the controls on the familiar task within the pre-registered margins.

## 4. Primary endpoint

**τ(H) = +0.00233 nats, 95% CI [−0.00028, +0.00494], p = 0.076, n = 16.**
d_z = 0.48. The secondary statistics:

* The BCa bootstrap CI [+0.00036, +0.0051] excludes zero, but the t CI
  does not. The pre-registered rule takes the more conservative result,
  so zero is not excluded.
* Wilcoxon p = 0.065.
* TOST against δ_G = 0.004 nats gives p = 0.095, so equivalence is not
  established either.

The category is therefore **C**.

## 5. Secondary family (Holm-corrected): nothing significant

| | Estimate | 95% CI | Holm p |
|---|---|---|---|
| S1 G_X (held-out compositions involving X) | +0.0025 | [−0.0002, +0.0051] | 0.35 |
| S2 G_¬X | −0.0007 | [−0.0028, +0.0015] | 1.00 |
| S3 G_inv (both rules agree) | +0.0021 | [+0.0001, +0.0041] | 0.23 |
| S4 I − S (coherent rule vs. matched noise) | +0.0013 | [−0.0025, +0.0052] | 1.00 |
| S5 dispersion κ | −0.00006 | [−0.0004, +0.0003] | 1.00 |
| S6 savings ρ | −0.023 | [−0.110, +0.063] | 1.00 |

The pilot's suggestive savings signal (S6, unadjusted p = 0.039 in 8
blocks) did not replicate: the confirmatory estimate has the opposite
sign.

## 6. Descriptive and exploratory

* **Persistence.** At the end of the window the intervention raises
  held-out NLL by **2.60 nats** (95% CI [2.34, 2.86]): the false rule was
  learned. The difference is +0.045 by h = 1,250, +0.0002 by h = 5,000,
  and +0.0023 at H. Correction removes almost all of the effect within
  the first few thousand steps.
* **Intervention vs. placebo (exploratory, not pre-registered).** I − N =
  −0.0003 nats, 95% CI [−0.0037, +0.0031], p = 0.84. The small upward
  tilt of I − C is matched by the placebo N − C (+0.0027), which only
  reshuffled the data order during the window. Whatever residue there is
  shows no sign of being specific to the false rule. This fits the
  blueprint's "generic perturbation" hypothesis (§10.7).
* **Stage B.** 3 of 16 blocks did not fully meet the manipulation-check
  thresholds and stayed in the primary analysis (intention to treat). The
  per-protocol sensitivity estimate excluding them is +0.0038, 95% CI
  [+0.0013, +0.0063]. That is a sensitivity analysis, not a confirmatory
  result.
* **Ceiling.** Held-out accuracy at H is about 99.8% in every arm, as
  the pilot warned. Effects can only show up as small shifts in
  confidence.

## 7. What this result does and does not say

* It **does** show that, in this setting, the false rule is learned, then
  corrected to familiar-task equivalence. After correction, any remaining
  held-out generalization difference is smaller than about 0.005 nats
  (upper 95% bound). It is also not distinguishable from the difference
  an innocuous data-order perturbation produces.
* It **does not** show that there is no afterimage. The study had 0.27
  power at the pre-registered smallest effect of interest (0.004 nats).
  The pilot's power analysis asked for n = 68.
* Scope: one synthetic language, one false rule, one 0.8M-parameter model,
  one correction horizon (blueprint §15).

## 8. TBD: the fully powered run (n = 68)

**Status: not run. Planned for later.**

| Item | Plan |
|---|---|
| Goal | reach the n = 68 blocks the pilot power analysis asked for (power 0.8 for both detection and equivalence at δ_G = 0.004 nats) |
| Design | unchanged and frozen. The same config and pre-registered analysis; the 16 analysed blocks plus 52 more from block 21 onward |
| Additions | a new pre-registration amendment, before any new data, that makes I − N a pre-registered secondary |
| Compute | about 4 hours of wall clock on 4 RunPod GPU pods (5 blocks per pod at a time, ~1–1.3 h per round) |
| Cost estimate | about $5–8 at the rates billed for the n = 16 run ($2.68/h for four pods) |
| What it can decide | between "persistent difference detected" (A), "practically equivalent" (B), and inconclusive (C) at the pre-registered margin |

## 9. Cost

From RunPod's pod billing read:

* Confirmatory run, four pods: **$1.65** (RTX 4090 $0.55, RTX PRO 4500
  $0.44, RTX PRO 4000 $0.35, RTX 3090 $0.31).
* Pilot, two pods: $3.04.
* **Total: $4.69.**
