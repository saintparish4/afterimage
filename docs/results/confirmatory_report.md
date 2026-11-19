# AFTERIMAGE analysis

- mode: `sealed` (split roles: sealed confirmatory sets)
- blocks analysed: 16 of 16 targeted
- science hash: `5e5d71edf731b419`

## Outcome: **C**, inconclusive

## Primary endpoint

held-out NLL, I - C at horizon H: estimate 0.002334 nats, 95% CI [-0.0002764, 0.004945], BCa [0.0003596, 0.005107], p = 0.07602, Wilcoxon p = 0.0654, d_z = 0.4765, n = 16. TOST at delta_G = 0.004012: p = 0.0954.

## Familiar-task equivalence (precondition)

| metric | estimate | 90% CI | margin | TOST p | equivalent |
|---|---|---|---|---|---|
| id_nll | 5.714e-05 | [-0.0001018, 0.000216] | 0.0003479 | 0.002934 | True |
| id_aff_nll | 0.0002331 | [-0.0001247, 0.0005909] | 0.0008006 | 0.006999 | True |
| id_acc | -3.125e-06 | [-2.348e-05, 1.723e-05] | 0.002 | 1.956e-26 | True |

## Secondary family (Holm)

| id | contrast | estimate | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| S1_heldout_x | G_X: held-out compositions involving X | 0.002456 | [-0.0002233, 0.005136] | 0.06963 | 0.3481 |
| S2_heldout_nonx | G_notX: held-out compositions without X | -0.0006545 | [-0.002771, 0.001462] | 0.5199 | 1 |
| S3_invariant | G_inv: rows where both rules agree | 0.002111 | [0.0001264, 0.004095] | 0.03859 | 0.2315 |
| S4_specificity | coherent rule vs matched noise, I - S | 0.001304 | [-0.002548, 0.005157] | 0.4817 | 1 |
| S5_dispersion | kappa = JS(I,C) - JS(N,C) on held-out rows | -5.924e-05 | [-0.0003823, 0.0002638] | 0.7014 | 1 |
| S6_savings | rho = AUC_C - AUC_I of false-label NLL during re-exposure (>0: I re-adopts faster) | -0.02344 | [-0.1102, 0.0633] | 0.5732 | 1 |

## Placebo (N - C)

estimate 0.002653, 95% CI [-4.121e-05, 0.005347], p = 0.05317

## Persistence curve (descriptive)

| h | estimate | 95% CI |
|---|---|---|
| 0 | 2.597 | [2.337, 2.856] |
| 1250 | 0.04451 | [-0.02677, 0.1158] |
| 2500 | 0.02995 | [-0.02174, 0.08163] |
| 5000 | 0.000153 | [-0.02824, 0.02854] |
| 10000 | -0.01405 | [-0.04388, 0.01577] |
| 20000 | 0.002334 | [-0.0002764, 0.004945] |

## Validity

- ok  enough_blocks: 16
- ok  single_design: [5e5d71edf731b419003001b5559bcbedd6e76f924e15e71360080d17bddfd03e]
- ok  stage_a_identical_starts: 16
- ok  postfork_failures: 0
- ok  placebo_null: 0.05317
- ok  determinism_replay: cuda (NVIDIA GeForce RTX 3090), cuda (NVIDIA GeForce RTX 4090), cuda (NVIDIA RTX PRO 4000 Blackwell), cuda (NVIDIA RTX PRO 4500 Blackwell)
- ok  leakage_audit: []

## Sensitivity (not confirmatory)

- Brier: estimate 0.0001783, 95% CI [-0.0002047, 0.0005613], sign agrees: True
- crossed (blocks x examples) bootstrap 95% CI: [-0.000211, 0.00521]
- per-protocol (Stage B passes only, 3 excluded): estimate 0.003773, 95% CI [0.001252, 0.006293]

## Descriptive

- ood_acc_C: 0.9982
- id_acc_C: 0.9999
- ood_acc_I: 0.9981
- id_acc_I: 0.9999
- ood_acc_N: 0.9978
- id_acc_N: 0.9999
- ood_acc_S: 0.9981
- id_acc_S: 0.9999
- ood_nll_I_minus_C_by_depth: {'2': None, '3': 7.56723781304594e-07, '4': 0.0026437585844819806}
- stage_b_I_aff_true_acc_at_window_end: 0.01256
