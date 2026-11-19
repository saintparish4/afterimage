# AFTERIMAGE analysis

- mode: `dev` (split roles: development sets)
- blocks analysed: 8 of 8 targeted
- science hash: `00d757d288a11528`

## Outcome: **uncalibrated**, margins not set; no category can be assigned

- reason: equivalence margins are NaN (pilot calibrates them)

## Primary endpoint

held-out NLL, I - C at horizon H: estimate 0.0006273 nats, 95% CI [-0.008691, 0.009946], BCa [-0.006159, 0.008166], p = 0.878, Wilcoxon p = 0.9453, d_z = 0.05627, n = 8. TOST at delta_G = nan: p = nan.

## Familiar-task equivalence (precondition)

| metric | estimate | 90% CI | margin | TOST p | equivalent |
|---|---|---|---|---|---|
| id_nll | -4.531e-05 | [-0.0001894, 9.879e-05] | nan | nan | False |
| id_aff_nll | 7.941e-05 | [-0.0002845, 0.0004433] | nan | nan | False |
| id_acc | 0 | [-3.101e-05, 3.101e-05] | 0.002 | 3.24e-13 | True |

## Secondary family (Holm)

| id | contrast | estimate | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| S1_heldout_x | G_X: held-out compositions involving X | 0.0009249 | [-0.008357, 0.01021] | 0.8205 | 1 |
| S2_heldout_nonx | G_notX: held-out compositions without X | -0.007097 | [-0.02514, 0.01094] | 0.3832 | 1 |
| S3_invariant | G_inv: rows where both rules agree | 0.001199 | [-0.00597, 0.008368] | 0.7042 | 1 |
| S4_specificity | coherent rule vs matched noise, I - S | 0.001657 | [-0.004894, 0.008208] | 0.5686 | 1 |
| S5_dispersion | kappa = JS(I,C) - JS(N,C) on held-out rows | 0.0005518 | [-4.973e-06, 0.001109] | 0.05158 | 0.2579 |
| S6_savings | rho = AUC_C - AUC_I of false-label NLL during re-exposure (>0: I re-adopts faster) | 0.1577 | [0.01067, 0.3046] | 0.03888 | 0.2333 |

## Placebo (N - C)

estimate -0.004982, 95% CI [-0.01311, 0.003149], p = 0.1907

## Persistence curve (descriptive)

| h | estimate | 95% CI |
|---|---|---|
| 0 | 2.896 | [2.664, 3.128] |
| 1250 | -0.01029 | [-0.03898, 0.0184] |
| 2500 | 0.01203 | [-0.008192, 0.03225] |
| 5000 | 0.01142 | [-0.02284, 0.04567] |
| 10000 | -0.004178 | [-0.03532, 0.02697] |
| 20000 | 0.0006273 | [-0.008691, 0.009946] |

## Validity

- ok  enough_blocks: 8
- ok  single_design: [00d757d288a1152881137b3dbaa50d4851088014e88ed9db3662247c9380f349]
- ok  stage_a_identical_starts: 8
- ok  postfork_failures: 0
- ok  placebo_null: 0.1907
- ok  determinism_replay: cuda
- ok  leakage_audit: []

## Sensitivity (not confirmatory)

- Brier: estimate 0.0002372, 95% CI [-0.001276, 0.00175], sign agrees: True
- crossed (blocks x examples) bootstrap 95% CI: [-0.006929, 0.008042]
- per-protocol (Stage B passes only, 1 excluded): estimate -0.0003317, 95% CI [-0.01113, 0.01047]

## Descriptive

- ood_acc_C: 0.9973
- id_acc_C: 0.9999
- ood_acc_I: 0.9969
- id_acc_I: 0.9999
- ood_acc_N: 0.9979
- id_acc_N: 0.9999
- ood_acc_S: 0.9971
- id_acc_S: 0.9999
- ood_nll_I_minus_C_by_depth: {'2': None, '3': 6.208240728453405e-06, '4': 0.000720967472986857}
- stage_b_I_aff_true_acc_at_window_end: 0.005308
