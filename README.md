# AFTERIMAGE

**A counterfactual laboratory for training-history dependence in neural networks.**

AFTERIMAGE asks one question:

> Can a temporary episode of incorrect supervision produce a persistent,
> causally attributable difference in a neural network's future
> generalization, even after corrective training has made its familiar-task
> performance practically equivalent to that of an otherwise identical control
> model?

A *behavioral afterimage* is this project's working name for such a
difference. It is an operational term defined in the blueprint, not an
established phenomenon, and the project does not assume it exists. Three
results are possible, and all three are publishable outcomes: the effect is
detected, it is bounded by an equivalence test, or the study shows that it
could not be resolved.

> **Status: complete.** The pre-registered confirmatory study (n = 16 seed
> blocks, run on RunPod GPUs) finished with outcome **C, inconclusive**.
> Familiar-task equivalence was established and every validity check
> passed. The held-out difference after correction was +0.0023 nats, 95% CI
> [−0.0003, +0.0049], and it was not distinguishable from the difference a
> data-order placebo produces. The study was underpowered by design (power
> 0.27 at the smallest effect of interest), so this is not evidence of
> absence. Read [`docs/results.md`](docs/results.md) first. The design is
> in [`docs/blueprint.md`](docs/blueprint.md), the pilot in
> [`docs/pilot_report.md`](docs/pilot_report.md), and the frozen plan in
> [`prereg/preregistration.md`](prereg/preregistration.md).

---

## The experiment in one paragraph

A small decoder-only Transformer (0.80M parameters) is trained from scratch
on a deterministic Boolean expression language in postfix notation. The
language has the operators `N`, `A`, `O`, `I` and `X`, where the target
operator `X` means XOR. Each *paired seed block* trains a base model,
checks a pre-treatment learnability gate, saves its complete training state
(weights, AdamW moments, RNG, step), and forks it into arms with identical
inputs and step counts:

| Arm | During the intervention window | Purpose |
|---|---|---|
| **C** control | correct labels | reference |
| **I** intervention | `X` supervised as **XNOR** for K steps | the treatment |
| **N** nuisance placebo | correct labels, different data order | measures generic perturbation |
| **S** matched noise | the same number of flipped labels per batch as I, at random | coherent rule vs. noise |
| hybrids (exploratory) | I's weights with C's optimizer state, and the reverse | where the effect is carried |

All arms then receive identical corrective training. The primary endpoint
is the paired I−C difference in NLL on a sealed held-out set of operator
adjacencies (parent, slot, child) that never occur in training. It is
interpreted only once familiar-task equivalence has been established by a
TOST.

## Quick start (CPU)

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned.

```bash
uv sync --frozen --extra cpu                      # or --extra cu130 on a GPU machine
uv run pytest                                     # fast suite
uv run pytest -m slow                             # exhaustive oracle check (10.5M trees) + end-to-end runs

C="--config configs/smoke.toml"
uv run afterimage data build  $C                  # deterministic, checksummed splits
uv run afterimage data audit  $C                  # exact leakage audit
uv run afterimage replay      $C --steps 200      # bitwise replay + resume on this device
uv run afterimage run         $C                  # 3 paired blocks (~6 min on a laptop CPU)
uv run afterimage analyze     $C --audit          # -> results/smoke/dev/report.md
```

The smoke model is far too small to learn the task, so its outcome is
expected to be uninformative (it reports "correction incomplete"). It exists
to exercise every code path.

## Running the study (GPU)

```bash
P="--config configs/pilot.toml"
uv run afterimage data build $P && uv run afterimage replay $P --device cuda
uv run afterimage launch $P --devices cuda:0 --workers-per-device 2 --budget-gpu-hours 10
uv run afterimage analyze $P && uv run afterimage calibrate $P   # margins + n for the confirmatory run
```

Then fill `prereg/preregistration.md` and `configs/confirmatory.toml` from
`results/pilot/calibration.json`, and commit them. Next, write that commit's
hash into `meta.prereg_commit` and commit again. After that:

```bash
F="--config configs/confirmatory.toml"
uv run afterimage data seal $F          # sealed test sets, seeded by the prereg commit
uv run afterimage replay $F --device cuda
uv run afterimage launch $F --devices cuda:0 --workers-per-device 2
uv run afterimage evaluate-sealed $F
uv run afterimage analyze $F --confirmatory     # runs once; refuses on a dirty tree or a changed design
```

`scripts/run_then_shutdown.sh` wraps `launch` for on-demand nodes, and the
`Dockerfile` provides a digest-pinned environment. Measured: about 5–20 ms
per training step depending on the host CPU, and about 1 h per block with
4–6 blocks per GPU. The whole study (pilot plus confirmatory) cost $4.69
on RunPod. To split a study across machines without shared storage, give
each machine a disjoint `run --blocks` list and combine the results with
`afterimage merge`.

## Commands

| Command | What it does |
|---|---|
| `data build / verify / audit / seal` | build splits; check hashes; exact leakage audit; sealed confirmatory sets |
| `replay` | check bitwise replay and kill-and-resume on this device |
| `run` / `launch` | run blocks in-process / spawn workers across GPUs (file locks, budget guard, ledger) |
| `evaluate-sealed` | predict the sealed splits from saved horizon checkpoints |
| `analyze [--confirmatory]` | pre-registered analysis and outcome category (A1/A2/A/B/C/D) |
| `calibrate` | pilot-only: equivalence margins, smallest effect of interest, sample size |
| `diagnose` | exploratory CKA, barriers, activation patching, gradient alignment, TracIn |
| `reproduce-r1` | rerun forks from released fork states and require identical bits |
| `merge` | combine study directories from machines that ran disjoint blocks |
| `status` / `audit` | block statuses and device-hours; data, leakage, determinism and Stage A checks |

## Design principles

- **One experiment, studied in depth.** No general-purpose framework.
- **Exact counterfactuals.** Arms fork a single saved state, and batches are
  a pure function of the step, so unit-level effects are computed, not
  estimated. Replay and crash-resume are tested bit for bit.
- **Pre-registration.** The endpoint, horizon, margins and sample size are
  fixed before confirmatory data exist. The sealed test set's seed comes
  from the commit hash of the frozen pre-registration.
- **Controls for the obvious failure modes.** A placebo arm, a matched-noise
  arm, equivalence on the familiar examples the false rule affects, and a
  held-out subset where both rules agree. Together they separate a real
  afterimage from incomplete correction.
- **Behavior first.** Representation and attribution diagnostics are
  secondary, are always compared with the placebo, and cannot change the
  outcome category.

## Repository layout

```
configs/          smoke / pilot / confirmatory
prereg/           the pre-registration (template until the pilot fills it)
scripts/          reproduce.sh (smoke | r1 | r2), run_then_shutdown.sh
src/afterimage/
  lang/           grammar, three independent interpreters, generator, enumerator
  data/           split builder (Parquet + SHA-256 manifest, audit, sealed sets), batch schedule
  model/          decoder-only Transformer
  train/          arms and batch plans, checkpoints and forks, deterministic step, gate
  eval/           predictions, metrics, subsets, Stage A/B checks
  experiment/     per-block runner, launcher, replay checks, sealed evaluation
  analysis/       endpoints, paired statistics, classification, calibration, report
  diagnostics/    CKA, barriers, activation patching, TracIn
  artifacts/      hashing, environment capture, device-hour ledger
tests/            mirrors src/; `-m slow` for exhaustive and end-to-end checks
docs/             blueprint, pilot report, results (+ official outputs in docs/results/),
                  environment, measured throughput
```

## Reproducibility

The project's releases will include source code at a tagged commit,
`uv.lock` and the container digest, frozen configs and the pre-registration,
dataset manifests with checksums, fork and intermediate checkpoints
(weights, optimizer and RNG state), logs, per-example predictions, analysis
outputs and the device-hour ledger. Replication is defined in three tiers:

1. **R1, bitwise.** Rerun forks from the released fork states (`reproduce-r1`).
2. **R2, same-design rerun.** Rerun the whole study with the released seeds.
3. **R3, independent rerun.** Use a fresh master seed, other hardware, or a
   reimplementation.

Each tier has explicit success, partial and failure criteria (blueprint
§14).

## Limitations

The project uses one synthetic language, one target operator, one false
rule, one model size, and a finite correction horizon. Claims are scoped to
that design. A study this size cannot say whether the result transfers to
pretrained or large models, and it cannot certify that information was
erased. See blueprint §15.

## License

No license has been chosen yet. Until one is added, all rights are reserved
by the author.
