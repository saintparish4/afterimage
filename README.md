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

> **Status: design stage.** No code has been written yet. The full research
> and engineering design is in [`docs/blueprint.md`](docs/blueprint.md).
> Commands below are the planned interface.

---

## The experiment in one paragraph

A small decoder-only Transformer (about 0.8M parameters) is trained from
scratch on a deterministic Boolean expression language in postfix notation.
The language has the operators `N`, `A`, `O`, `I` and `X`, where the target
operator `X` means XOR. Each *paired seed block* trains a base model, saves its
complete training state (weights, AdamW moments, scheduler, RNG, step), and
forks it into four arms with identical inputs and step counts:

| Arm | What it gets during the intervention window | Purpose |
|---|---|---|
| **C** control | correct labels | reference |
| **I** intervention | `X` supervised as **XNOR** for K steps | the treatment |
| **N** nuisance placebo | correct labels, different data order | measures generic perturbation |
| **S** matched noise | the same number of flipped labels per batch as I, chosen at random | coherent rule vs. noise |

All arms then receive identical corrective training. The primary endpoint is
the paired I−C difference in NLL on a sealed held-out set. That set contains
operator adjacencies (parent, slot, child) that never appear in training, and
the difference is measured only after familiar-task equivalence has been
established by a TOST.

## Design principles

- **One experiment, studied in depth.** No general-purpose framework.
- **Exact counterfactuals.** Deterministic forks of a single saved state
  produce unit-level effects that are computed, not estimated. Replay and
  resume are tested bit for bit.
- **Pre-registration.** The endpoint, horizon, margins and sample size are
  fixed before confirmatory data exist. The confirmatory test set's seed is
  derived from the commit hash of the frozen pre-registration.
- **Controls for the obvious failure modes.** A placebo arm, a matched-noise
  arm, equivalence on the subset of familiar examples the false rule affects,
  and a held-out subset where the true and false rules agree. Together these
  separate a real afterimage from incomplete correction.
- **Behavior first.** CKA, TracIn, probes and activation patching are
  secondary diagnostics. They are never the evidence for the main claim.
- **Minimal compute.** The whole confirmatory study is estimated at a few
  GPU-hours on one GPU. The estimate is replaced by measured throughput before
  scaling.

## Repository layout (planned)

```
configs/          smoke / pilot / confirmatory TOML configs
prereg/           frozen pre-registration (its commit seeds the sealed test set)
src/afterimage/
  lang/           grammar, two independent reference interpreters, generator, intervention
  data/           split builder (Parquet + SHA-256 manifest), stateless batch schedule
  model/          decoder-only Transformer
  train/          TrainState serialization (safetensors), fork, loop, determinism, gate
  experiment/     arms and windows, per-block runner, file-lock launcher
  eval/           metrics, probes, Stage A/B/C checks
  analysis/       endpoints, paired tests, TOST, bootstrap, Holm, power
  diagnostics/    CKA, interpolation barriers, activation patching, TracIn, hybrid forks
  artifacts/      hashing, manifests, environment capture, GPU-hour ledger
tests/            exhaustive + property-based oracle tests, determinism tests
docs/             blueprint, environment, measured throughput, limitations
```

## Planned workflow

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/). CUDA is optional:
every correctness test runs on CPU.

```bash
uv sync --frozen
afterimage data build  --config configs/smoke.toml --verify
afterimage test        --all
afterimage run         --config configs/smoke.toml --blocks all   # CPU, minutes
afterimage analyze     --config configs/smoke.toml --out results/
afterimage audit       --results results/
```

`--confirmatory` analysis refuses to run on a dirty tree, on a config that
does not match the pre-registration, or a second time.

## Roadmap

Milestones are ordered by scientific dependency, not by feature count. Each
milestone has acceptance criteria and failure conditions in the blueprint
(§13).

1. Validate the grammar and the reference interpreters, with exhaustive
   checks up to depth 3.
2. Establish a learnable Transformer baseline that passes the gate.
3. Exact checkpoint forks: bit-identical replay and resume.
4. Confirm that the false supervision has its intended immediate effect.
5. Pilot the design, then freeze the pre-registration.
6. Run the confirmatory corrective-training experiment.
7. Evaluate familiar-task equivalence and held-out generalization on the
   sealed sets.
8. Estimate uncertainty across seed blocks and classify the outcome.
9. Representation diagnostics, as secondary evidence.
10. Package everything for independent replication.

## Reproducibility

Planned releases include source code at a tagged commit, `uv.lock` and a
container digest, frozen configs and the pre-registration, dataset manifests
with checksums, fork and intermediate checkpoints (weights, optimizer and RNG
state), training logs, raw per-example evaluation results, analysis scripts,
and the GPU-hour ledger.

Replication is defined in three tiers: bitwise replay, a same-design rerun,
and an independent rerun. Each tier has explicit success, partial and failure
criteria (blueprint §14).

## Limitations

The project uses one synthetic language, one target operator, one false rule,
one model size, and a finite correction horizon. Claims are scoped to that
design. A study this size cannot say whether the result transfers to
pretrained or large models, and it cannot certify that information was erased.
See blueprint §15.

## License

No license has been chosen yet. Until one is added, all rights are reserved
by the author.
