"""Arms, batch plans and the learning-rate schedule.

``plan_batch`` is the *only* place where arms differ. It is a pure function
of (arm, step), which is what the intervention-contract tests check:

* C: main schedule, true labels.
* I: main schedule. Inside the window, the false-rule labels.
* N: inside the window, the ``alt`` schedule (different inputs, true labels).
  Outside it, the main schedule.
* S: main schedule. Inside the window, the labels of ``|A_t|`` uniformly
  random positions are flipped, where ``A_t`` is the set the false rule would
  change in this batch.
* HYB_*: hybrid forks start after the window, so they always behave like C.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..data.schedule import batch_indices, sham_positions

ARM_KINDS = ("C", "I", "N", "S", "HYB")


@dataclass(frozen=True)
class ArmSpec:
    name: str
    kind: str
    weights_from: str | None = None  # hybrids only
    opt_from: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ARM_KINDS:
            raise ValueError(f"unknown arm kind {self.kind!r}")


def arms_for(cfg: Config) -> list[ArmSpec]:
    arms = [ArmSpec(a, a) for a in cfg.design.arms]
    if cfg.design.hybrids:
        arms += [
            ArmSpec("HYB_wI_oC", "HYB", weights_from="I", opt_from="C"),
            ArmSpec("HYB_wC_oI", "HYB", weights_from="C", opt_from="I"),
        ]
    return arms


@dataclass(frozen=True)
class BatchPlan:
    step: int
    key: str
    idx: np.ndarray     # pool indices
    y: np.ndarray       # labels used for training (0/1)
    in_window: bool
    n_affected: int     # |A_t|: rows the false rule would change
    n_wrong: int        # rows whose training label differs from the true label


def plan_batch(
    kind: str, step: int, window: range, block_seed: int, batch: int,
    y_true: np.ndarray, y_false: np.ndarray,
) -> BatchPlan:
    in_w = step in window
    key = "alt" if (kind == "N" and in_w) else "main"
    idx = batch_indices(block_seed, key, step, len(y_true), batch)
    yt, yf = y_true[idx], y_false[idx]
    n_aff = int((yt != yf).sum())
    y = yt.copy()
    if in_w and kind == "I":
        y = yf.copy()
    elif in_w and kind == "S":
        pos = sham_positions(block_seed, step, batch, n_aff)
        y[pos] ^= 1
    return BatchPlan(step, key, idx, y, in_w, n_aff, int((y != yt).sum()))


def plan_relearn(r: int, cfg: Config, block_seed: int, y_true: np.ndarray, y_false: np.ndarray) -> BatchPlan:
    """Savings probe step r: replay window batches (same for every arm) with false labels."""
    d = cfg.design
    step = d.T0 + (r % d.K)
    idx = batch_indices(block_seed, "main", step, len(y_true), cfg.optim.batch)
    yt, yf = y_true[idx], y_false[idx]
    return BatchPlan(step, "main", idx, yf.copy(), True, int((yt != yf).sum()), int((yt != yf).sum()))


@dataclass(frozen=True)
class LRSchedule:
    """Linear warmup, constant, then a linear decay over the last part of correction.

    The learning rate is a pure function of the global step, so there is no
    scheduler state to save or fork.
    """

    peak: float
    warmup: int
    decay_start: int
    decay_end: int
    final_frac: float

    @staticmethod
    def from_config(cfg: Config) -> "LRSchedule":
        d, o = cfg.design, cfg.optim
        corr0 = d.T0 + d.K
        return LRSchedule(
            peak=o.lr, warmup=o.warmup,
            decay_start=corr0 + int(round((1 - o.final_decay_frac) * d.H)),
            decay_end=d.end, final_frac=o.final_lr_frac,
        )

    def __call__(self, step: int) -> float:
        if step < self.warmup:
            return self.peak * (step + 1) / self.warmup
        if step >= self.decay_start:
            span = max(1, self.decay_end - self.decay_start)
            frac = min(1.0, (step - self.decay_start) / span)
            return self.peak * (1.0 - (1.0 - self.final_frac) * frac)
        return self.peak
