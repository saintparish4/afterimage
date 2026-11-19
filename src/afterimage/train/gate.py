"""Pre-treatment learnability gate (docs/blueprint.md §5.6).

Evaluated on the base model at T0, before any fork. A block that fails is
excluded and replaced; that is valid because no arm has been treated yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..config import GateConfig


@dataclass
class GateResult:
    passed: bool
    checks: dict[str, dict]

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": self.checks}


def _check(value: float, ok: bool, rule: str) -> dict:
    return {"value": float(value), "ok": bool(ok), "rule": rule}


def evaluate_gate(metrics: dict[str, dict[str, dict]], losses: list[tuple[int, float]], g: GateConfig) -> GateResult:
    """``metrics[split][subset]`` is a summarize() row; ``losses`` is [(step, loss)] for the base phase."""
    c: dict[str, dict] = {}
    small = metrics["id_small"]
    for d in (1, 2):
        r = small.get(f"depth{d}")
        if r:
            c[f"id_small_depth{d}_acc"] = _check(r["acc"], r["acc"] >= g.id_small_min_depth_acc, f">= {g.id_small_min_depth_acc}")
    idd = metrics["id_dev"]
    c["id_dev_acc"] = _check(idd["all"]["acc"], idd["all"]["acc"] >= g.id_min_acc, f">= {g.id_min_acc}")
    for key, r in idd.items():
        if key.startswith("depth"):
            c[f"id_dev_{key}_acc"] = _check(r["acc"], r["acc"] >= g.id_min_depth_acc, f">= {g.id_min_depth_acc}")
    if "has_x" in idd:
        c["id_dev_has_x_acc"] = _check(idd["has_x"]["acc"], idd["has_x"]["acc"] >= g.id_x_min_acc, f">= {g.id_x_min_acc}")
    ood = metrics["ood_dev"]["all"]["acc"]
    c["ood_dev_acc_band"] = _check(ood, g.ood_acc_low <= ood <= g.ood_acc_high, f"in [{g.ood_acc_low}, {g.ood_acc_high}]")

    arr = np.array(losses, dtype=np.float64).reshape(-1, 2)
    finite = bool(np.isfinite(arr[:, 1]).all()) if len(arr) else True
    c["losses_finite"] = _check(float(finite), finite, "no NaN/Inf")
    if len(arr) >= 10:
        last = arr[arr[:, 0] >= arr[-1, 0] - g.loss_trend_window]
        if len(last) >= 5 and np.ptp(last[:, 0]) > 0:
            res = stats.linregress(last[:, 0], last[:, 1])
            lo = res.slope - 1.96 * res.stderr
            c["loss_not_increasing"] = _check(res.slope, lo <= 0, "slope 95% CI includes <= 0")
        med = np.median(arr[len(arr) // 10 :, 1]) if len(arr) > 10 else np.median(arr[:, 1])
        tail = arr[len(arr) // 2 :, 1]
        spikes = int((tail > max(g.spike_factor * med, g.spike_floor)).sum())
        c["no_loss_spikes"] = _check(spikes, spikes == 0,
                                     f"no loss > max({g.spike_factor} x median, {g.spike_floor}) in second half")
    return GateResult(all(v["ok"] for v in c.values()), c)
