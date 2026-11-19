"""Pilot calibration of margins and sample size (docs/blueprint.md §2.7, §9.2).

* eps_m: the ``eps_quantile`` quantile of |m_N - m_C| at H over pilot blocks,
  for each familiar-task metric. Accuracy is floored at ``eps_id_acc``.
* delta_G: ``delta_sd_frac`` x the between-block SD of control held-out NLL at H.
* sigma_d: SD of the per-block primary difference I - C.
* n: max(n for detection, n for TOST equivalence, n_min).

Pilot *effect* estimates are reported as exploratory and never reused.
"""

from __future__ import annotations

import math

import numpy as np

from .endpoints import Cube
from .plan import ID_EQUIV, PRIMARY
from .stats import n_for_detection, n_for_equivalence


def calibrate(cube: Cube) -> dict:
    cfg = cube.cfg
    a, d = cfg.analysis, cfg.design
    H = d.T0 + d.K + d.H
    out: dict = {"blocks": cube.blocks, "horizon_step": H}
    eps = {}
    for c in ID_EQUIV:
        diff = np.abs(cube.values("N", H, c.role, c.subset, c.metric) - cube.values("C", H, c.role, c.subset, c.metric))
        q = float(np.nanquantile(diff, a.eps_quantile)) if np.isfinite(diff).any() else math.nan
        if c.key == "id_acc":
            q = max(q, a.eps_id_acc) if np.isfinite(q) else a.eps_id_acc
        eps[c.key] = q
    yc = cube.values("C", H, PRIMARY.role, PRIMARY.subset, PRIMARY.metric)
    delta = a.delta_sd_frac * float(np.nanstd(yc, ddof=1)) if np.isfinite(yc).sum() >= 2 else math.nan
    di = cube.values("I", H, "ood", "all", "nll") - yc
    sigma = float(np.nanstd(di, ddof=1)) if np.isfinite(di).sum() >= 2 else math.nan
    n_det = n_for_detection(sigma, delta, a.alpha, a.power) if np.isfinite(sigma * delta) else None
    n_eq = n_for_equivalence(sigma, delta, a.alpha, a.power) if np.isfinite(sigma * delta) else None
    n = max(x for x in (n_det, n_eq, a.n_min) if x is not None)
    # Diagnostic only: how many pilot blocks would pass familiar-task equivalence block by block.
    within = {}
    for c in ID_EQUIV:
        diff = np.abs(cube.values("I", H, c.role, c.subset, c.metric) - cube.values("C", H, c.role, c.subset, c.metric))
        within[c.key] = float(np.nanmean(diff <= eps[c.key])) if np.isfinite(eps[c.key]) else None
    stage_b = [cube.stage(b).get("B", {}).get("ok") for b in cube.blocks]
    out |= {
        "eps": eps,
        "delta_G": delta,
        "sigma_d": sigma,
        "n_detection": n_det,
        "n_equivalence": n_eq,
        "n_blocks": n,
        "n_exceeds_max": bool(n > a.n_max),
        "diagnostics": {
            "fraction_blocks_I_within_eps_of_C": within,
            "stage_b_pass_rate": float(np.mean([x is True for x in stage_b])) if stage_b else None,
            "exploratory_pilot_primary_estimate": float(np.nanmean(di)) if np.isfinite(di).any() else None,
        },
        "toml": _toml(eps, delta, n),
    }
    return out


def _toml(eps: dict, delta: float, n: int) -> str:
    return (
        "[design]\n"
        f"n_blocks = {n}\n\n"
        "[analysis]\n"
        f"eps_id_nll = {eps['id_nll']:.6g}\n"
        f"eps_id_aff_nll = {eps['id_aff_nll']:.6g}\n"
        f"eps_id_acc = {eps['id_acc']:.6g}\n"
        f"delta_G = {delta:.6g}\n"
    )
