"""The pre-registered analysis (docs/blueprint.md §8-9) and outcome classification.

Everything that decides the result is defined here, before any confirmatory
data exist: the primary endpoint, the equivalence precondition, the
secondary family (Holm-corrected), the placebo validity check and the
category rules A1/A2/A/B/C/D.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import Config
from .endpoints import Cube
from .stats import PairedResult, crossed_bootstrap, holm, paired


@dataclass(frozen=True)
class Contrast:
    key: str
    a: str
    b: str
    role: str
    subset: str
    metric: str
    description: str


PRIMARY = Contrast("primary", "I", "C", "ood", "all", "nll", "held-out NLL, I - C")
ID_EQUIV = (
    Contrast("id_nll", "I", "C", "id", "all", "nll", "familiar-task NLL, I - C"),
    Contrast("id_aff_nll", "I", "C", "id", "aff", "nll", "familiar-task NLL on affected rows, I - C"),
    Contrast("id_acc", "I", "C", "id", "all", "acc", "familiar-task accuracy, I - C"),
)
PLACEBO = Contrast("placebo", "N", "C", "ood", "all", "nll", "held-out NLL, N - C (must be null)")
SECONDARY = (
    Contrast("S1_heldout_x", "I", "C", "ood", "heldout_x", "nll", "G_X: held-out compositions involving X"),
    Contrast("S2_heldout_nonx", "I", "C", "ood", "heldout_nonx_only", "nll", "G_notX: held-out compositions without X"),
    Contrast("S3_invariant", "I", "C", "ood", "inv", "nll", "G_inv: rows where both rules agree"),
    Contrast("S4_specificity", "I", "S", "ood", "all", "nll", "coherent rule vs matched noise, I - S"),
)


def margin_for(cfg: Config, key: str) -> float:
    a = cfg.analysis
    return {"id_nll": a.eps_id_nll, "id_aff_nll": a.eps_id_aff_nll, "id_acc": a.eps_id_acc,
            "primary": a.delta_G}.get(key, math.nan)


def _diff(cube: Cube, c: Contrast, step: int) -> np.ndarray:
    return cube.values(c.a, step, c.role, c.subset, c.metric) - cube.values(c.b, step, c.role, c.subset, c.metric)


def run_analysis(cube: Cube, *, determinism: dict | None, audit: dict | None) -> dict[str, Any]:
    cfg = cube.cfg
    a, d = cfg.analysis, cfg.design
    H = d.T0 + d.K + d.H
    kw = dict(alpha=a.alpha, n_boot=a.n_bootstrap, seed=a.analysis_seed)
    res: dict[str, Any] = {"mode": cube.mode, "blocks": cube.blocks, "n_target": d.n_blocks,
                           "science_hash": cfg.science_hash, "statuses": {str(k): v for k, v in cube.statuses.items()}}

    primary_d = _diff(cube, PRIMARY, H)
    primary = paired(primary_d, margin=a.delta_G, **kw)
    res["primary"] = {"contrast": PRIMARY.description, "per_block": primary_d.tolist(), **primary.to_dict()}

    res["id_equivalence"] = {}
    for c in ID_EQUIV:
        r = paired(_diff(cube, c, H), margin=margin_for(cfg, c.key), **kw)
        res["id_equivalence"][c.key] = {"contrast": c.description, **r.to_dict(), "equivalent": r.equivalent(a.alpha)}

    pl = paired(_diff(cube, PLACEBO, H), **kw)
    res["placebo"] = {"contrast": PLACEBO.description, **pl.to_dict()}

    sec: dict[str, PairedResult] = {c.key: paired(_diff(cube, c, H), **kw) for c in SECONDARY}
    kappa = cube.dispersion("I", "C", H) - cube.dispersion("N", "C", H)
    sec["S5_dispersion"] = paired(kappa, **kw)
    rho = cube.relearn_auc("C") - cube.relearn_auc("I")
    sec["S6_savings"] = paired(rho, **kw)
    adj = holm({k: v.p for k, v in sec.items()}, a.alpha)
    desc = {c.key: c.description for c in SECONDARY}
    desc["S5_dispersion"] = "kappa = JS(I,C) - JS(N,C) on held-out rows"
    desc["S6_savings"] = "rho = AUC_C - AUC_I of false-label NLL during re-exposure (>0: I re-adopts faster)"
    res["secondary"] = {k: {"contrast": desc[k], **v.to_dict(), "p_holm": adj[k]} for k, v in sec.items()}

    res["persistence"] = {}
    for h in d.horizons:
        r = paired(_diff(cube, PRIMARY, d.T0 + d.K + h), **kw)
        res["persistence"][str(h)] = {"estimate": r.estimate, "ci95": r.ci95, "n": r.n}

    res["sensitivity"] = _sensitivity(cube, H, primary, kw)
    res["descriptive"] = _descriptive(cube, H)
    res["validity"] = validity(cube, primary_d, pl, determinism, audit)
    res["category"] = classify(primary, [paired(_diff(cube, c, H), margin=margin_for(cfg, c.key), **kw) for c in ID_EQUIV],
                               res["validity"], a.alpha, a.delta_G)
    return res


def _sensitivity(cube: Cube, H: int, primary: PairedResult, kw: dict) -> dict:
    a = cube.cfg.analysis
    out: dict[str, Any] = {}
    b = paired(cube.values("I", H, "ood", "all", "brier") - cube.values("C", H, "ood", "all", "brier"), **kw)
    out["brier"] = {**b.to_dict(), "sign_agrees_with_primary": bool(np.sign(b.estimate) == np.sign(primary.estimate))}
    D = cube.example_matrix("I", "C", H, "ood", "nll")
    if D.ndim == 2 and D.shape[0] >= 2:
        out["crossed_bootstrap_ci95"] = crossed_bootstrap(D, n_boot=min(2000, a.n_bootstrap), seed=a.analysis_seed)
    pp = []
    for i, blk in enumerate(cube.blocks):
        sb = cube.stage(blk).get("B", {})
        pp.append(sb.get("ok") is not False)
    keep = np.array(pp, dtype=bool)
    d = cube.values("I", H, "ood", "all", "nll") - cube.values("C", H, "ood", "all", "nll")
    out["per_protocol_stage_b"] = {**paired(d[keep], **kw).to_dict(), "n_excluded": int((~keep).sum())}
    missing = ~np.isfinite(d)
    if missing.any() and np.isfinite(d).sum() >= 2:
        worst = d.copy()
        fill = -np.sign(np.nanmean(d)) * np.nanmax(np.abs(d))
        worst[missing] = fill
        out["worst_case_imputation"] = {**paired(worst, **kw).to_dict(), "n_imputed": int(missing.sum())}
    return out


def _descriptive(cube: Cube, H: int) -> dict:
    out: dict[str, Any] = {}
    for arm in ("C", "I", "N", "S"):
        out[f"ood_acc_{arm}"] = float(np.nanmean(cube.values(arm, H, "ood", "all", "acc")))
        out[f"id_acc_{arm}"] = float(np.nanmean(cube.values(arm, H, "id", "all", "acc")))
    by_depth = {}
    for dep in (2, 3, 4):
        diff = cube.values("I", H, "ood", f"depth{dep}", "nll") - cube.values("C", H, "ood", f"depth{dep}", "nll")
        by_depth[str(dep)] = float(np.nanmean(diff)) if np.isfinite(diff).any() else None
    out["ood_nll_I_minus_C_by_depth"] = by_depth
    end = cube.cfg.design.T0 + cube.cfg.design.K
    i_fa = cube.values("I", end, "id", "aff", "acc")
    out["stage_b_I_aff_true_acc_at_window_end"] = float(np.nanmean(i_fa)) if np.isfinite(i_fa).any() else None
    return out


def validity(cube: Cube, primary_d: np.ndarray, placebo: PairedResult, determinism: dict | None,
             audit: dict | None) -> dict:
    cfg = cube.cfg
    a = cfg.analysis
    checks: dict[str, dict] = {}
    n_blocks = len(cube.blocks)
    checks["enough_blocks"] = {"ok": n_blocks >= cfg.design.n_blocks, "value": n_blocks}
    hashes = {cube.block_info(b).get("science_hash") for b in cube.blocks}
    checks["single_design"] = {"ok": hashes == {cfg.science_hash}, "value": sorted(h or "?" for h in hashes)}
    stage_a = [cube.stage(b).get("A", {}).get("ok") for b in cube.blocks]
    checks["stage_a_identical_starts"] = {"ok": all(x is True for x in stage_a), "value": stage_a.count(True)}
    failed = sum(cube.statuses.get(b) == "postfork_failed" for b in cube.blocks)
    frac = failed / n_blocks if n_blocks else 1.0
    checks["postfork_failures"] = {"ok": frac <= a.max_postfork_fail_frac, "value": failed}
    p = placebo.p
    checks["placebo_null"] = {"ok": not (np.isfinite(p) and p < a.placebo_invalid_p), "value": p}
    if determinism is None:
        checks["determinism_replay"] = {"ok": cube.mode != "sealed", "value": "not run",
                                        "note": "required for confirmatory analysis"}
    else:
        checks["determinism_replay"] = {"ok": bool(determinism.get("passed")), "value": determinism.get("device")}
    if audit is not None:
        bad = [k for k, (ok, _) in audit.items() if not ok]
        checks["leakage_audit"] = {"ok": not bad, "value": bad}
    return {"ok": all(c["ok"] for c in checks.values()), "checks": checks}


def classify(primary: PairedResult, id_equiv: list[PairedResult], valid: dict, alpha: float, delta_G: float) -> dict:
    """Pre-registered decision rules (blueprint §9.6)."""
    reasons: list[str] = []
    if not valid["ok"]:
        reasons = [k for k, c in valid["checks"].items() if not c["ok"]]
        return {"category": "D", "label": "invalid", "reasons": reasons}
    if not all(np.isfinite(r.p_tost) for r in id_equiv) or not np.isfinite(delta_G):
        return {"category": "uncalibrated", "label": "margins not set; no category can be assigned",
                "reasons": ["equivalence margins are NaN (pilot calibrates them)"]}
    if not all(r.equivalent(alpha) for r in id_equiv):
        return {"category": "D", "label": "correction incomplete at H",
                "reasons": ["familiar-task equivalence not established"]}
    if primary.excludes_zero:
        lo90, hi90 = primary.ci90
        if abs(primary.estimate) >= delta_G:
            return {"category": "A1", "label": "persistent difference detected, practically meaningful", "reasons": []}
        if -delta_G < lo90 and hi90 < delta_G:
            return {"category": "A2", "label": "persistent difference detected, below the smallest effect of interest", "reasons": []}
        return {"category": "A", "label": "persistent difference detected; practical size undetermined", "reasons": []}
    if primary.equivalent(alpha):
        return {"category": "B", "label": "practical equivalence", "reasons": []}
    return {"category": "C", "label": "inconclusive", "reasons": []}
