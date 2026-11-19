"""Per-block diagnostics report (milestone M9). Exploratory by construction.

Every I-vs-C quantity sits next to the same quantity for N-vs-C, because a
raw difference between forks means nothing until it is compared with what
an innocuous perturbation produces.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from safetensors import safe_open

from ..artifacts.hashing import atomic_write_json
from ..config import Config
from ..data.schedule import block_seed
from ..data.splits import load_splits
from ..eval.metrics import js, nll
from ..experiment.layout import arm_dir, block_dir
from ..train.plan import LRSchedule, arms_for, plan_batch
from ..train.state import load_state, verify
from ..train.trainer import make_model
from . import cka, probes, tracin


def _weights(cfg: Config, b: int, arm: str, step: int) -> dict | None:
    d = cfg.design
    ad = arm_dir(cfg, b, arm)
    p = ad / ("post_window" if step == d.T0 + d.K and not (ad / f"ckpt_{step:08d}").exists() else f"ckpt_{step:08d}")
    return load_state(p).model if verify(p) else None


def _preds(cfg: Config, b: int, arm: str, step: int, split: str) -> np.ndarray | None:
    f = arm_dir(cfg, b, arm) / "preds" / f"step_{step:08d}.safetensors"
    if not f.exists():
        return None
    with safe_open(str(f), framework="numpy") as h:
        return h.get_tensor(split) if split in h.keys() else None


def _rel_dist(a: dict, c: dict) -> float:
    num = sum(float(((a[k] - c[k]).double() ** 2).sum()) for k in c)
    den = sum(float((c[k].double() ** 2).sum()) for k in c)
    return float(np.sqrt(num / den))


def run_diagnostics(cfg: Config, b: int, device: torch.device, *, with_tracin: bool = True,
                    log: Callable[[str], None] = print, n_rows: int = 1024) -> dict:
    d = cfg.design
    H, W1 = d.T0 + d.K + d.H, d.T0 + d.K
    splits = load_splits(cfg.data_path(), ["probe", "ood_dev", "pool"])
    probe = splits["probe"].take(np.arange(min(n_rows, len(splits["probe"]))))
    ood = splits["ood_dev"].take(np.arange(min(2 * n_rows, len(splits["ood_dev"]))))
    pool = splits["pool"]
    arms = [a for a in ("C", "I", "N", "S") if a in d.arms]
    out: dict = {"block": b, "exploratory": True, "horizon_step": H}

    # 1. parameter distance relative to the control, against the nuisance reference
    pd = {}
    for h in d.horizons:
        s = W1 + h
        wc = _weights(cfg, b, "C", s)
        if wc is None:
            continue
        pd[str(h)] = {f"{a}-C": _rel_dist(w, wc) for a in arms[1:] if (w := _weights(cfg, b, a, s)) is not None}
    out["param_distance_rel"] = pd

    # 2. functional divergence on the fixed probes
    fj = {}
    for h in d.horizons:
        s = W1 + h
        pc = _preds(cfg, b, "C", s, "probe")
        if pc is not None:
            fj[str(h)] = {f"{a}-C": float(js(p, pc).mean()) for a in arms[1:] if (p := _preds(cfg, b, a, s, "probe")) is not None}
    out["probe_js"] = fj

    model = make_model(cfg, device)
    other = make_model(cfg, device)
    W = {a: _weights(cfg, b, a, H) for a in arms}
    if W.get("C") is None:
        raise FileNotFoundError(f"block {b}: arm C has no checkpoint at H")

    # 3. linear-interpolation barriers at H
    alphas = np.linspace(0, 1, 11)
    out["barrier"] = {}
    for a in ("I", "N"):
        if W.get(a) is not None:
            curve = probes.interpolation_curve(model, W[a], W["C"], ood, device, alphas)
            out["barrier"][f"{a}-C"] = {"barrier": probes.barrier(curve), "curve": curve}

    # 4. layerwise CKA at the EQ and X-token positions
    out["cka"] = {}
    for label, step in (("window_end", W1), ("H", H)):
        wc = _weights(cfg, b, "C", step)
        if wc is None:
            continue
        model.load_state_dict(wc)
        ref = {w: probes.activations(model, probe, device, w) for w in ("eq", "x")}
        res = {}
        for a in ("I", "N"):
            wa = _weights(cfg, b, a, step)
            if wa is None:
                continue
            other.load_state_dict(wa)
            for w in ("eq", "x"):
                acts = probes.activations(other, probe, device, w)
                res[f"{a}-C:{w}"] = [cka.linear_cka(x, y) for x, y in zip(acts, ref[w])]
        out["cka"][label] = res

    # 5. cross-fork activation patching at H (C's activations into I, and into N as reference)
    out["patching"] = {}
    model.load_state_dict(W["C"])
    for a in ("I", "N"):
        if W.get(a) is None:
            continue
        other.load_state_dict(W[a])
        out["patching"][f"{a}<-C"] = {
            where: [probes.patch_gap_closed(other, model, ood, device, layer, where)["fraction_closed"]
                    for layer in range(cfg.model.layers + 1)]
            for where in ("all", "eq")
        }

    # 6. gradient alignment on one fixed correction batch
    seed = block_seed(cfg.meta.master_seed, b)
    plan = plan_batch("C", W1, d.window, seed, cfg.optim.batch, pool.y_true, pool.y_false)
    L = int(pool.eq_pos[plan.idx].max()) + 1
    xb = torch.from_numpy(pool.tokens[plan.idx, :L].astype(np.int64)).to(device)
    eb = torch.from_numpy(pool.eq_pos[plan.idx].astype(np.int64)).to(device)
    yb = tracin.to_tokens(plan.y, device)
    gc = {}
    for h in d.horizons:
        s = W1 + h
        wc = _weights(cfg, b, "C", s)
        if wc is None:
            continue
        model.load_state_dict(wc)
        g0 = tracin.batch_grad(model, xb, eb, yb)
        row = {}
        for a in arms[1:]:
            wa = _weights(cfg, b, a, s)
            if wa is not None:
                other.load_state_dict(wa)
                ga = tracin.batch_grad(other, xb, eb, yb)
                row[f"{a}-C"] = float(torch.nn.functional.cosine_similarity(g0, ga, dim=0))
        gc[str(h)] = row
    out["grad_cosine"] = gc

    # 7. hybrid decomposition: which of weights / optimizer state carries the effect
    hyb = {}
    pc = _preds(cfg, b, "C", H, "ood_dev")
    for arm in arms_for(cfg):
        if arm.name == "C":
            continue
        p = _preds(cfg, b, arm.name, H, "ood_dev")
        if p is not None and pc is not None:
            sp = splits["ood_dev"]
            hyb[arm.name] = float(nll(p, sp.y_true).mean() - nll(pc, sp.y_true).mean())
    out["held_out_nll_minus_C_at_H"] = hyb

    # 8. first-order attribution of the window versus the exact fork effect
    if with_tracin and "I" in d.arms:
        out["tracin_window"] = _tracin_window(cfg, b, device, model, pool, splits["ood_dev"], seed)
    atomic_write_json(block_dir(cfg, b) / "diagnostics.json", out)
    log(f"[block {b:03d}] diagnostics written")
    return out


def _tracin_window(cfg: Config, b: int, device: torch.device, model, pool, ood_full, seed: int, n_test: int = 1024,
                   n_batches: int = 20) -> dict:
    d = cfg.design
    W1 = d.T0 + d.K
    fork_st = load_state(block_dir(cfg, b) / "fork")
    post = load_state(arm_dir(cfg, b, "C") / "post_window")
    ckpts = [(d.T0, fork_st.model, fork_st.opt), (W1, post.model, post.opt)]
    test = ood_full.take(np.arange(min(n_test, len(ood_full))))
    Lt = int(test.eq_pos.max()) + 1
    test_b = (torch.from_numpy(test.tokens[:, :Lt].astype(np.int64)).to(device),
              torch.from_numpy(test.eq_pos.astype(np.int64)).to(device), tracin.to_tokens(test.y_true, device))
    steps = np.unique(np.linspace(d.T0, W1 - 1, n_batches).astype(int))
    w = d.K / len(steps)
    batches = []
    for t in steps:
        plan = plan_batch("C", int(t), d.window, seed, cfg.optim.batch, pool.y_true, pool.y_false)
        L = int(pool.eq_pos[plan.idx].max()) + 1
        batches.append((int(t), torch.from_numpy(pool.tokens[plan.idx, :L].astype(np.int64)).to(device),
                        torch.from_numpy(pool.eq_pos[plan.idx].astype(np.int64)).to(device),
                        tracin.to_tokens(pool.y_true[plan.idx], device), tracin.to_tokens(pool.y_false[plan.idx], device), w))
    pred = tracin.first_order_window_effect(model, ckpts, batches, test_b, LRSchedule.from_config(cfg),
                                            tuple(cfg.optim.betas), cfg.optim.eps)
    pi, pc = _preds(cfg, b, "I", W1, "ood_dev"), _preds(cfg, b, "C", W1, "ood_dev")
    exact = None
    if pi is not None and pc is not None:
        idx = np.arange(len(test))
        exact = float(nll(pi[idx], test.y_true).mean() - nll(pc[idx], test.y_true).mean())
    return {"predicted_delta_nll": pred, "exact_delta_nll": exact,
            "ratio": (pred / exact) if exact not in (None, 0.0) else None,
            "note": "first-order, two checkpoints, Adam-preconditioned; exploratory"}
