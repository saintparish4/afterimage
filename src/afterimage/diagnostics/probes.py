"""Activation extraction, interpolation barriers and cross-fork activation patching."""

from __future__ import annotations

import numpy as np
import torch

from ..data.splits import Split
from ..eval.metrics import nll, predict
from ..lang.grammar import Op
from ..model.transformer import Transformer


def _batches(sp: Split, batch: int):
    for i in range(0, len(sp), batch):
        sl = slice(i, i + batch)
        L = int(sp.eq_pos[sl].max()) + 1
        yield sl, L


@torch.no_grad()
def activations(model: Transformer, sp: Split, device: torch.device, where: str, batch: int = 1024) -> list[np.ndarray]:
    """Residual stream per layer (0 = embeddings) at the EQ position or at every X token.

    Returns one (rows, d_model) array per layer. For ``where="x"`` the rows are
    all X-token positions in ``sp``, in a fixed order.
    """
    model.eval()
    per_layer: list[list[np.ndarray]] | None = None
    for sl, L in _batches(sp, batch):
        x = torch.from_numpy(sp.tokens[sl, :L].astype(np.int64)).to(device)
        out = model(x, capture=True)
        if per_layer is None:
            per_layer = [[] for _ in out.hidden]
        if where == "eq":
            rows = torch.arange(x.shape[0], device=device)
            eq = torch.from_numpy(sp.eq_pos[sl].astype(np.int64)).to(device)
            for i, h in enumerate(out.hidden):
                per_layer[i].append(h[rows, eq].cpu().numpy())
        elif where == "x":
            mask = x == int(Op.X)
            for i, h in enumerate(out.hidden):
                per_layer[i].append(h[mask].cpu().numpy())
        else:
            raise ValueError(where)
    return [np.concatenate(p) for p in (per_layer or [])]


def interpolation_curve(model: Transformer, sd_a: dict, sd_b: dict, sp: Split, device: torch.device,
                        alphas: np.ndarray) -> list[tuple[float, float]]:
    """Mean NLL along the straight line (1 - alpha) * a + alpha * b."""
    out = []
    for a in alphas:
        model.load_state_dict({k: (1 - a) * sd_a[k] + a * sd_b[k] for k in sd_a})
        out.append((float(a), float(nll(predict(model, sp, device), sp.y_true).mean())))
    return out


def barrier(curve: list[tuple[float, float]]) -> float:
    a = np.array([c[0] for c in curve])
    y = np.array([c[1] for c in curve])
    line = (1 - a) * y[0] + a * y[-1]
    return float(np.max(y - line))


@torch.no_grad()
def patch_gap_closed(target: Transformer, source: Transformer, sp: Split, device: torch.device, layer: int,
                     where: str = "all", batch: int = 1024) -> dict:
    """Replace ``target``'s residual stream at ``layer`` with ``source``'s, and measure
    how much of the target-minus-source NLL gap closes.

    ``where="all"`` patches every position, ``"eq"`` only the EQ position.
    A fraction of 1 means the gap is entirely mediated through that site.
    """
    target.eval()
    source.eval()
    tot = {"target": 0.0, "source": 0.0, "patched": 0.0}
    n = 0
    for sl, L in _batches(sp, batch):
        x = torch.from_numpy(sp.tokens[sl, :L].astype(np.int64)).to(device)
        eq = torch.from_numpy(sp.eq_pos[sl].astype(np.int64)).to(device)
        y = sp.y_true[sl]
        src_h = source(x, capture=True).hidden[layer]

        if where == "all":
            fn = lambda h: src_h  # noqa: E731
        elif where == "eq":
            rows = torch.arange(x.shape[0], device=device)

            def fn(h: torch.Tensor) -> torch.Tensor:
                h = h.clone()
                h[rows, eq] = src_h[rows, eq]
                return h
        else:
            raise ValueError(where)

        def lp(m: Transformer, patch=None) -> np.ndarray:
            logp = torch.log_softmax(m.answer_logits(x, eq, patch=patch).float(), -1)
            return logp[:, [3, 4]].cpu().numpy()

        tot["target"] += nll(lp(target), y).sum()
        tot["source"] += nll(lp(source), y).sum()
        tot["patched"] += nll(lp(target, {layer: fn}), y).sum()
        n += len(y)
    m = {k: v / n for k, v in tot.items()}
    gap = m["target"] - m["source"]
    m["fraction_closed"] = float((m["target"] - m["patched"]) / gap) if abs(gap) > 1e-12 else float("nan")
    return m
