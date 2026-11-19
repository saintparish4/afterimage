"""Answer predictions and per-example metrics.

A *prediction* for one example is the pair of full-vocabulary log-probabilities
``(log p(0), log p(1))`` at the EQ position. Everything else derives from it:

* NLL of the correct answer under the full softmax (the primary metric),
* the 2-way distribution renormalized over {0, 1} (Brier score, JS divergence),
* accuracy, with the prediction taken as the argmax over {0, 1}.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from ..data.splits import Split
from ..lang.grammar import ONE, ZERO

SubsetFn = Callable[[Split], np.ndarray]

SUBSETS: dict[str, SubsetFn] = {
    "all": lambda s: np.ones(len(s), dtype=bool),
    "aff": lambda s: s.affected,                    # the false rule changes the answer
    "inv": lambda s: ~s.affected,                   # both rules give the same answer
    "has_x": lambda s: s.has_x,
    "heldout_x": lambda s: s.heldout_x,             # G_X
    "heldout_nonx_only": lambda s: s.heldout_nonx & ~s.heldout_x,  # G_notX
}
for _d in range(1, 6):
    SUBSETS[f"depth{_d}"] = (lambda d: (lambda s: s.depth == d))(_d)


@torch.no_grad()
def predict(model: torch.nn.Module, split: Split, device: torch.device, batch: int = 4096) -> np.ndarray:
    was_training = model.training
    model.eval()
    out = np.empty((len(split), 2), dtype=np.float32)
    for i in range(0, len(split), batch):
        sl = slice(i, i + batch)
        L = int(split.eq_pos[sl].max()) + 1
        x = torch.from_numpy(split.tokens[sl, :L].astype(np.int64)).to(device)
        eq = torch.from_numpy(split.eq_pos[sl].astype(np.int64)).to(device)
        logp = torch.log_softmax(model.answer_logits(x, eq).float(), dim=-1)
        out[sl] = logp[:, [ZERO, ONE]].cpu().numpy()
    model.train(was_training)
    return out


def nll(lp: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -lp[np.arange(len(y)), y.astype(np.int64)].astype(np.float64)


def correct(lp: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (lp[:, 1] > lp[:, 0]).astype(np.uint8) == y


def p_one(lp: np.ndarray) -> np.ndarray:
    """2-way probability of answer 1: sigmoid(log p1 - log p0)."""
    z = lp[:, 1].astype(np.float64) - lp[:, 0].astype(np.float64)
    return 1.0 / (1.0 + np.exp(-z))


def brier(lp: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (p_one(lp) - y.astype(np.float64)) ** 2


def _h(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-300, 1.0)
    r = np.clip(1.0 - p, 1e-300, 1.0)
    return -(p * np.log(q) + (1.0 - p) * np.log(r))


def js(lp_a: np.ndarray, lp_b: np.ndarray) -> np.ndarray:
    """Per-example Jensen-Shannon divergence (nats) between 2-way answer distributions."""
    pa, pb = p_one(lp_a), p_one(lp_b)
    return _h(0.5 * (pa + pb)) - 0.5 * (_h(pa) + _h(pb))


def summarize(lp: np.ndarray, split: Split, subsets: tuple[str, ...] | None = None) -> list[dict]:
    """Aggregate metrics (true labels) per subset, plus false-label agreement."""
    rows = []
    for name in subsets or tuple(SUBSETS):
        m = SUBSETS[name](split)
        n = int(m.sum())
        if n == 0:
            continue
        yt, yf, l = split.y_true[m], split.y_false[m], lp[m]
        rows.append({
            "subset": name,
            "n": n,
            "nll": float(nll(l, yt).mean()),
            "acc": float(correct(l, yt).mean()),
            "brier": float(brier(l, yt).mean()),
            "false_agree": float(correct(l, yf).mean()),
            "false_nll": float(nll(l, yf).mean()),
        })
    return rows
