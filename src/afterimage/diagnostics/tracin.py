"""TracIn-style first-order attribution with an Adam preconditioner.

TracIn-CP (Pruthi et al. 2020) scores a training example z against a test
loss as  sum_c  eta_c <g_c(z), g_c(test)>,  which assumes plain SGD. Under
Adam the step is (approximately) preconditioned by 1 / (sqrt(v_hat) + eps),
so this module uses  sum_c eta_c <P_c g_c(z), g_c(test)>. This is still a
first-order, sparse-checkpoint approximation: it ignores momentum, the
nonlinearity of Adam's normalization, and curvature between checkpoints.

``first_order_window_effect`` predicts the change in held-out loss caused by
swapping the window labels from true to false. The fork design measures that
change *exactly*, so comparing the two shows how far first-order
attribution can be trusted in this setting.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call, grad, vmap

from ..lang.grammar import ZERO
from ..model.transformer import Transformer, param_groups

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (tokens, eq_pos, answer token ids)


class _AnswerHead(nn.Module):
    """Wraps a Transformer so that forward(x, eq) returns answer logits."""

    def __init__(self, m: Transformer):
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor, eq: torch.Tensor) -> torch.Tensor:
        return self.m.answer_logits(x, eq)


def preconditioner(model: Transformer, opt_state: dict, betas: tuple[float, float], eps: float) -> torch.Tensor:
    """Flattened 1 / (sqrt(v_hat) + eps), in ``model.parameters()`` order.

    AdamW state indices follow the optimizer's param-group order (decayed
    weights first), not ``named_parameters`` order, so map through it.
    """
    order = [p for g in param_groups(model, 0.0) for p in g["params"]]
    index_of = {id(p): i for i, p in enumerate(order)}
    parts = []
    for p in model.parameters():
        st = opt_state["state"][index_of[id(p)]]
        step = float(st["step"])
        v_hat = st["exp_avg_sq"].to(p.device).float() / (1 - betas[1] ** step)
        parts.append((1.0 / (v_hat.sqrt() + eps)).reshape(-1))
    return torch.cat(parts)


def batch_grad(model: Transformer, x: torch.Tensor, eq: torch.Tensor, y_tok: torch.Tensor) -> torch.Tensor:
    """Flattened gradient of the mean answer loss, in ``model.parameters()`` order."""
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model.answer_logits(x, eq), y_tok).backward()
    g = torch.cat([p.grad.reshape(-1) for p in model.parameters()]).detach()
    model.zero_grad(set_to_none=True)
    return g


def per_example_grads(model: Transformer, x: torch.Tensor, eq: torch.Tensor, y_tok: torch.Tensor) -> torch.Tensor:
    """(B, P) per-example gradients via torch.func, in ``model.parameters()`` order."""
    head = _AnswerHead(model)
    params = {k: v.detach() for k, v in head.named_parameters()}

    def loss_fn(p: dict, xi: torch.Tensor, ei: torch.Tensor, yi: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(functional_call(head, p, (xi[None], ei[None])), yi[None])

    g = vmap(grad(loss_fn), in_dims=(None, 0, 0, 0))(params, x, eq, y_tok)
    return torch.cat([g[n].reshape(x.shape[0], -1) for n in params], dim=1)


def tracin_scores(model: Transformer, checkpoints: list[tuple[dict, dict, float]], train: Batch, test: Batch,
                  betas: tuple[float, float], eps: float) -> np.ndarray:
    """Score each training example against the mean test loss.

    ``checkpoints`` is [(model_state_dict, optimizer_state_dict, lr)]. A
    positive score means a step on that example lowers the test loss.
    """
    total = torch.zeros(train[0].shape[0], dtype=torch.float64)
    for sd, osd, lr in checkpoints:
        model.load_state_dict(sd)
        P = preconditioner(model, osd, betas, eps)
        g_test = batch_grad(model, *test)
        G = per_example_grads(model, *train)
        total += (lr * ((G * P[None]) @ g_test)).double().cpu()
    return total.numpy()


def first_order_window_effect(model: Transformer, checkpoints: list[tuple[int, dict, dict]],
                              window_batches: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]],
                              test: Batch, lr_at: Callable[[int], float], betas: tuple[float, float], eps: float) -> float:
    """Predicted change in mean test loss from training the window on false instead of true labels.

    ``checkpoints``: [(step, model_sd, opt_sd)]; each batch uses the nearest one.
    ``window_batches``: [(step, x, eq, y_true_tok, y_false_tok, weight)], where
    ``weight`` is the number of window steps the sampled batch stands for.
    """
    steps = np.array([c[0] for c in checkpoints])
    cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    pred = 0.0
    for step, x, eq, yt, yf, w in window_batches:
        ci = int(np.argmin(np.abs(steps - step)))
        model.load_state_dict(checkpoints[ci][1])
        if ci not in cache:
            cache[ci] = (preconditioner(model, checkpoints[ci][2], betas, eps), batch_grad(model, *test))
        P, g_test = cache[ci]
        dg = batch_grad(model, x, eq, yf) - batch_grad(model, x, eq, yt)
        pred += -w * lr_at(step) * float((P * dg) @ g_test)
    return pred


def to_tokens(y01: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(y01).astype(np.int64) + ZERO).to(device)
