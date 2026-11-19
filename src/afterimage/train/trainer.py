"""The deterministic training step.

``Trainer`` owns the model, the AdamW optimizer and the training pool on the
device. One call to ``step`` consumes one ``BatchPlan``. Training dynamics
(per-tensor gradient and update norms, Adam second-moment statistics) are
recorded at a configurable cadence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from ..config import Config
from ..data.splits import Split
from ..lang.grammar import ZERO
from ..model.transformer import Transformer, init_weights, param_groups
from .plan import BatchPlan, LRSchedule


def make_model(cfg: Config, device: torch.device, seed: int | None = None) -> Transformer:
    m = Transformer(cfg.model)
    if seed is not None:
        init_weights(m, seed)
    return m.to(device)


def make_optimizer(model: Transformer, cfg: Config) -> torch.optim.AdamW:
    o = cfg.optim
    return torch.optim.AdamW(param_groups(model, o.weight_decay), lr=o.lr, betas=tuple(o.betas), eps=o.eps,
                             foreach=False, fused=False)


@dataclass
class Logs:
    steps: list[dict] = field(default_factory=list)
    dyn: list[dict] = field(default_factory=list)

    def clear(self) -> None:
        self.steps.clear()
        self.dyn.clear()


class Trainer:
    def __init__(self, cfg: Config, model: Transformer, device: torch.device, pool: Split):
        self.cfg, self.model, self.device = cfg, model, device
        self.opt = make_optimizer(model, cfg)
        self.lr = LRSchedule.from_config(cfg)
        self.pool_tokens = torch.from_numpy(pool.tokens).to(device)
        self.pool_eq_np = pool.eq_pos.astype(np.int64)
        self.pool_eq = torch.from_numpy(self.pool_eq_np).to(device)
        self.names = [n for n, _ in model.named_parameters()]
        self.logs = Logs()
        self.arm = ""

    def step(self, plan: BatchPlan, lr: float | None = None, record: bool = True) -> float | None:
        """Apply one optimizer step. ``lr`` overrides the schedule (savings probe);
        ``record=False`` skips logging (steps outside the main history)."""
        cfg, ev = self.cfg, self.cfg.eval
        t = plan.step if record else -1
        lr = self.lr(plan.step) if lr is None else lr
        idx = torch.from_numpy(plan.idx).to(self.device)
        L = int(self.pool_eq_np[plan.idx].max()) + 1
        x = self.pool_tokens[idx, :L].long()
        eq = self.pool_eq[idx]
        y = torch.from_numpy(plan.y.astype(np.int64) + ZERO).to(self.device)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.model.train()
        loss = F.cross_entropy(self.model.answer_logits(x, eq), y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        clipped = False
        if cfg.optim.grad_clip > 0:
            total = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.optim.grad_clip)
            clipped = bool(total > cfg.optim.grad_clip)
        dyn = t >= 0 and t % ev.dyn_every == 0
        if dyn:
            before = [p.detach().clone() for p in self.model.parameters()]
            gnorm = [p.grad.norm().item() if p.grad is not None else 0.0 for p in self.model.parameters()]
        self.opt.step()
        out = None
        if t >= 0 and (t % ev.log_every == 0 or plan.in_window):
            out = loss.item()
            self.logs.steps.append({
                "arm": self.arm, "step": t, "loss": out, "lr": lr, "key": plan.key,
                "in_window": plan.in_window, "n_affected": plan.n_affected, "n_wrong": plan.n_wrong,
                "clipped": clipped,
            })
        if dyn:
            nu_now = t % ev.nu_every == 0
            for i, (name, p) in enumerate(zip(self.names, self.model.parameters())):
                row = {
                    "arm": self.arm, "step": t, "param": name,
                    "grad_norm": gnorm[i],
                    "update_norm": (p.detach() - before[i]).norm().item(),
                    "param_norm": p.detach().norm().item(),
                }
                if nu_now:
                    v = self.opt.state[p]["exp_avg_sq"].detach().flatten()
                    q = torch.quantile(v.float().cpu(), torch.tensor([0.5, 0.99]))
                    row |= {"nu_mean": v.mean().item(), "nu_p50": q[0].item(), "nu_p99": q[1].item(),
                            "nu_max": v.max().item()}
                self.logs.dyn.append(row)
        return out
