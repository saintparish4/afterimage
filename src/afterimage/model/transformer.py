"""A small pre-LN decoder-only Transformer (docs/blueprint.md §3, §5).

Attention is written out explicitly instead of calling fused kernels, so the
backward pass is deterministic on every backend. The output projection is
tied to the token embedding through ``F.linear`` (no second parameter), which
keeps the state dict free of shared tensors for safetensors.

Residual streams can be captured (``capture=True``) and replaced
(``patch={layer: fn}``) for the diagnostics in §10. Layer 0 is the embedding
output; layer ``l`` is the output of block ``l``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ModelConfig
from ..lang.grammar import VOCAB_SIZE

Patch = dict[int, Callable[[Tensor], Tensor]]


class Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, T, self.heads, self.dh).transpose(1, 2)
        k = k.view(B, T, self.heads, self.dh).transpose(1, 2)
        v = v.view(B, T, self.heads, self.dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        att = att.masked_fill(mask[:T, :T], float("-inf"))
        y = att.softmax(dim=-1) @ v
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, eps=cfg.ln_eps)
        self.attn = Attention(cfg.d_model, cfg.heads)
        self.ln2 = nn.LayerNorm(cfg.d_model, eps=cfg.ln_eps)
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x), mask)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


@dataclass
class Output:
    logits: Tensor
    hidden: list[Tensor] | None = None


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(VOCAB_SIZE, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.ctx, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=cfg.ln_eps)
        mask = torch.triu(torch.ones(cfg.ctx, cfg.ctx, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def residual(self, tokens: Tensor, *, capture: bool = False, patch: Patch | None = None) -> tuple[Tensor, list[Tensor] | None]:
        T = tokens.shape[1]
        pos = torch.arange(T, device=tokens.device)
        h = self.tok_emb(tokens) + self.pos_emb(pos)[None]
        hidden: list[Tensor] | None = [] if capture else None
        if patch and 0 in patch:
            h = patch[0](h)
        if hidden is not None:
            hidden.append(h)
        for i, blk in enumerate(self.blocks, start=1):
            h = blk(h, self.mask)
            if patch and i in patch:
                h = patch[i](h)
            if hidden is not None:
                hidden.append(h)
        return h, hidden

    def head(self, h: Tensor) -> Tensor:
        return F.linear(self.ln_f(h), self.tok_emb.weight)

    def forward(self, tokens: Tensor, *, capture: bool = False, patch: Patch | None = None) -> Output:
        h, hidden = self.residual(tokens, capture=capture, patch=patch)
        return Output(self.head(h), hidden)

    def answer_logits(self, tokens: Tensor, eq_pos: Tensor, *, patch: Patch | None = None) -> Tensor:
        """Logits at the EQ position only: (B, vocab)."""
        h, _ = self.residual(tokens, patch=patch)
        rows = torch.arange(tokens.shape[0], device=tokens.device)
        return self.head(h[rows, eq_pos])


def init_weights(model: Transformer, seed: int) -> None:
    """GPT-2 style init from a CPU generator, so it is identical on every device."""
    g = torch.Generator().manual_seed(seed)
    std, L = 0.02, model.cfg.layers
    with torch.no_grad():
        for name, p in model.named_parameters():
            target = torch.empty(p.shape, dtype=p.dtype)
            if name.endswith("ln1.weight") or name.endswith("ln2.weight") or name == "ln_f.weight":
                target.fill_(1.0)
            elif name.endswith(".bias"):
                target.zero_()
            elif name.endswith("attn.proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(target, 0.0, std / math.sqrt(2 * L), generator=g)
            else:
                nn.init.normal_(target, 0.0, std, generator=g)
            p.copy_(target.to(p.device))


def param_groups(model: Transformer, weight_decay: float) -> list[dict]:
    """Decay 2-D weight matrices except embeddings; never biases or LayerNorm."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.ndim == 2 and not name.startswith(("tok_emb", "pos_emb")):
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
