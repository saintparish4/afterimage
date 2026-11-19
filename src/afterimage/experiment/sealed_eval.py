"""Evaluate saved horizon checkpoints on the sealed confirmatory splits.

Training never sees the sealed sets: they may not even exist yet when the
runs happen. This pass loads each arm's weights at every horizon and writes
``preds_sealed/step_XXXXXXXX.safetensors``. Fork-state predictions (T0) are
included too, as the Stage A baseline.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from ..config import Config
from ..data.splits import SEALED_SPLITS, load_splits, sealed_available
from ..eval.metrics import predict
from ..train.plan import arms_for
from ..train.state import load_state, verify
from ..train.trainer import make_model
from .layout import analysis_blocks, arm_dir, block_dir
from .run_block import _save_preds


def evaluate_sealed(cfg: Config, device: torch.device, log: Callable[[str], None] = print) -> int:
    data = cfg.data_path()
    if not sealed_available(data):
        raise FileNotFoundError("sealed splits not built; run `afterimage data seal` first")
    splits = load_splits(data, SEALED_SPLITS)
    blocks, _ = analysis_blocks(cfg)
    model = make_model(cfg, device)
    n = 0
    d = cfg.design
    for b in blocks:
        targets = [(block_dir(cfg, b) / "fork", block_dir(cfg, b) / "base" / "preds_sealed", d.T0)]
        for arm in arms_for(cfg):
            ad = arm_dir(cfg, b, arm.name)
            for h in d.horizons:
                s = d.T0 + d.K + h
                targets.append((ad / f"ckpt_{s:08d}", ad / "preds_sealed", s))
        for ckpt, outdir, step in targets:
            out = outdir / f"step_{step:08d}.safetensors"
            if out.exists() or not verify(ckpt):
                continue
            model.load_state_dict(load_state(ckpt).model)
            _save_preds({k: predict(model, v, device, cfg.eval.batch) for k, v in splits.items()}, out)
            n += 1
        log(f"[block {b:03d}] sealed predictions written")
    return n
