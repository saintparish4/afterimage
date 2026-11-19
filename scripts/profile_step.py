"""Profile one training step on the configured device (throughput diagnostics).

Usage: uv run python scripts/profile_step.py configs/pilot.toml [device]
Reports per-phase wall time with and without deterministic algorithms and
with the for-loop vs foreach AdamW implementation. Results are printed only;
nothing in a study directory is touched.
"""

import dataclasses
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    from afterimage.config import load_config
    from afterimage.data.schedule import block_seed
    from afterimage.data.splits import read_split
    from afterimage.model.transformer import param_groups
    from afterimage.train import determinism
    from afterimage.train.plan import plan_batch
    from afterimage.train.trainer import Trainer, make_model

    cfg = load_config(sys.argv[1])
    dev = torch.device(sys.argv[2] if len(sys.argv) > 2 else cfg.runtime.device)
    pool = read_split(cfg.data_path() / "pool.parquet")
    seed = block_seed(cfg.meta.master_seed, 0)
    B = cfg.optim.batch
    for det in (True, False):
        torch.use_deterministic_algorithms(False)
        determinism.configure(dataclasses.replace(cfg.runtime, deterministic=det))
        for foreach in (False, True):
            tr = Trainer(cfg, make_model(cfg, dev, seed=0), dev, pool)
            tr.opt = torch.optim.AdamW(param_groups(tr.model, 0.01), lr=1e-3, foreach=foreach)
            plans = [plan_batch("C", t, cfg.design.window, seed, B, pool.y_true, pool.y_false) for t in range(60)]
            for p in plans[:10]:
                tr.step(p, record=False)
            sync(dev)
            n = 50
            t0 = time.perf_counter()
            for p in plans[10:]:
                tr.step(p, record=False)
            sync(dev)
            full = (time.perf_counter() - t0) / n
            p = plans[0]
            x = tr.pool_tokens[torch.from_numpy(p.idx).to(dev), :34].long()
            eq = tr.pool_eq[torch.from_numpy(p.idx).to(dev)]
            y = torch.from_numpy(p.y.astype(np.int64) + 3).to(dev)
            sync(dev)
            t0 = time.perf_counter()
            for _ in range(n):
                loss = F.cross_entropy(tr.model.answer_logits(x, eq), y)
                tr.opt.zero_grad()
                loss.backward()
            sync(dev)
            fb = (time.perf_counter() - t0) / n
            t0 = time.perf_counter()
            for _ in range(n):
                tr.opt.step()
            sync(dev)
            op = (time.perf_counter() - t0) / n
            t0 = time.perf_counter()
            for t in range(n):
                plan_batch("C", t, cfg.design.window, seed, B, pool.y_true, pool.y_false)
            pl = (time.perf_counter() - t0) / n
            print(f"deterministic={det!s:5} foreach={foreach!s:5} step={full*1e3:6.2f}ms "
                  f"fwd+bwd={fb*1e3:6.2f}ms opt={op*1e3:6.2f}ms plan={pl*1e3:6.2f}ms", flush=True)


if __name__ == "__main__":
    main()
