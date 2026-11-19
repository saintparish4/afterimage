"""Hardware determinism check (milestone M3, blueprint §6.5).

On the device the study will use, this check verifies two things:

1. replay: two runs of arm C from the same state give a bit-identical state;
2. resume: stopping mid-run, saving, reloading into a fresh process-state
   model and continuing gives the same bits as running straight through.

The result goes to ``runs/<name>/determinism.json``. Confirmatory analysis
refuses to run unless it passed on the confirmatory device type.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from ..artifacts.env import capture_env
from ..artifacts.hashing import atomic_write_json
from ..data.schedule import KEY_ID, block_seed, derive_seed
from ..model.transformer import init_weights
from ..train import determinism
from ..train.plan import plan_batch
from ..train.state import capture, load_state, restore, save_state
from ..train.trainer import Trainer, make_model
from .layout import study_dir
from .run_block import Context


def replay_from_released(ctx: Context, released_study: Path, blocks: list[int]) -> dict:
    """Replication tier R1: re-run every arm from the *released* fork states and
    compare final-state digests with the released ones, bit for bit.

    ``ctx.cfg`` must point ``runs_dir`` somewhere fresh: the released fork and
    gate are copied in, so the base phase is skipped and only the forks rerun.
    """
    import shutil

    from ..train.plan import arms_for
    from ..train.state import recorded_digest, verify
    from .layout import arm_dir, block_dir
    from .run_block import BlockRunner

    cfg = ctx.cfg
    out: dict = {"blocks": {}, "passed": True}
    for b in blocks:
        src = released_study / f"block_{b:03d}"
        if not verify(src / "fork"):
            raise FileNotFoundError(f"released fork state missing or corrupt: {src / 'fork'}")
        dst = block_dir(cfg, b)
        if dst.exists():
            raise FileExistsError(f"{dst} exists; R1 replays into an empty study directory")
        dst.mkdir(parents=True)
        shutil.copytree(src / "fork", dst / "fork")
        shutil.copy2(src / "gate.json", dst / "gate.json")
        BlockRunner(ctx, b).run()
        arms = {}
        for arm in arms_for(cfg):
            rel = src / "arms" / arm.name / "final"
            if verify(rel):
                mine = recorded_digest(arm_dir(cfg, b, arm.name) / "final")
                arms[arm.name] = {"released": recorded_digest(rel), "replayed": mine,
                                  "identical": mine == recorded_digest(rel)}
        ok = bool(arms) and all(a["identical"] for a in arms.values())
        out["blocks"][str(b)] = {"arms": arms, "identical": ok}
        out["passed"] &= ok
    out["env"] = capture_env(str(ctx.device))
    atomic_write_json(study_dir(cfg) / "replication_r1.json", out)
    return out


def check_determinism(ctx: Context, steps: int = 500) -> dict:
    cfg = ctx.cfg
    seed = block_seed(cfg.meta.master_seed, 0)
    d = cfg.design

    def fresh() -> Trainer:
        m = make_model(cfg, ctx.device)
        init_weights(m, derive_seed(seed, KEY_ID["init"]))
        return Trainer(cfg, m, ctx.device, ctx.pool)

    def run(tr: Trainer, a: int, b: int, kind: str = "C") -> None:
        for t in range(a, b):
            tr.step(plan_batch(kind, t, d.window, seed, cfg.optim.batch, ctx.pool.y_true, ctx.pool.y_false), record=False)

    t0 = time.perf_counter()
    tr = fresh()
    run(tr, 0, steps)
    first = capture(tr.model, tr.opt, steps).digest()
    elapsed = time.perf_counter() - t0
    tr = fresh()
    run(tr, 0, steps)
    second = capture(tr.model, tr.opt, steps).digest()

    mid = steps // 2
    tr = fresh()
    run(tr, 0, mid)
    with tempfile.TemporaryDirectory() as tmp:
        save_state(capture(tr.model, tr.opt, mid), Path(tmp) / "mid")
        tr = fresh()
        restore(load_state(Path(tmp) / "mid"), tr.model, tr.opt)
    run(tr, mid, steps)
    resumed = capture(tr.model, tr.opt, steps).digest()

    result = {
        "device": str(ctx.device),
        "steps": steps,
        "replay_identical": first == second,
        "resume_identical": first == resumed,
        "passed": first == second == resumed,
        "digests": {"first": first, "second": second, "resumed": resumed},
        "steps_per_second": steps / elapsed,
        "env": capture_env(str(ctx.device)),
        "determinism": determinism.flags(),
        "science_hash": cfg.science_hash,
    }
    atomic_write_json(study_dir(cfg) / "determinism.json", result)
    return result
