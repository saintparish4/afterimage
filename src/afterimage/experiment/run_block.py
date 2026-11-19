"""Run one paired seed block end to end (docs/blueprint.md §6.4).

    base (0..T0) -> gate -> fork state
      for arm in C, I, N, S:  fork -> window (T0..T0+K) -> correction (..T0+K+H) -> savings probe
      hybrids (optional):     combine post-window states of C and I -> correction -> savings probe

Every stage is idempotent and resumable. A crash resumes from the latest
verified full state, and because batches are a pure function of the step,
the resumed history is bit-identical to an uninterrupted one.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file

from ..artifacts.env import capture_env
from ..artifacts.hashing import atomic_write_json, read_json
from ..config import Config
from ..data.schedule import KEY_ID, block_seed, derive_seed
from ..data.splits import DEV_SPLITS, Split, load_splits, read_split
from ..eval.metrics import predict, summarize
from ..eval.stages import stage_a, stage_b
from ..train import determinism
from ..train.gate import evaluate_gate
from ..train.plan import ArmSpec, arms_for, plan_batch, plan_relearn
from ..train.state import TrainState, capture, fork, load_state, recorded_digest, restore, save_state, verify
from ..train.trainer import Trainer, make_model
from .layout import STATUS_COMPLETE, STATUS_EXCLUDED, STATUS_FAILED, arm_dir, block_dir

WINDOW_EVAL = ("probe", "id_dev", "ood_dev")


class Diverged(RuntimeError):
    pass


@dataclass
class Context:
    cfg: Config
    device: torch.device
    pool: Split
    evals: dict[str, Split]
    log: Callable[[str], None] = print
    relearn_split: Split = field(init=False)

    def __post_init__(self) -> None:
        idd = self.evals["id_dev"]
        self.relearn_split = idd.take(np.flatnonzero(idd.affected), "id_dev_aff")


def load_context(cfg: Config, device: torch.device, log: Callable[[str], None] = print) -> Context:
    data = cfg.data_path()
    if not (data / "manifest.json").exists():
        raise FileNotFoundError(f"no dataset at {data}; run `afterimage data build --config {cfg.source}`")
    m = read_json(data / "manifest.json")
    if m["data_hash"] != cfg.data_hash:
        raise ValueError("dataset manifest does not match this config's data section")
    return Context(cfg, device, read_split(data / "pool.parquet", "pool"), load_splits(data, DEV_SPLITS), log)


def base_checkpoints(cfg: Config) -> list[int]:
    T0, n = cfg.design.T0, cfg.design.base_checkpoints
    pts = np.unique(np.round(np.geomspace(1, T0, n)).astype(int))
    return sorted(set(pts.tolist()) | {T0})


def arm_checkpoints(cfg: Config, hybrid: bool) -> dict[int, str]:
    d = cfg.design
    out = {} if hybrid else {d.T0 + int(round(f * d.K)): "window" for f in d.window_fracs}
    out.update({d.T0 + d.K + h: "horizon" for h in d.horizons})
    return dict(sorted(out.items()))


def _write_table(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), tmp)
    tmp.replace(path)


def _save_preds(preds: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    save_file({k: torch.from_numpy(v) for k, v in preds.items()}, str(tmp))
    tmp.replace(path)


class BlockRunner:
    def __init__(self, ctx: Context, b: int):
        self.ctx, self.cfg, self.b = ctx, ctx.cfg, b
        self.seed = block_seed(self.cfg.meta.master_seed, b)
        self.dir = block_dir(self.cfg, b)
        self.throughput: dict[str, float] = {}

    # --- helpers ----------------------------------------------------------------
    def log(self, msg: str) -> None:
        self.ctx.log(f"[block {self.b:03d}] {msg}")

    def _trainer(self, arm: str) -> Trainer:
        m = make_model(self.cfg, self.ctx.device)
        tr = Trainer(self.cfg, m, self.ctx.device, self.ctx.pool)
        tr.arm = arm
        return tr

    def _evaluate(self, tr: Trainer, arm: str, step: int, out: Path, splits: tuple[str, ...], save: tuple[str, ...]) -> dict:
        preds = {s: predict(tr.model, self.ctx.evals[s], self.ctx.device, self.cfg.eval.batch) for s in splits}
        rows, metrics = [], {}
        for s, lp in preds.items():
            r = summarize(lp, self.ctx.evals[s])
            metrics[s] = {x["subset"]: x for x in r}
            rows += [{"block": self.b, "arm": arm, "step": step, "split": s, **x} for x in r]
        _write_table(rows, out / "eval" / f"step_{step:08d}.parquet")
        if save:
            _save_preds({s: preds[s] for s in save}, out / "preds" / f"step_{step:08d}.safetensors")
        return metrics

    def _flush_logs(self, tr: Trainer, out: Path, step: int) -> None:
        _write_table(tr.logs.steps, out / "logs" / f"steps_{step:08d}.parquet")
        _write_table(tr.logs.dyn, out / "logs" / f"dyn_{step:08d}.parquet")
        tr.logs.clear()

    def _check_finite(self, tr: Trainer, loss: float | None, step: int) -> None:
        if loss is not None and not np.isfinite(loss):
            raise Diverged(f"non-finite loss at step {step}")

    def _train(self, tr: Trainer, kind: str, start: int, stop: int, out: Path,
               on_ckpt: Callable[[int], None], ckpts: set[int], label: str) -> None:
        d = self.cfg.design
        t0 = time.perf_counter()
        for t in range(start, stop):
            plan = plan_batch(kind, t, d.window, self.seed, self.cfg.optim.batch, self.ctx.pool.y_true, self.ctx.pool.y_false)
            self._check_finite(tr, tr.step(plan), t)
            s = t + 1
            if s in ckpts:
                if not all(torch.isfinite(p).all() for p in tr.model.parameters()):
                    raise Diverged(f"non-finite parameters at step {s}")
                self._flush_logs(tr, out, s)
                on_ckpt(s)
            elif s % d.resume_every == 0 and s < stop:
                self._flush_logs(tr, out, s)
                save_state(capture(tr.model, tr.opt, s), out / "resume")
        self._flush_logs(tr, out, stop)
        if stop > start:
            self.throughput[label] = (stop - start) / (time.perf_counter() - t0)

    def _resume_point(self, out: Path, candidates: list[str]) -> TrainState | None:
        best: TrainState | None = None
        for name in candidates:
            p = out / name
            if verify(p):
                st = load_state(p)
                if best is None or st.step > best.step:
                    best = st
        return best

    # --- base ------------------------------------------------------------------
    def run_base(self) -> tuple[str, dict]:
        cfg, d = self.cfg, self.cfg.design
        out = self.dir / "base"
        if verify(self.dir / "fork") and (self.dir / "gate.json").exists():
            return recorded_digest(self.dir / "fork"), read_json(self.dir / "gate.json")
        tr = self._trainer("base")
        # A verified fork without gate.json means a crash right after saving it.
        st = load_state(self.dir / "fork") if verify(self.dir / "fork") else self._resume_point(out, ["resume"])
        if st is not None:
            restore(st, tr.model, tr.opt)
            start = st.step
            self.log(f"base: resuming at step {start}")
        else:
            from ..model.transformer import init_weights

            init_weights(tr.model, derive_seed(self.seed, KEY_ID["init"]))
            start = 0
        ckpts = set(base_checkpoints(cfg))
        gate_metrics: dict = {}

        def on_ckpt(s: int) -> None:
            if s == d.T0:
                gate_metrics.update(self._evaluate(tr, "base", s, out, DEV_SPLITS, DEV_SPLITS))
            else:
                self._evaluate(tr, "base", s, out, WINDOW_EVAL, ("probe",))

        self.log(f"base: steps {start}..{d.T0}")
        self._train(tr, "C", start, d.T0, out, on_ckpt, ckpts, "base")
        if not gate_metrics:  # resumed exactly at T0
            gate_metrics.update(self._evaluate(tr, "base", d.T0, out, DEV_SPLITS, DEV_SPLITS))
        losses = self._base_losses()
        gate = evaluate_gate(gate_metrics, losses, cfg.gate).to_dict()
        digest = save_state(capture(tr.model, tr.opt, d.T0, {"block": self.b, "role": "fork"}), self.dir / "fork")
        atomic_write_json(self.dir / "gate.json", gate)
        return digest, gate

    def _base_losses(self) -> list[tuple[int, float]]:
        t = read_parts_steps(sorted((self.dir / "base" / "logs").glob("steps_*.parquet")))
        return list(zip(t["step"], t["loss"]))

    # --- arms -------------------------------------------------------------------
    def run_arm(self, arm: ArmSpec) -> dict:
        cfg, d = self.cfg, self.cfg.design
        out = arm_dir(cfg, self.b, arm.name)
        if (out / "done").exists():
            return read_json(out / "done")
        out.mkdir(parents=True, exist_ok=True)
        hybrid = arm.kind == "HYB"
        tr = self._trainer(arm.name)
        st = self._resume_point(out, ["final", "resume", "post_window"])
        if st is None:
            if hybrid:
                st = self._hybrid_start(arm, out)
            else:
                fork(self.dir / "fork", out / "start", {"arm": arm.name})
                st = load_state(out / "start")
                atomic_write_json(out / "stage_a.json", {"start_digest": st.digest(),
                                                          "parent_digest": st.meta["parent_digest"]})
        restore(st, tr.model, tr.opt)
        ckpts = arm_checkpoints(cfg, hybrid)

        def on_ckpt(s: int) -> None:
            if ckpts[s] == "horizon":
                self._evaluate(tr, arm.name, s, out, DEV_SPLITS, DEV_SPLITS)
                save_state(capture(tr.model, None, s, {"arm": arm.name}, with_rng=False), out / f"ckpt_{s:08d}")
            else:
                self._evaluate(tr, arm.name, s, out, WINDOW_EVAL, ("probe",))
            if s == d.T0 + d.K:
                save_state(capture(tr.model, tr.opt, s, {"arm": arm.name}), out / "post_window")

        # Hybrids start exactly at T0+K (horizon 0), where no training step fires a checkpoint.
        if st.step in ckpts and not (out / "eval" / f"step_{st.step:08d}.parquet").exists():
            on_ckpt(st.step)
        if st.step < d.end:
            self.log(f"arm {arm.name}: steps {st.step}..{d.end}")
            if st.step < d.T0 + d.K:
                self._train(tr, arm.kind, st.step, d.T0 + d.K, out, on_ckpt, set(ckpts), f"{arm.name}:window")
            self._train(tr, arm.kind, max(st.step, d.T0 + d.K), d.end, out, on_ckpt, set(ckpts), f"{arm.name}:correction")
            save_state(capture(tr.model, tr.opt, d.end, {"arm": arm.name}), out / "final")
        self._relearn(arm.name, out)
        summary = {"arm": arm.name, "status": "ok", "throughput": {k: v for k, v in self.throughput.items() if k.startswith(arm.name)}}
        atomic_write_json(out / "done", summary)
        return summary

    def _hybrid_start(self, arm: ArmSpec, out: Path) -> TrainState:
        w = load_state(arm_dir(self.cfg, self.b, arm.weights_from) / "post_window")
        o = load_state(arm_dir(self.cfg, self.b, arm.opt_from) / "post_window")
        st = TrainState(w.model, o.opt, w.step, w.rng_cpu, w.rng_cuda,
                        {"arm": arm.name, "weights_from": arm.weights_from, "opt_from": arm.opt_from})
        save_state(st, out / "start")
        return st

    def _relearn(self, arm: str, out: Path) -> None:
        cfg = self.cfg
        if (out / "relearn.parquet").exists():
            return
        tr = self._trainer(f"{arm}:relearn")
        restore(load_state(out / "final"), tr.model, tr.opt)
        sp = self.ctx.relearn_split
        rows = []

        def ev(r: int) -> None:
            if len(sp) == 0:
                return
            m = {x["subset"]: x for x in summarize(predict(tr.model, sp, self.ctx.device, cfg.eval.batch), sp, ("all",))}
            rows.append({"arm": arm, "r": r, "false_nll": m["all"]["false_nll"], "false_agree": m["all"]["false_agree"],
                         "true_nll": m["all"]["nll"]})

        for r in range(cfg.design.R):
            if r % cfg.design.relearn_eval_every == 0:
                ev(r)
            tr.step(plan_relearn(r, cfg, self.seed, self.ctx.pool.y_true, self.ctx.pool.y_false),
                    lr=cfg.optim.lr, record=False)
        ev(cfg.design.R)
        _write_table(rows, out / "relearn.parquet")

    # --- block ------------------------------------------------------------------
    def run(self) -> dict:
        cfg = self.cfg
        self.dir.mkdir(parents=True, exist_ok=True)
        bj = self.dir / "block.json"
        if bj.exists() and read_json(bj).get("status") in (STATUS_COMPLETE, STATUS_EXCLUDED, STATUS_FAILED):
            return read_json(bj)
        started = time.time()
        info = {"block": self.b, "seed": self.seed, "science_hash": cfg.science_hash, "data_hash": cfg.data_hash,
                "status": "running", "device": str(self.ctx.device), "started_unix": started,
                "env": capture_env(str(self.ctx.device)), "determinism": determinism.flags()}
        atomic_write_json(bj, info)
        fork_digest, gate = self.run_base()
        info["gate_passed"] = gate["passed"]
        if not gate["passed"] and cfg.gate.enforce:
            self.log("gate failed: block excluded before treatment")
            info |= {"status": STATUS_EXCLUDED, "finished_unix": time.time(), "throughput": self.throughput}
            atomic_write_json(bj, info)
            return info
        arm_status: dict[str, str] = {}
        for arm in arms_for(cfg):
            if arm.kind == "HYB" and any(arm_status.get(a) != "ok" for a in (arm.weights_from, arm.opt_from)):
                arm_status[arm.name] = "skipped"
                continue
            try:
                arm_status[arm.name] = self.run_arm(arm)["status"]
            except Diverged as e:
                self.log(f"arm {arm.name} diverged: {e}")
                atomic_write_json(arm_dir(cfg, self.b, arm.name) / "failed.json", {"reason": str(e)})
                arm_status[arm.name] = "failed"
        info["stages"] = self._stages(fork_digest)
        atomic_write_json(self.dir / "stages.json", info["stages"])
        failed = any(arm_status.get(a) != "ok" for a in cfg.design.arms)
        info |= {"status": STATUS_FAILED if failed else STATUS_COMPLETE, "arms": arm_status,
                 "finished_unix": time.time(), "throughput": self.throughput}
        atomic_write_json(bj, info)
        self.log(f"done: {info['status']}")
        return info

    def _stages(self, fork_digest: str) -> dict:
        cfg = self.cfg
        starts = {}
        for a in cfg.design.arms:
            p = arm_dir(cfg, self.b, a) / "stage_a.json"
            if p.exists():
                starts[a] = read_json(p)["start_digest"]
        end = cfg.design.T0 + cfg.design.K
        wm: dict[str, dict] = {}
        for a in cfg.design.arms:
            f = arm_dir(cfg, self.b, a) / "eval" / f"step_{end:08d}.parquet"
            if f.exists():
                t = pq.read_table(f).to_pylist()
                wm[a] = {r["subset"]: r for r in t if r["split"] == "id_dev"}
        return {"A": stage_a(fork_digest, starts), "B": stage_b(cfg, wm)}


def read_parts_steps(files: list[Path]) -> dict[str, list]:
    steps: dict[int, float] = {}
    for f in files:
        t = pq.read_table(f, columns=["step", "loss"]).to_pydict()
        for s, l in zip(t["step"], t["loss"]):
            steps[int(s)] = float(l)
    ks = sorted(steps)
    return {"step": ks, "loss": [steps[k] for k in ks]}


def run_block(cfg: Config, b: int, ctx: Context) -> dict:
    return BlockRunner(ctx, b).run()
