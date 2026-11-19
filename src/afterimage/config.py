"""Typed experiment configuration loaded from TOML.

The scientific configuration (everything except ``[runtime]`` and
``[paths]``) is hashed into ``science_hash``. Runs, checkpoints and analysis
outputs record that hash, so results produced under different designs can
never be mixed silently.
"""

from __future__ import annotations

import dataclasses
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts.hashing import json_digest
from .lang.grammar import Op, Triple, parse_triple

PHASES = ("smoke", "pilot", "confirmatory")
ARM_KINDS = ("C", "I", "N", "S")


@dataclass(frozen=True)
class MetaConfig:
    name: str
    phase: str
    master_seed: int
    data_seed: int
    prereg_commit: str = ""


@dataclass(frozen=True)
class PathsConfig:
    data_dir: str = "data/generated"
    runs_dir: str = "runs"


@dataclass(frozen=True)
class DataConfig:
    pool_size: int
    train_depths: tuple[int, ...] = (1, 2, 3, 4)
    depth_weights: tuple[float, ...] | None = None
    id_eval_depths: tuple[int, ...] = (3, 4)
    ood_depths: tuple[int, ...] = (2, 3, 4)
    depth5: int = 5
    heldout_x: tuple[tuple[str, int, str], ...] = ()
    heldout_nonx: tuple[tuple[str, int, str], ...] = ()
    heldout_reserve: tuple[tuple[str, int, str], ...] = ()
    ctx: int = 40
    n_id_dev: int = 20_000
    n_ood_dev: int = 20_000
    n_id_conf: int = 20_000
    n_ood_conf: int = 20_000
    n_depth5: int = 10_000
    n_probe_id: int = 2048
    n_probe_ood: int = 2048
    chunk_size: int = 50_000
    workers: int = 0  # 0 = os.cpu_count(); never affects the output

    @property
    def max_expr_len(self) -> int:
        """Longest postfix expression at the training depths (a full binary tree)."""
        return 2 ** (max(self.train_depths) + 1) - 1

    @property
    def max_row_len(self) -> int:
        return self.max_expr_len + 2  # BOS ... EQ

    def heldout_triples(self) -> tuple[frozenset[Triple], frozenset[Triple]]:
        hx = frozenset(parse_triple(t) for t in self.heldout_x)
        hn = frozenset(parse_triple(t) for t in self.heldout_nonx)
        return hx, hn


@dataclass(frozen=True)
class ModelConfig:
    layers: int = 4
    d_model: int = 128
    heads: int = 4
    d_ff: int = 512
    ctx: int = 40
    ln_eps: float = 1e-5


@dataclass(frozen=True)
class OptimConfig:
    lr: float = 5e-4
    warmup: int = 1000
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-8
    weight_decay: float = 0.01
    batch: int = 256
    final_decay_frac: float = 0.2
    final_lr_frac: float = 0.1
    grad_clip: float = 0.0  # 0 disables clipping


@dataclass(frozen=True)
class DesignConfig:
    n_blocks: int
    T0: int
    K: int
    H: int
    horizons: tuple[int, ...]
    base_checkpoints: int = 20
    window_fracs: tuple[float, ...] = (0.25, 0.5)
    R: int = 300
    relearn_eval_every: int = 10
    arms: tuple[str, ...] = ("C", "I", "N", "S")
    hybrids: bool = False
    resume_every: int = 2000

    @property
    def window(self) -> range:
        return range(self.T0, self.T0 + self.K)

    @property
    def end(self) -> int:
        return self.T0 + self.K + self.H


@dataclass(frozen=True)
class GateConfig:
    enforce: bool = True
    id_small_min_depth_acc: float = 0.99
    id_min_acc: float = 0.995
    id_min_depth_acc: float = 0.99
    id_x_min_acc: float = 0.995
    ood_acc_low: float = 0.60
    ood_acc_high: float = 0.97
    loss_trend_window: int = 2000
    spike_factor: float = 3.0


@dataclass(frozen=True)
class StageBConfig:
    false_agree_min: float = 0.95
    aff_gap_min: float = 0.90


@dataclass(frozen=True)
class EvalConfig:
    batch: int = 4096
    log_every: int = 10
    dyn_every: int = 50
    nu_every: int = 500


@dataclass(frozen=True)
class AnalysisConfig:
    alpha: float = 0.05
    eps_id_nll: float = math.nan
    eps_id_aff_nll: float = math.nan
    eps_id_acc: float = 0.002
    delta_G: float = math.nan
    placebo_invalid_p: float = 0.01
    n_bootstrap: int = 100_000
    analysis_seed: int = 0
    max_postfork_fail_frac: float = 0.10
    eps_quantile: float = 0.90
    delta_sd_frac: float = 0.5
    power: float = 0.80
    n_min: int = 16
    n_max: int = 128


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cpu"
    threads: int = 0  # 0 = leave torch default; set >0 for bitwise CPU replay
    deterministic: bool = True


@dataclass(frozen=True)
class Config:
    meta: MetaConfig
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig
    design: DesignConfig
    gate: GateConfig = field(default_factory=GateConfig)
    stage_b: StageBConfig = field(default_factory=StageBConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    source: str | None = None  # path the config was loaded from (not hashed)

    # --- derived identities -------------------------------------------------
    def science_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        for k in ("runtime", "paths", "source"):
            d.pop(k)
        d["meta"].pop("prereg_commit")
        return d

    @property
    def science_hash(self) -> str:
        return json_digest(_nan_safe(self.science_dict()))

    @property
    def data_hash(self) -> str:
        return json_digest(_nan_safe({"data": dataclasses.asdict(self.data), "seed": self.meta.data_seed}))[:16]

    def data_path(self, root: Path | None = None) -> Path:
        base = Path(self.paths.data_dir)
        if root is not None and not base.is_absolute():
            base = root / base
        return base / self.data_hash

    def runs_path(self, root: Path | None = None) -> Path:
        base = Path(self.paths.runs_dir)
        if root is not None and not base.is_absolute():
            base = root / base
        return base / self.meta.name

    def replace(self, **sections: Any) -> "Config":
        return dataclasses.replace(self, **sections)


def _nan_safe(obj: Any) -> Any:
    if isinstance(obj, float) and math.isnan(obj):
        return "nan"
    if isinstance(obj, dict):
        return {k: _nan_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_safe(v) for v in obj]
    return obj


def _build(cls: type, raw: dict[str, Any] | None, section: str) -> Any:
    raw = dict(raw or {})
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(fields)
    if unknown:
        raise ValueError(f"[{section}] unknown keys: {sorted(unknown)}")
    kwargs = {}
    for name, value in raw.items():
        kwargs[name] = _coerce(value)
    return cls(**kwargs)


def _coerce(v: Any) -> Any:
    if isinstance(v, list):
        return tuple(_coerce(x) for x in v)
    return v


def validate(cfg: Config) -> None:
    m, d, o, des = cfg.meta, cfg.data, cfg.optim, cfg.design
    if m.phase not in PHASES:
        raise ValueError(f"meta.phase must be one of {PHASES}")
    hx, hn = d.heldout_triples()
    if not hx or not hn:
        raise ValueError("data.heldout_x and data.heldout_nonx must both be non-empty")
    if hx & hn:
        raise ValueError("held-out triple lists overlap")
    if not all(Op.X in (t[0], t[2]) for t in hx):
        raise ValueError("every heldout_x triple must involve X")
    if any(Op.X in (t[0], t[2]) for t in hn):
        raise ValueError("no heldout_nonx triple may involve X")
    if d.ctx < d.max_row_len:
        raise ValueError(f"data.ctx={d.ctx} too small; need >= {d.max_row_len}")
    if d.depth5 <= max(d.train_depths):
        raise ValueError("data.depth5 must exceed every training depth")
    if cfg.model.ctx != d.ctx:
        raise ValueError("model.ctx must equal data.ctx")
    if cfg.model.d_model % cfg.model.heads:
        raise ValueError("model.d_model must be divisible by model.heads")
    if not set(des.arms) <= set(ARM_KINDS) or "C" not in des.arms:
        raise ValueError(f"design.arms must be a subset of {ARM_KINDS} containing C")
    if des.hybrids and not {"C", "I"} <= set(des.arms):
        raise ValueError("design.hybrids requires arms C and I")
    if 0 not in des.horizons or max(des.horizons) != des.H or list(des.horizons) != sorted(set(des.horizons)):
        raise ValueError("design.horizons must be sorted, unique, start at 0 and end at H")
    if des.K <= 0 or des.H <= 0 or des.T0 <= o.warmup:
        raise ValueError("need K > 0, H > 0 and T0 > warmup")
    if not 0 < o.final_decay_frac < 1:
        raise ValueError("optim.final_decay_frac must be in (0, 1)")
    if d.n_probe_id > d.n_id_dev or d.n_probe_ood > d.n_ood_dev:
        raise ValueError("probe sizes exceed dev split sizes")


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    known = {"meta", "data", "model", "optim", "design", "gate", "stage_b", "eval", "analysis", "runtime", "paths"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    cfg = Config(
        meta=_build(MetaConfig, raw.get("meta"), "meta"),
        data=_build(DataConfig, raw.get("data"), "data"),
        model=_build(ModelConfig, raw.get("model"), "model"),
        optim=_build(OptimConfig, raw.get("optim"), "optim"),
        design=_build(DesignConfig, raw.get("design"), "design"),
        gate=_build(GateConfig, raw.get("gate"), "gate"),
        stage_b=_build(StageBConfig, raw.get("stage_b"), "stage_b"),
        eval=_build(EvalConfig, raw.get("eval"), "eval"),
        analysis=_build(AnalysisConfig, raw.get("analysis"), "analysis"),
        runtime=_build(RuntimeConfig, raw.get("runtime"), "runtime"),
        paths=_build(PathsConfig, raw.get("paths"), "paths"),
        source=str(path),
    )
    validate(cfg)
    return cfg
