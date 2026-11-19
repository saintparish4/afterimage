"""Training-state serialization and forking (docs/blueprint.md §6.1-6.2).

A checkpoint directory contains::

    state.safetensors   model/<name>, opt/<idx>/<key>, rng/cpu, rng/cuda/<i>
    state.json          step, optimizer param_groups, metadata
    state.sha256        {"state.safetensors": ..., "state.json": ...}

Directories are written to ``<name>.tmp`` and renamed into place, so a
checkpoint either exists complete or not at all. ``verify`` re-hashes the
files. The *digest* of a state is ``array_digest`` over model and optimizer
tensors plus the step, which is what replay and fork checks compare.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from ..artifacts.hashing import array_digest, atomic_write_json, publish_dir, read_json, sha256_file

FILES = ("state.safetensors", "state.json")


@dataclass
class TrainState:
    model: dict[str, torch.Tensor]
    opt: dict[str, Any] | None          # torch optimizer state_dict; None for weights-only
    step: int
    rng_cpu: torch.Tensor | None = None
    rng_cuda: list[torch.Tensor] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def weights_only(self) -> bool:
        return self.opt is None

    def digest(self) -> str:
        return state_digest(self.model, self.opt, self.step)


def state_digest(model: dict[str, torch.Tensor], opt: dict[str, Any] | None, step: int) -> str:
    arrays: dict[str, Any] = {f"model/{k}": v for k, v in model.items()}
    if opt is not None:
        for idx, st in opt["state"].items():
            for k, v in st.items():
                arrays[f"opt/{idx}/{k}"] = v if torch.is_tensor(v) else torch.tensor(v)
    arrays["step"] = torch.tensor([step], dtype=torch.int64)
    return array_digest(arrays)


def capture(model: torch.nn.Module, opt: torch.optim.Optimizer | None, step: int, meta: dict | None = None,
            with_rng: bool = True) -> TrainState:
    msd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    osd = None
    if opt is not None:
        raw = opt.state_dict()
        osd = {
            "state": {int(i): {k: (v.detach().cpu().clone() if torch.is_tensor(v) else v) for k, v in s.items()}
                      for i, s in raw["state"].items()},
            "param_groups": [dict(g) for g in raw["param_groups"]],
        }
    rng_cpu = torch.get_rng_state() if with_rng else None
    rng_cuda = list(torch.cuda.get_rng_state_all()) if (with_rng and torch.cuda.is_available()) else []
    return TrainState(msd, osd, step, rng_cpu, rng_cuda, dict(meta or {}))


def restore(state: TrainState, model: torch.nn.Module, opt: torch.optim.Optimizer | None) -> None:
    model.load_state_dict(state.model)
    if opt is not None:
        if state.opt is None:
            raise ValueError("cannot restore an optimizer from a weights-only checkpoint")
        opt.load_state_dict(state.opt)
    if state.rng_cpu is not None:
        torch.set_rng_state(state.rng_cpu)
    if state.rng_cuda and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state.rng_cuda)


def save_state(state: TrainState, path: Path) -> str:
    """Write a checkpoint atomically and return its state digest."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    tensors: dict[str, torch.Tensor] = {f"model/{k}": v.contiguous() for k, v in state.model.items()}
    groups = None
    if state.opt is not None:
        for idx, st in state.opt["state"].items():
            for k, v in st.items():
                tensors[f"opt/{idx}/{k}"] = (v if torch.is_tensor(v) else torch.tensor(v)).contiguous()
        groups = [{k: (list(v) if isinstance(v, tuple) else v) for k, v in g.items()} for g in state.opt["param_groups"]]
    if state.rng_cpu is not None:
        tensors["rng/cpu"] = state.rng_cpu
    for i, r in enumerate(state.rng_cuda):
        tensors[f"rng/cuda/{i}"] = r
    save_file(tensors, str(tmp / "state.safetensors"))
    digest = state.digest()
    meta = {"step": state.step, "digest": digest, "weights_only": state.weights_only,
            "param_groups": groups, "meta": state.meta}
    with open(tmp / "state.json", "w") as f:
        json.dump(meta, f, sort_keys=True, indent=2, default=str)
    atomic_write_json(tmp / "state.sha256", {n: sha256_file(tmp / n) for n in FILES})
    publish_dir(tmp, path)
    return digest


def verify(path: Path) -> bool:
    path = Path(path)
    sums = path / "state.sha256"
    if not sums.exists():
        return False
    try:
        want = read_json(sums)
        return all(sha256_file(path / n) == want[n] for n in FILES)
    except (OSError, KeyError, ValueError):
        return False


def load_state(path: Path, *, check: bool = True) -> TrainState:
    path = Path(path)
    if check and not verify(path):
        raise ValueError(f"checkpoint {path} is missing or corrupt")
    tensors = load_file(str(path / "state.safetensors"))
    meta = read_json(path / "state.json")
    model = {k[len("model/"):]: v for k, v in tensors.items() if k.startswith("model/")}
    opt = None
    if not meta["weights_only"]:
        st: dict[int, dict[str, torch.Tensor]] = {}
        for k, v in tensors.items():
            if k.startswith("opt/"):
                _, idx, name = k.split("/", 2)
                st.setdefault(int(idx), {})[name] = v
        groups = [{k: (tuple(v) if k == "betas" else v) for k, v in g.items()} for g in meta["param_groups"]]
        opt = {"state": dict(sorted(st.items())), "param_groups": groups}
    rng_cuda = [tensors[k] for k in sorted((k for k in tensors if k.startswith("rng/cuda/")), key=lambda s: int(s.rsplit("/", 1)[1]))]
    state = TrainState(model, opt, int(meta["step"]), tensors.get("rng/cpu"), rng_cuda, meta.get("meta", {}))
    if check and state.digest() != meta["digest"]:
        raise ValueError(f"checkpoint {path}: digest mismatch")
    return state


def recorded_digest(path: Path) -> str:
    return read_json(Path(path) / "state.json")["digest"]


def fork(parent: Path, child: Path, meta_update: dict) -> str:
    """Byte-copy a verified checkpoint into a new history and record its parent."""
    if not verify(parent):
        raise ValueError(f"cannot fork from unverified checkpoint {parent}")
    st = load_state(parent)
    st.meta = {**st.meta, **meta_update, "parent": str(parent), "parent_digest": recorded_digest(parent)}
    return save_state(st, child)
