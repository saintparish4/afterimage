"""Per-block endpoint values computed from saved per-example predictions.

``role`` names a population: ``"id"`` is the familiar task, ``"ood"`` the
held-out compositions. In ``dev`` mode (pilot, smoke) the roles map to
``id_dev`` / ``ood_dev``; in ``sealed`` mode (confirmatory) they map to
``id_conf`` / ``ood_conf``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from safetensors import safe_open

from ..config import Config
from ..data.splits import Split, load_splits
from ..eval.metrics import SUBSETS, brier, correct, js, nll
from ..experiment.layout import analysis_blocks, arm_dir, block_dir, load_block_json

ROLES = {"dev": {"id": "id_dev", "ood": "ood_dev"}, "sealed": {"id": "id_conf", "ood": "ood_conf"}}


def per_example(lp: np.ndarray, sp: Split, metric: str) -> np.ndarray:
    if metric == "nll":
        return nll(lp, sp.y_true)
    if metric == "brier":
        return brier(lp, sp.y_true)
    if metric == "acc":
        return correct(lp, sp.y_true).astype(np.float64)
    raise ValueError(metric)


@dataclass
class Cube:
    cfg: Config
    mode: str
    blocks: list[int]
    statuses: dict[int, str]
    splits: dict[str, Split]
    _cache: dict = field(default_factory=dict)

    @staticmethod
    def load(cfg: Config, mode: str) -> "Cube":
        if mode not in ROLES:
            raise ValueError(mode)
        blocks, statuses = analysis_blocks(cfg)
        splits = load_splits(cfg.data_path(), ROLES[mode].values())
        return Cube(cfg, mode, blocks, statuses, splits)

    def split(self, role: str) -> Split:
        return self.splits[ROLES[self.mode][role]]

    def _file(self, b: int, arm: str, step: int) -> Path:
        sub = "preds_sealed" if self.mode == "sealed" else "preds"
        return arm_dir(self.cfg, b, arm) / sub / f"step_{step:08d}.safetensors"

    def preds(self, b: int, arm: str, step: int, role: str) -> np.ndarray | None:
        key = (b, arm, step, role)
        if key not in self._cache:
            f = self._file(b, arm, step)
            name = ROLES[self.mode][role]
            if not f.exists():
                self._cache[key] = None
            else:
                with safe_open(str(f), framework="numpy") as h:
                    self._cache[key] = h.get_tensor(name) if name in h.keys() else None
        return self._cache[key]

    def values(self, arm: str, step: int, role: str, subset: str, metric: str) -> np.ndarray:
        """One mean per analysis block (NaN where the arm has no prediction)."""
        sp = self.split(role)
        mask = SUBSETS[subset](sp)
        out = np.full(len(self.blocks), np.nan)
        for i, b in enumerate(self.blocks):
            lp = self.preds(b, arm, step, role)
            if lp is not None and mask.any():
                out[i] = per_example(lp[mask], sp.take(np.flatnonzero(mask)), metric).mean()
        return out

    def example_matrix(self, a: str, c: str, step: int, role: str, metric: str) -> np.ndarray:
        """(blocks x examples) matrix of per-example differences a - c (blocks with both arms only)."""
        sp = self.split(role)
        rows = []
        for b in self.blocks:
            la, lc = self.preds(b, a, step, role), self.preds(b, c, step, role)
            if la is not None and lc is not None:
                rows.append(per_example(la, sp, metric) - per_example(lc, sp, metric))
        return np.array(rows)

    def dispersion(self, a: str, ref: str, step: int, role: str = "ood") -> np.ndarray:
        out = np.full(len(self.blocks), np.nan)
        for i, b in enumerate(self.blocks):
            la, lr = self.preds(b, a, step, role), self.preds(b, ref, step, role)
            if la is not None and lr is not None:
                out[i] = js(la, lr).mean()
        return out

    def relearn_auc(self, arm: str) -> np.ndarray:
        out = np.full(len(self.blocks), np.nan)
        for i, b in enumerate(self.blocks):
            f = arm_dir(self.cfg, b, arm) / "relearn.parquet"
            if f.exists():
                t = pq.read_table(f).to_pydict()
                r, y = np.array(t["r"], float), np.array(t["false_nll"], float)
                if r.size >= 2:
                    out[i] = float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(r)) / (r[-1] - r[0]))
        return out

    def stage(self, b: int) -> dict:
        p = block_dir(self.cfg, b) / "stages.json"
        import json

        return json.loads(p.read_text()) if p.exists() else {}

    def block_info(self, b: int) -> dict:
        return load_block_json(self.cfg, b)
