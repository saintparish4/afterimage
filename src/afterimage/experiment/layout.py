"""On-disk layout of a study and helpers to read what it contains.

::

    runs/<name>/
      ledger.csv  determinism.json  locks/
      block_007/
        block.json           seed, status, gate summary, throughput
        gate.json  stages.json
        base/                logs/  eval/  preds/  resume/
        fork/                full state at T0 (all arms start here)
        arms/<ARM>/
          stage_a.json  relearn.parquet  done
          resume/ post_window/ final/        full states
          ckpt_<step>/                       weights at each horizon
          logs/ eval/ preds/ preds_sealed/
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..artifacts.hashing import read_json
from ..config import Config

STATUS_COMPLETE = "complete"
STATUS_EXCLUDED = "excluded_gate"
STATUS_FAILED = "postfork_failed"
TERMINAL = (STATUS_COMPLETE, STATUS_EXCLUDED, STATUS_FAILED)


def study_dir(cfg: Config) -> Path:
    return cfg.runs_path()


def block_dir(cfg: Config, b: int) -> Path:
    return study_dir(cfg) / f"block_{b:03d}"


def arm_dir(cfg: Config, b: int, arm: str) -> Path:
    return block_dir(cfg, b) / "arms" / arm


def block_status(cfg: Config, b: int) -> str | None:
    p = block_dir(cfg, b) / "block.json"
    return read_json(p).get("status") if p.exists() else None


def list_blocks(cfg: Config) -> list[int]:
    d = study_dir(cfg)
    if not d.exists():
        return []
    return sorted(int(p.name.split("_")[1]) for p in d.glob("block_[0-9][0-9][0-9]") if p.is_dir())


def analysis_blocks(cfg: Config) -> tuple[list[int], dict[int, str]]:
    """The first n_blocks blocks (by index) that passed the gate, plus every status.

    Gate-excluded blocks are replaced by later indices (pre-treatment exclusion).
    Post-fork failures are *kept* in the list: they are never silently dropped.
    """
    statuses = {b: block_status(cfg, b) or "missing" for b in list_blocks(cfg)}
    chosen = [b for b in sorted(statuses) if statuses[b] in (STATUS_COMPLETE, STATUS_FAILED)]
    return chosen[: cfg.design.n_blocks], statuses


def read_parts(directory: Path, keys: list[str]) -> pa.Table | None:
    files = sorted(directory.glob("*.parquet")) if directory.exists() else []
    if not files:
        return None
    t = pa.concat_tables([pq.read_table(f) for f in files], promote_options="default")
    # A resumed run rewrites rows it had already logged: keep the last occurrence.
    cols = [t.column(k).to_pylist() for k in keys]
    last: dict[tuple, int] = {}
    for i, key in enumerate(zip(*cols)):
        last[key] = i
    return t.take(pa.array(sorted(last.values()), type=pa.int64()))


def read_eval(cfg: Config, b: int) -> pa.Table | None:
    bd = block_dir(cfg, b)
    parts = [read_parts(bd / "base" / "eval", ["arm", "step", "split", "subset"])]
    for a in sorted((bd / "arms").glob("*")) if (bd / "arms").exists() else []:
        parts.append(read_parts(a / "eval", ["arm", "step", "split", "subset"]))
    parts = [p for p in parts if p is not None]
    return pa.concat_tables(parts, promote_options="default") if parts else None


def load_block_json(cfg: Config, b: int) -> dict:
    return read_json(block_dir(cfg, b) / "block.json")
