"""Split construction, Parquet I/O, manifests and the leakage audit.

Splits (see docs/blueprint.md §4.5):

* ``id_small``: every tree of depth 1-2 without a held-out triple (enumerated).
  The pool covers these almost completely, so they measure lookup, not
  generalization, and they are reported separately.
* ``id_dev`` / ``id_conf``: depth 3-4, no held-out triple, disjoint from the
  pool and from each other.
* ``ood_dev`` / ``ood_conf``: depth 2-4, at least one held-out triple. They
  cannot occur in the pool by construction and are disjoint from each other.
* ``depth5``: depth 5, no held-out triple, no longer than the longest
  training row (exploratory productivity check).
* ``probe``: the first rows of ``id_dev`` and ``ood_dev`` (fixed probes).
* ``pool``: the training stream for every arm. Depth 1-4, no held-out
  triple, excludes every ``id_dev`` row. Duplicates are allowed.

``id_conf`` and ``ood_conf`` are *sealed*: their seed is derived from the
commit hash of the frozen pre-registration, so they cannot exist before the
analysis plan does.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..artifacts.hashing import array_digest, atomic_write_json, read_json, sha256_file
from ..config import Config, DataConfig
from ..lang import interp_vec
from ..lang.enumerate import trees_up_to_depth
from ..lang.generate import Draws, sample_depth, sample_tree
from ..lang.grammar import BOS, EQ, Op, Tree, contains_op, depth, from_postfix, signature, to_postfix, triples_mask
from ..lang.interp import Semantics, evaluate

GENERATOR_VERSION = "afterimage-data-v1"
SPLIT_KEY = {"id_dev": 1, "ood_dev": 2, "depth5": 3, "pool": 4, "id_conf": 5, "ood_conf": 6}
DEV_SPLITS = ("id_small", "id_dev", "ood_dev", "depth5", "probe")
SEALED_SPLITS = ("id_conf", "ood_conf")


@dataclass
class Split:
    """Column arrays for one split. All arrays have length ``n``."""

    name: str
    tokens: np.ndarray       # (n, ctx) uint8, BOS expr EQ PAD...
    eq_pos: np.ndarray       # (n,) uint8, index of EQ
    depth: np.ndarray        # (n,) uint8
    y_true: np.ndarray       # (n,) uint8
    y_false: np.ndarray      # (n,) uint8
    has_x: np.ndarray        # (n,) bool
    sig_mask: np.ndarray     # (n,) uint64, bit i = TRIPLES[i] present
    heldout_x: np.ndarray    # (n,) bool, contains a held-out triple involving X
    heldout_nonx: np.ndarray  # (n,) bool, contains a held-out triple not involving X
    source: np.ndarray       # (n,) uint8, 0 = id, 1 = ood (meaningful for probe)

    COLUMNS = ("tokens", "eq_pos", "depth", "y_true", "y_false", "has_x", "sig_mask",
               "heldout_x", "heldout_nonx", "source")

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def affected(self) -> np.ndarray:
        return self.y_true != self.y_false

    def arrays(self) -> dict[str, np.ndarray]:
        return {c: getattr(self, c) for c in self.COLUMNS}

    def take(self, idx: np.ndarray, name: str | None = None) -> "Split":
        return Split(name=name or self.name, **{c: getattr(self, c)[idx] for c in self.COLUMNS})

    @staticmethod
    def concat(parts: list["Split"], name: str) -> "Split":
        return Split(name=name, **{c: np.concatenate([getattr(p, c) for p in parts]) for c in Split.COLUMNS})

    def content_digest(self) -> str:
        return array_digest(self.arrays())


# --- encoding -----------------------------------------------------------------

def _masks(cfg: DataConfig) -> tuple[int, int]:
    hx, hn = cfg.heldout_triples()
    return triples_mask(hx), triples_mask(hn)


def encode(trees: list[Tree], ctx: int, hx_mask: int, hn_mask: int, name: str, source: int = 0) -> Split:
    n = len(trees)
    tokens = np.zeros((n, ctx), dtype=np.uint8)
    eq_pos = np.zeros(n, dtype=np.uint8)
    dep = np.zeros(n, dtype=np.uint8)
    has_x = np.zeros(n, dtype=bool)
    sig = np.zeros(n, dtype=np.uint64)
    for i, t in enumerate(trees):
        toks = to_postfix(t)
        L = len(toks)
        tokens[i, 0] = BOS
        tokens[i, 1 : L + 1] = toks
        tokens[i, L + 1] = EQ
        eq_pos[i] = L + 1
        dep[i] = depth(t)
        has_x[i] = contains_op(t, Op.X)
        sig[i] = triples_mask(signature(t))
    y_true = interp_vec.evaluate_batch(tokens)
    y_false = interp_vec.evaluate_batch(tokens, false_rule=True)
    return Split(
        name=name, tokens=tokens, eq_pos=eq_pos, depth=dep, y_true=y_true, y_false=y_false,
        has_x=has_x, sig_mask=sig,
        heldout_x=(sig & np.uint64(hx_mask)) != 0,
        heldout_nonx=(sig & np.uint64(hn_mask)) != 0,
        source=np.full(n, source, dtype=np.uint8),
    )


def row_hash(tokens: np.ndarray) -> np.ndarray:
    """64-bit FNV-style hash per row; used only to *reject* candidates.

    A collision can only cause a valid candidate to be rejected (harmless and
    deterministic). Exact disjointness is verified byte-for-byte by ``audit``.
    """
    tokens = np.ascontiguousarray(tokens, dtype=np.uint8)
    n, L = tokens.shape
    pad = (-L) % 8
    if pad:
        tokens = np.pad(tokens, ((0, 0), (0, pad)))
    words = np.ascontiguousarray(tokens).view(np.uint64)
    h = np.full(n, 0xCBF29CE484222325, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(words.shape[1]):
            h ^= words[:, j]
            h *= np.uint64(0x100000001B3)
            h ^= h >> np.uint64(29)
    return h


def _member(sorted_hashes: np.ndarray, h: np.ndarray) -> np.ndarray:
    if sorted_hashes.size == 0:
        return np.zeros(h.shape, dtype=bool)
    pos = np.searchsorted(sorted_hashes, h)
    pos = np.minimum(pos, sorted_hashes.size - 1)
    return sorted_hashes[pos] == h


# --- sampling -----------------------------------------------------------------

HeldoutMode = Literal["exclude", "require"]


def sample_split(
    n: int, *, name: str, seed_keys: list[int], depths: tuple[int, ...], weights: tuple[float, ...] | None,
    heldout: HeldoutMode, cfg: DataConfig, exclude_hashes: np.ndarray | None = None,
    unique: bool = True, max_len: int | None = None, batch: int = 2048, source: int = 0,
) -> Split:
    """Rejection-sample ``n`` examples deterministically from ``seed_keys``."""
    hx_mask, hn_mask = _masks(cfg)
    h_all = hx_mask | hn_mask
    max_len = max_len or cfg.max_expr_len
    excl = np.sort(exclude_hashes) if exclude_hashes is not None else np.empty(0, np.uint64)
    draws = Draws(np.random.default_rng(seed_keys))
    seen: set[int] = set()
    parts: list[Split] = []
    have = 0
    while have < n:
        cand: list[Tree] = []
        while len(cand) < batch:
            t = sample_tree(sample_depth(depths, weights, draws), draws)
            m = triples_mask(signature(t))
            if (heldout == "exclude" and m & h_all) or (heldout == "require" and not m & h_all):
                continue
            if len(to_postfix(t)) > max_len:
                continue
            cand.append(t)
        sp = encode(cand, cfg.ctx, hx_mask, hn_mask, name, source)
        h = row_hash(sp.tokens)
        keep = ~_member(excl, h)
        if unique:
            for i in np.flatnonzero(keep):
                hi = int(h[i])
                if hi in seen:
                    keep[i] = False
                else:
                    seen.add(hi)
        idx = np.flatnonzero(keep)[: n - have]
        parts.append(sp.take(idx))
        have += len(idx)
    out = Split.concat(parts, name)
    _cross_check(out)
    return out


def _cross_check(sp: Split, k: int = 2000) -> None:
    """Defense in depth: relabel a deterministic sample with the recursive oracle."""
    idx = np.linspace(0, len(sp) - 1, num=min(k, len(sp)), dtype=np.int64) if len(sp) else []
    for i in idx:
        expr = sp.tokens[i, 1 : int(sp.eq_pos[i])]
        t = from_postfix(expr)
        if evaluate(t, Semantics.TRUE) != sp.y_true[i] or evaluate(t, Semantics.FALSE_XNOR) != sp.y_false[i]:
            raise AssertionError(f"oracle disagreement in split {sp.name} row {i}")


def _pool_chunk(args: tuple[DataConfig, int, int, int, np.ndarray]) -> Split:
    cfg, data_seed, chunk, n, excl = args
    return sample_split(
        n, name="pool", seed_keys=[data_seed, SPLIT_KEY["pool"], chunk], depths=cfg.train_depths,
        weights=cfg.depth_weights, heldout="exclude", cfg=cfg, exclude_hashes=excl, unique=False,
    )


def build_pool(cfg: DataConfig, data_seed: int, exclude: np.ndarray) -> Split:
    sizes = [cfg.chunk_size] * (cfg.pool_size // cfg.chunk_size)
    if cfg.pool_size % cfg.chunk_size:
        sizes.append(cfg.pool_size % cfg.chunk_size)
    jobs = [(cfg, data_seed, c, n, exclude) for c, n in enumerate(sizes)]
    workers = cfg.workers or os.cpu_count() or 1
    if workers == 1 or len(jobs) == 1:
        parts = [_pool_chunk(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(min(workers, len(jobs))) as p:
            parts = p.map(_pool_chunk, jobs, chunksize=1)  # map preserves chunk order
    return Split.concat(parts, "pool")


def build_id_small(cfg: DataConfig) -> Split:
    hx_mask, hn_mask = _masks(cfg)
    trees = [t for t in trees_up_to_depth(2) if depth(t) >= 1 and not triples_mask(signature(t)) & (hx_mask | hn_mask)]
    return encode(trees, cfg.ctx, hx_mask, hn_mask, "id_small")


# --- Parquet I/O -----------------------------------------------------------

def to_table(sp: Split) -> pa.Table:
    n, ctx = sp.tokens.shape
    tok = pa.FixedSizeBinaryArray.from_buffers(pa.binary(ctx), n, [None, pa.py_buffer(sp.tokens.tobytes())])
    cols = {"tokens": tok}
    for c in Split.COLUMNS[1:]:
        cols[c] = pa.array(getattr(sp, c))
    return pa.table(cols)


def write_split(sp: Split, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(to_table(sp), tmp, compression="zstd")
    os.replace(tmp, path)
    return split_stats(sp) | {"file": path.name, "file_sha256": sha256_file(path)}


def read_split(path: Path, name: str | None = None) -> Split:
    t = pq.read_table(path)
    tok = t.column("tokens").combine_chunks()
    ctx = tok.type.byte_width
    buf = tok.buffers()[1]
    tokens = np.frombuffer(buf, dtype=np.uint8, count=len(tok) * ctx, offset=tok.offset * ctx).reshape(len(tok), ctx)
    cols = {c: np.array(t.column(c).to_numpy(), copy=True) for c in Split.COLUMNS[1:]}
    return Split(name=name or path.stem, tokens=tokens.copy(), **cols)


def split_stats(sp: Split) -> dict:
    d, c = np.unique(sp.depth, return_counts=True)
    mean = lambda a: float(np.mean(a)) if len(a) else float("nan")  # noqa: E731
    return {
        "rows": len(sp),
        "content_digest": sp.content_digest(),
        "depth_counts": {int(k): int(v) for k, v in zip(d, c)},
        "affected_rate": mean(sp.affected),
        "y_true_mean": mean(sp.y_true),
        "y_false_mean": mean(sp.y_false),
        "has_x_rate": mean(sp.has_x),
        "heldout_x_rate": mean(sp.heldout_x),
        "heldout_nonx_rate": mean(sp.heldout_nonx),
    }


# --- building ------------------------------------------------------------------

def build_dev(cfg: Config, out: Path, log=print) -> dict:
    """Build pool + development splits. Deterministic in (data config, data_seed)."""
    d, seed = cfg.data, cfg.meta.data_seed
    out.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict] = {}

    log("id_small: enumerating depth <= 2")
    id_small = build_id_small(d)
    log("id_dev")
    id_dev = sample_split(d.n_id_dev, name="id_dev", seed_keys=[seed, SPLIT_KEY["id_dev"]],
                          depths=d.id_eval_depths, weights=None, heldout="exclude", cfg=d)
    log("ood_dev")
    ood_dev = sample_split(d.n_ood_dev, name="ood_dev", seed_keys=[seed, SPLIT_KEY["ood_dev"]],
                           depths=d.ood_depths, weights=None, heldout="require", cfg=d, source=1)
    log("depth5")
    depth5 = sample_split(d.n_depth5, name="depth5", seed_keys=[seed, SPLIT_KEY["depth5"]],
                          depths=(d.depth5,), weights=None, heldout="exclude", cfg=d, max_len=d.max_expr_len)
    probe = Split.concat([id_dev.take(np.arange(d.n_probe_id)), ood_dev.take(np.arange(d.n_probe_ood))], "probe")
    log(f"pool: {d.pool_size:,} rows")
    pool = build_pool(d, seed, row_hash(id_dev.tokens))

    for sp in (id_small, id_dev, ood_dev, depth5, probe, pool):
        entries[sp.name] = write_split(sp, out / f"{sp.name}.parquet")
    manifest = {
        "generator": GENERATOR_VERSION,
        "data_hash": cfg.data_hash,
        "data_seed": seed,
        "data_config": _jsonable(d),
        "heldout_x": [[o.name if isinstance(o, Op) else o for o in t] for t in sorted(d.heldout_triples()[0])],
        "heldout_nonx": [[o.name if isinstance(o, Op) else o for o in t] for t in sorted(d.heldout_triples()[1])],
        "splits": entries,
    }
    atomic_write_json(out / "manifest.json", manifest)
    return manifest


def sealed_seed(commit: str) -> int:
    return int(hashlib.sha256(b"afterimage-sealed:" + commit.encode()).hexdigest()[:16], 16)


def build_sealed(cfg: Config, data_dir: Path, commit: str, log=print) -> dict:
    """Build id_conf / ood_conf from a seed derived from the pre-registration commit."""
    d = cfg.data
    if (data_dir / "sealed_manifest.json").exists():
        raise FileExistsError("sealed splits already exist; they are generated exactly once")
    seed = sealed_seed(commit)
    pool = read_split(data_dir / "pool.parquet")
    id_dev = read_split(data_dir / "id_dev.parquet")
    ood_dev = read_split(data_dir / "ood_dev.parquet")
    log("id_conf")
    id_conf = sample_split(d.n_id_conf, name="id_conf", seed_keys=[seed, SPLIT_KEY["id_conf"]],
                           depths=d.id_eval_depths, weights=None, heldout="exclude", cfg=d,
                           exclude_hashes=np.concatenate([row_hash(pool.tokens), row_hash(id_dev.tokens)]))
    log("ood_conf")
    ood_conf = sample_split(d.n_ood_conf, name="ood_conf", seed_keys=[seed, SPLIT_KEY["ood_conf"]],
                            depths=d.ood_depths, weights=None, heldout="require", cfg=d,
                            exclude_hashes=row_hash(ood_dev.tokens), source=1)
    entries = {sp.name: write_split(sp, data_dir / f"{sp.name}.parquet") for sp in (id_conf, ood_conf)}
    manifest = {"prereg_commit": commit, "seed": seed,
                "seed_derivation": "int(sha256(b'afterimage-sealed:' + commit).hexdigest()[:16], 16)",
                "splits": entries}
    atomic_write_json(data_dir / "sealed_manifest.json", manifest)
    return manifest


def _jsonable(d: DataConfig) -> dict:
    return {f.name: getattr(d, f.name) for f in fields(d)}


# --- loading and verification -------------------------------------------------

def load_splits(data_dir: Path, names: Iterable[str]) -> dict[str, Split]:
    return {n: read_split(data_dir / f"{n}.parquet", n) for n in names}


def sealed_available(data_dir: Path) -> bool:
    return (data_dir / "sealed_manifest.json").exists()


def verify(data_dir: Path) -> list[str]:
    """Return a list of problems (empty if every file matches its manifest)."""
    problems: list[str] = []
    for mf in ("manifest.json", "sealed_manifest.json"):
        p = data_dir / mf
        if not p.exists():
            if mf == "manifest.json":
                problems.append("manifest.json missing")
            continue
        for name, e in read_json(p)["splits"].items():
            f = data_dir / e["file"]
            if not f.exists():
                problems.append(f"{name}: file missing")
                continue
            if sha256_file(f) != e["file_sha256"]:
                problems.append(f"{name}: file sha256 mismatch")
            if read_split(f).content_digest() != e["content_digest"]:
                problems.append(f"{name}: content digest mismatch")
    return problems


def _void(tokens: np.ndarray) -> np.ndarray:
    t = np.ascontiguousarray(tokens)
    return t.view(np.dtype((np.void, t.shape[1]))).ravel()


def audit(cfg: Config, data_dir: Path) -> dict:
    """Exact leakage audit. Returns {check_name: (ok, detail)}."""
    hx_mask, hn_mask = _masks(cfg.data)
    h_all = np.uint64(hx_mask | hn_mask)
    names = [n for n in ("pool", "id_small", "id_dev", "ood_dev", "depth5", "id_conf", "ood_conf")
             if (data_dir / f"{n}.parquet").exists()]
    sp = load_splits(data_dir, names)
    out: dict[str, tuple[bool, str]] = {}
    pool = sp["pool"]
    n_bad = int(((pool.sig_mask & h_all) != 0).sum())
    out["pool_has_no_heldout_triples"] = (n_bad == 0, f"{n_bad} rows")
    for n in ("ood_dev", "ood_conf"):
        if n in sp:
            n_bad = int(((sp[n].sig_mask & h_all) == 0).sum())
            out[f"{n}_all_contain_heldout"] = (n_bad == 0, f"{n_bad} rows lack a held-out triple")
    for n in ("id_dev", "id_conf", "depth5", "id_small"):
        if n in sp:
            n_bad = int(((sp[n].sig_mask & h_all) != 0).sum())
            out[f"{n}_has_no_heldout_triples"] = (n_bad == 0, f"{n_bad} rows")
    pool_v = np.unique(_void(pool.tokens))
    for n in ("id_dev", "id_conf", "ood_dev", "ood_conf", "depth5"):
        if n in sp:
            k = np.intersect1d(pool_v, _void(sp[n].tokens)).size
            out[f"pool_disjoint_from_{n}"] = (k == 0, f"{k} shared rows")
    evals = [n for n in ("id_small", "id_dev", "id_conf", "ood_dev", "ood_conf", "depth5") if n in sp]
    for i, a in enumerate(evals):
        va = _void(sp[a].tokens)
        out[f"{a}_rows_unique"] = (np.unique(va).size == va.size, "")
        for b in evals[i + 1 :]:
            k = np.intersect1d(va, _void(sp[b].tokens)).size
            out[f"{a}_disjoint_from_{b}"] = (k == 0, f"{k} shared rows")
    for n, s in sp.items():
        ok = bool((interp_vec.evaluate_batch(s.tokens) == s.y_true).all()
                  and (interp_vec.evaluate_batch(s.tokens, false_rule=True) == s.y_false).all())
        out[f"{n}_labels_recompute"] = (ok, "")
    return out
