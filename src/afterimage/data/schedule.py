"""Stateless, counter-based batch schedule and seed derivation.

The batch at step ``t`` is a pure function of ``(block_seed, key, t)``. There
is no iterator state to save, so resuming from a checkpoint and forking a
training history are exact by construction.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

KEY_ID = {"main": 0, "alt": 1, "sham": 2, "init": 3, "relearn": 4}


def block_seed(master_seed: int, block: int) -> int:
    """Independent 64-bit seed for block ``block`` (SeedSequence spawn semantics)."""
    ss = np.random.SeedSequence(master_seed, spawn_key=(block,))
    a, b = ss.generate_state(2, dtype=np.uint32)
    return (int(a) << 32) | int(b)


def derive_seed(*keys: int) -> int:
    """63-bit seed from a tuple of non-negative ints (for torch generators)."""
    a, b = np.random.SeedSequence(list(keys)).generate_state(2, dtype=np.uint32)
    return ((int(a) << 32) | int(b)) & ((1 << 63) - 1)


@lru_cache(maxsize=6)
def _permutation(block_seed_: int, key_id: int, epoch: int, pool_size: int) -> np.ndarray:
    perm = np.random.default_rng([block_seed_, key_id, epoch]).permutation(pool_size)
    perm.setflags(write=False)
    return perm


def batch_indices(block_seed_: int, key: str, step: int, pool_size: int, batch: int) -> np.ndarray:
    """Pool indices for ``step``. Epochs are fresh permutations; batches may straddle two."""
    if batch > pool_size:
        raise ValueError("batch larger than pool")
    kid = KEY_ID[key]
    start = step * batch
    epoch, offset = divmod(start, pool_size)
    perm = _permutation(block_seed_, kid, epoch, pool_size)
    idx = perm[offset : offset + batch]
    if idx.size < batch:
        nxt = _permutation(block_seed_, kid, epoch + 1, pool_size)
        idx = np.concatenate([idx, nxt[: batch - idx.size]])
    return idx.astype(np.int64, copy=True)


def sham_positions(block_seed_: int, step: int, batch: int, n_flip: int) -> np.ndarray:
    """Uniformly random batch positions whose labels arm S flips at ``step``."""
    rng = np.random.default_rng([block_seed_, KEY_ID["sham"], step])
    return np.sort(rng.choice(batch, size=n_flip, replace=False))
