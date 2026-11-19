"""Random expression generator with exact depth control.

``sample_tree(d)`` returns a tree whose depth is exactly ``d``: at each
operator node one argument (chosen uniformly) carries depth ``d-1`` and the
other gets a depth drawn uniformly from ``0..d-1``. Operators are uniform
over (N, A, O, I, X).

All randomness flows through ``Draws``, which buffers uniforms from a NumPy
Generator. Python's ``random`` module is never used.
"""

from __future__ import annotations

import numpy as np

from .grammar import ALL_OPS, Op, Tree

_BUF = 4096


class Draws:
    """Buffered integer draws from a ``numpy.random.Generator``."""

    def __init__(self, rng: np.random.Generator):
        self._rng = rng
        self._buf = rng.random(_BUF)
        self._i = 0

    def int(self, k: int) -> int:
        """Uniform integer in [0, k)."""
        if self._i >= _BUF:
            self._buf = self._rng.random(_BUF)
            self._i = 0
        u = self._buf[self._i]
        self._i += 1
        return min(int(u * k), k - 1)


def sample_tree(d: int, draws: Draws) -> Tree:
    if d < 0:
        raise ValueError("depth must be >= 0")
    if d == 0:
        return Tree.const(draws.int(2))
    op = ALL_OPS[draws.int(len(ALL_OPS))]
    if op == Op.N:
        return Tree(op=op, kids=(sample_tree(d - 1, draws),))
    deep_slot = draws.int(2)
    deep = sample_tree(d - 1, draws)
    other = sample_tree(draws.int(d), draws)
    kids = (deep, other) if deep_slot == 0 else (other, deep)
    return Tree(op=op, kids=kids)


def sample_depth(depths: tuple[int, ...], weights: tuple[float, ...] | None, draws: Draws) -> int:
    if weights is None:
        return depths[draws.int(len(depths))]
    total = float(sum(weights))
    u = draws.int(1 << 30) / float(1 << 30) * total
    acc = 0.0
    for d, w in zip(depths, weights):
        acc += w
        if u < acc:
            return d
    return depths[-1]
