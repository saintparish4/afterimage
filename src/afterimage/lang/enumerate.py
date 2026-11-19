"""Exhaustive enumeration of small expression spaces.

Counts of trees with depth <= d: 2, 20, 1,622, 10,525,160 for d = 0..3
(T(d) = 2 + T(d-1) + 4 * T(d-1)^2).
"""

from __future__ import annotations

from .grammar import BINARY_OPS, Op, Tree


def count_up_to_depth(d: int) -> int:
    t = 2
    for _ in range(d):
        t = 2 + t + 4 * t * t
    return t


def trees_up_to_depth(d: int) -> list[Tree]:
    """All trees with depth <= d, in a fixed deterministic order.

    Practical for d <= 2 (1,622 trees). Depth 3 is checked with a vectorized
    construction in the tests instead of Python objects.
    """
    level: list[Tree] = [Tree.const(0), Tree.const(1)]
    for _ in range(d):
        nxt: list[Tree] = [Tree.const(0), Tree.const(1)]
        nxt += [Tree(op=Op.N, kids=(a,)) for a in level]
        for op in BINARY_OPS:
            nxt += [Tree(op=op, kids=(a, b)) for a in level for b in level]
        level = nxt
    return level
