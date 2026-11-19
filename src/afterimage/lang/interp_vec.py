"""Reference interpreter #3: a vectorized stack machine over a token matrix.

Evaluates every row of an (n, L) uint8 matrix of padded example rows at once.
Used to label the training pool and the evaluation splits; agreement with
the other two interpreters is verified exhaustively in the tests.
"""

from __future__ import annotations

import numpy as np

from .grammar import ONE, ZERO, Op


def evaluate_batch(tokens: np.ndarray, false_rule: bool = False) -> np.ndarray:
    tokens = np.asarray(tokens)
    if tokens.ndim != 2:
        raise ValueError("tokens must be 2-D (n, L)")
    n, L = tokens.shape
    stack = np.zeros((n, L + 1), dtype=np.uint8)
    sp = np.zeros(n, dtype=np.int64)
    rows = np.arange(n)
    for j in range(L):
        tok = tokens[:, j]
        m = (tok == ZERO) | (tok == ONE)
        if m.any():
            r = rows[m]
            stack[r, sp[m]] = (tok[m] - ZERO).astype(np.uint8)
            sp[m] += 1
        m = tok == Op.N
        if m.any():
            r = rows[m]
            if (sp[m] < 1).any():
                raise ValueError("stack underflow at N")
            stack[r, sp[m] - 1] ^= 1
        for op in (Op.A, Op.O, Op.I, Op.X):
            m = tok == op
            if not m.any():
                continue
            r = rows[m]
            if (sp[m] < 2).any():
                raise ValueError(f"stack underflow at {op.name}")
            a = stack[r, sp[m] - 2].astype(bool)
            b = stack[r, sp[m] - 1].astype(bool)
            if op == Op.A:
                v = a & b
            elif op == Op.O:
                v = a | b
            elif op == Op.I:
                v = ~a | b
            else:
                v = a ^ b
                if false_rule:
                    v = ~v
            stack[r, sp[m] - 2] = v.astype(np.uint8)
            sp[m] -= 1
    if (sp != 1).any():
        bad = int(np.flatnonzero(sp != 1)[0])
        raise ValueError(f"row {bad} leaves {int(sp[bad])} values on the stack")
    return stack[:, 0].copy()
