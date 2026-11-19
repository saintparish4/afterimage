"""M1 slow check: every tree of depth <= 3 (10,525,160 trees), both semantics.

Depth-3 token rows are built combinatorially from the 1,622 trees of depth
<= 2. The expected value of each row is composed from the recursive
interpreter's depth-<=2 values and literal truth tables, then compared with
the vectorized interpreter on every row and with the scalar stack machine on
a random sample.
"""

import numpy as np
import pytest

from afterimage.lang import interp_stack, interp_vec
from afterimage.lang.enumerate import count_up_to_depth, trees_up_to_depth
from afterimage.lang.grammar import ONE, ZERO, Op, to_postfix
from afterimage.lang.interp import Semantics, evaluate

TABLE = {
    Op.A: np.array([[0, 0], [0, 1]], dtype=np.uint8),
    Op.O: np.array([[0, 1], [1, 1]], dtype=np.uint8),
    Op.I: np.array([[1, 1], [0, 1]], dtype=np.uint8),
    Op.X: np.array([[0, 1], [1, 0]], dtype=np.uint8),
}
XNOR = np.array([[1, 0], [0, 1]], dtype=np.uint8)
WIDTH = 15  # longest depth-3 postfix expression


@pytest.mark.slow
@pytest.mark.parametrize("false_rule", [False, True])
def test_all_trees_up_to_depth_3(false_rule):
    sem = Semantics.FALSE_XNOR if false_rule else Semantics.TRUE
    base = trees_up_to_depth(2)
    toks = [to_postfix(t) for t in base]
    vals = np.array([evaluate(t, sem) for t in base], dtype=np.uint8)
    lens = np.array([len(t) for t in toks])
    padded = np.zeros((len(base), 7), dtype=np.uint8)
    for i, t in enumerate(toks):
        padded[i, : len(t)] = t
    by_len = {int(L): np.flatnonzero(lens == L) for L in np.unique(lens)}
    rng = np.random.default_rng(0)
    total = 0

    def check(mat: np.ndarray, want: np.ndarray) -> None:
        nonlocal total
        got = interp_vec.evaluate_batch(mat, false_rule=false_rule)
        np.testing.assert_array_equal(got, want)
        sample = rng.choice(len(mat), size=min(2000, len(mat)), replace=False)
        for i in sample:
            assert interp_stack.evaluate_tokens(mat[i], false_rule) == want[i]
        total += len(mat)

    # depth 0: constants
    check(np.array([[ZERO], [ONE]], dtype=np.uint8), np.array([0, 1], dtype=np.uint8))
    # N over each depth<=2 tree
    m = np.zeros((len(base), WIDTH), dtype=np.uint8)
    m[:, :7] = padded
    m[np.arange(len(base)), lens] = Op.N
    check(m, 1 - vals)
    # binary op over every ordered pair
    for op in (Op.A, Op.O, Op.I, Op.X):
        table = XNOR if (false_rule and op == Op.X) else TABLE[op]
        for la, ia in by_len.items():
            for lb, ib in by_len.items():
                na, nb = len(ia), len(ib)
                mat = np.zeros((na * nb, WIDTH), dtype=np.uint8)
                mat[:, :la] = np.repeat(padded[ia, :la], nb, axis=0)
                mat[:, la : la + lb] = np.tile(padded[ib, :lb], (na, 1))
                mat[:, la + lb] = op
                want = table[np.repeat(vals[ia], nb), np.tile(vals[ib], na)]
                check(mat, want)
    assert total == count_up_to_depth(3) == 10_525_160
