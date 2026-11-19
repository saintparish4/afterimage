"""M1: the reference interpreters, the intervention contract and the generator."""

from collections import Counter

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from afterimage.lang import interp_stack, interp_vec
from afterimage.lang.enumerate import count_up_to_depth, trees_up_to_depth
from afterimage.lang.generate import Draws, sample_tree
from afterimage.lang.grammar import (
    ALL_OPS, BOS, EQ, PAD, TRIPLES, Op, Tree, contains_op, depth, from_postfix,
    from_text, parse_triple, signature, to_postfix, to_text, triples_mask,
)
from afterimage.lang.interp import Semantics, affected, evaluate

T, F = Semantics.TRUE, Semantics.FALSE_XNOR

# Literal truth tables, typed here so the tests share no code with the interpreters.
LITERAL = {
    "A": {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1},
    "O": {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1},
    "I": {(0, 0): 1, (0, 1): 1, (1, 0): 0, (1, 1): 1},
    "X": {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0},
    "XNOR": {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 1},
}


def trees(max_depth: int = 6):
    leaves = st.sampled_from([Tree.const(0), Tree.const(1)])

    def extend(children):
        unary = children.map(lambda k: Tree.node(Op.N, k))
        binary = st.tuples(st.sampled_from([Op.A, Op.O, Op.I, Op.X]), children, children).map(
            lambda t: Tree.node(t[0], t[1], t[2])
        )
        return unary | binary

    return st.recursive(leaves, extend, max_leaves=2**max_depth).filter(lambda t: depth(t) <= max_depth)


def row(t: Tree) -> list[int]:
    return [BOS, *to_postfix(t), EQ]


# --- truth tables and agreement ---------------------------------------------

@pytest.mark.parametrize("name", ["A", "O", "I", "X"])
def test_truth_tables_match_literal(name):
    op = Op[name]
    for (a, b), want in LITERAL[name].items():
        t = Tree.node(op, Tree.const(a), Tree.const(b))
        assert evaluate(t, T) == want
        assert interp_stack.evaluate_tokens(to_postfix(t)) == want
        assert interp_vec.evaluate_batch(np.array([row(t)], dtype=np.uint8))[0] == want
    for v in (0, 1):
        assert evaluate(Tree.node(Op.N, Tree.const(v))) == 1 - v


def test_false_rule_is_xnor_only_at_x():
    for (a, b), want in LITERAL["XNOR"].items():
        t = Tree.node(Op.X, Tree.const(a), Tree.const(b))
        assert evaluate(t, F) == want
        assert interp_stack.evaluate_tokens(to_postfix(t), false_rule=True) == want
    for name in ("A", "O", "I"):
        for (a, b), want in LITERAL[name].items():
            assert evaluate(Tree.node(Op[name], Tree.const(a), Tree.const(b)), F) == want


def test_enumeration_counts():
    assert [count_up_to_depth(d) for d in range(4)] == [2, 20, 1622, 10_525_160]
    small = trees_up_to_depth(2)
    assert len(small) == 1622
    assert len({tuple(to_postfix(t)) for t in small}) == 1622


@pytest.mark.parametrize("sem", [T, F])
def test_three_interpreters_agree_exhaustively_to_depth_2(sem):
    small = trees_up_to_depth(2)
    rec = np.array([evaluate(t, sem) for t in small])
    stk = np.array([interp_stack.evaluate_tokens(to_postfix(t), sem is F) for t in small])
    mat = np.zeros((len(small), 12), dtype=np.uint8)
    for i, t in enumerate(small):
        r = row(t)
        mat[i, : len(r)] = r
    vec = interp_vec.evaluate_batch(mat, false_rule=sem is F)
    np.testing.assert_array_equal(rec, stk)
    np.testing.assert_array_equal(rec, vec)


@settings(max_examples=400, deadline=None)
@given(trees(8))
def test_three_interpreters_agree_on_random_trees(t):
    mat = np.array([row(t)], dtype=np.uint8)
    for sem in (T, F):
        r = evaluate(t, sem)
        assert interp_stack.evaluate_tokens(to_postfix(t), sem is F) == r
        assert interp_vec.evaluate_batch(mat, false_rule=sem is F)[0] == r


# --- algebraic properties ----------------------------------------------------

@settings(max_examples=300, deadline=None)
@given(trees(5), trees(5), trees(5))
def test_algebraic_identities(a, b, c):
    ev = evaluate
    N = lambda x: Tree.node(Op.N, x)  # noqa: E731
    B = Tree.node
    assert ev(N(N(a))) == ev(a)
    assert ev(N(B(Op.A, a, b))) == ev(B(Op.O, N(a), N(b)))
    assert ev(N(B(Op.O, a, b))) == ev(B(Op.A, N(a), N(b)))
    for op in (Op.A, Op.O, Op.X):
        assert ev(B(op, a, b)) == ev(B(op, b, a))
    assert ev(B(Op.X, B(Op.X, a, b), c)) == ev(B(Op.X, a, B(Op.X, b, c)))
    assert ev(B(Op.I, a, b)) == ev(B(Op.O, N(a), b))


# --- the intervention contract ----------------------------------------------

def _rewrite_x_as_x_then_not(t: Tree) -> Tree:
    if t.is_const:
        return t
    kids = tuple(_rewrite_x_as_x_then_not(k) for k in t.kids)
    node = Tree(op=t.op, kids=kids)
    return Tree.node(Op.N, node) if t.op == Op.X else node


@settings(max_examples=400, deadline=None)
@given(trees(7))
def test_intervention_contract(t):
    # 1. No X: both semantics agree.
    if not contains_op(t, Op.X):
        assert evaluate(t, F) == evaluate(t, T)
    # 2. The false semantics equals true evaluation with every X node followed by N.
    assert evaluate(t, F) == evaluate(_rewrite_x_as_x_then_not(t), T)
    # 3. affected() is defined by evaluation, not by the presence of X.
    assert affected(t) == (evaluate(t, T) != evaluate(t, F))


def test_nested_x_flips_cancel():
    # a b X c X: two flipped X nodes cancel, so the label is unchanged.
    t = from_postfix(from_text("0 1 X 1 X"))
    assert contains_op(t, Op.X) and not affected(t)
    assert affected(from_postfix(from_text("0 1 X")))


def test_label_restoration_is_exact():
    rng = np.random.default_rng(0)
    d = Draws(rng)
    ts = [sample_tree(4, d) for _ in range(500)]
    mat = np.zeros((len(ts), 40), dtype=np.uint8)
    for i, t in enumerate(ts):
        r = row(t)
        mat[i, : len(r)] = r
    before = mat.copy()
    y_true = interp_vec.evaluate_batch(mat)
    y_false = interp_vec.evaluate_batch(mat, false_rule=True)
    np.testing.assert_array_equal(mat, before)  # evaluation never mutates inputs
    np.testing.assert_array_equal(interp_vec.evaluate_batch(mat), y_true)  # true labels recoverable
    assert (y_true != y_false).any() and (y_true == y_false).any()


# --- grammar -----------------------------------------------------------------

@settings(max_examples=300, deadline=None)
@given(trees(7))
def test_postfix_round_trip(t):
    toks = to_postfix(t)
    assert from_postfix(toks) == t
    assert from_text(to_text(toks)) == toks


@pytest.mark.parametrize("bad", ["0 1", "A", "0 A", "1 1 1 X", "0 N N 1"])
def test_malformed_postfix_rejected(bad):
    with pytest.raises(ValueError):
        from_postfix(from_text(bad))
    mat = np.array([[BOS, *from_text(bad), EQ]], dtype=np.uint8)
    with pytest.raises(ValueError):
        interp_vec.evaluate_batch(mat)


def test_signature_and_triples():
    assert len(TRIPLES) == 45
    t = from_postfix(from_text("0 1 X 1 I"))  # X is slot-1 child of I
    assert signature(t) == {(Op.I, 1, Op.X)}
    t = from_postfix(from_text("1 0 1 O A N"))  # O is slot-2 child of A, A child of N
    assert signature(t) == {(Op.A, 2, Op.O), (Op.N, 1, Op.A)}
    assert signature(Tree.node(Op.X, Tree.const(0), Tree.const(1))) == frozenset()
    assert parse_triple(["I", 1, "X"]) == (Op.I, 1, Op.X)
    with pytest.raises(ValueError):
        parse_triple(["N", 2, "X"])
    assert triples_mask([(Op.I, 1, Op.X)]).bit_count() == 1


# --- generator ----------------------------------------------------------------

@pytest.mark.parametrize("d", [0, 1, 2, 3, 4, 5])
def test_sample_tree_has_exact_depth(d):
    draws = Draws(np.random.default_rng(d))
    for _ in range(300):
        assert depth(sample_tree(d, draws)) == d


def test_generator_is_deterministic():
    da, db = Draws(np.random.default_rng([7, 1])), Draws(np.random.default_rng([7, 1]))
    assert [to_postfix(sample_tree(4, da)) for _ in range(200)] == [
        to_postfix(sample_tree(4, db)) for _ in range(200)
    ]


def test_operator_distribution_is_uniform():
    # The root operator of depth-d trees is uniform over the five operators.
    draws = Draws(np.random.default_rng(123))
    counts = Counter(sample_tree(3, draws).op for _ in range(10_000))
    obs = np.array([counts[o] for o in ALL_OPS])
    assert stats.chisquare(obs).pvalue > 1e-3


def test_padding_is_ignored_by_vector_interpreter():
    t = from_postfix(from_text("1 0 I N"))
    short = np.array([row(t)], dtype=np.uint8)
    long = np.zeros((1, 40), dtype=np.uint8)
    long[0, : short.shape[1]] = short[0]
    assert long[0, -1] == PAD
    assert interp_vec.evaluate_batch(short)[0] == interp_vec.evaluate_batch(long)[0] == evaluate(t)
