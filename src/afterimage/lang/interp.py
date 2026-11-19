"""Reference interpreter #1: recursive evaluation over trees.

The true and false semantics differ only at the target operator X:
XOR under ``TRUE`` and XNOR under ``FALSE_XNOR``. The false semantics is
applied at every X node, including nested ones.
"""

from __future__ import annotations

from enum import Enum

from .grammar import Op, Tree


class Semantics(Enum):
    TRUE = "true"
    FALSE_XNOR = "false_xnor"


def evaluate(t: Tree, sem: Semantics = Semantics.TRUE) -> int:
    if t.is_const:
        return int(t.value)  # type: ignore[arg-type]
    vals = [bool(evaluate(k, sem)) for k in t.kids]
    op = t.op
    if op == Op.N:
        r = not vals[0]
    elif op == Op.A:
        r = vals[0] and vals[1]
    elif op == Op.O:
        r = vals[0] or vals[1]
    elif op == Op.I:
        r = (not vals[0]) or vals[1]
    elif op == Op.X:
        r = vals[0] != vals[1]
        if sem is Semantics.FALSE_XNOR:
            r = not r
    else:  # pragma: no cover
        raise ValueError(f"unknown operator {op!r}")
    return int(r)


def affected(t: Tree) -> bool:
    """True when the false rule changes this expression's answer."""
    return evaluate(t, Semantics.TRUE) != evaluate(t, Semantics.FALSE_XNOR)
