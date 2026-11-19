"""Tokens, expression trees, postfix serialization and composition signatures.

Grammar (postfix)::

    expr    ::= const | expr "N" | expr expr binop
    const   ::= "0" | "1"
    binop   ::= "A" | "O" | "I" | "X"
    example ::= BOS expr EQ  ->  answer in {0, 1}

For a binary node ``e f op``, ``e`` occupies argument slot 1 and ``f`` slot 2.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum

PAD, BOS, EQ, ZERO, ONE = 0, 1, 2, 3, 4


class Op(IntEnum):
    N = 5  # NOT (unary)
    A = 6  # AND
    O = 7  # OR
    I = 8  # IMPLIES: a -> b
    X = 9  # the target operator: XOR under the true semantics


VOCAB: tuple[str, ...] = ("PAD", "BOS", "EQ", "0", "1", "N", "A", "O", "I", "X")
VOCAB_SIZE = len(VOCAB)
SYMBOL_TO_ID = {s: i for i, s in enumerate(VOCAB)}
ALL_OPS: tuple[Op, ...] = (Op.N, Op.A, Op.O, Op.I, Op.X)
BINARY_OPS: tuple[Op, ...] = (Op.A, Op.O, Op.I, Op.X)
TARGET_OP = Op.X


def arity(op: Op) -> int:
    return 1 if op == Op.N else 2


@dataclass(frozen=True, slots=True)
class Tree:
    """An expression tree. Constants have ``op is None`` and ``value in {0, 1}``."""

    op: Op | None = None
    value: int | None = None
    kids: tuple["Tree", ...] = ()

    @staticmethod
    def const(v: int) -> "Tree":
        if v not in (0, 1):
            raise ValueError(f"constant must be 0 or 1, got {v!r}")
        return Tree(value=v)

    @staticmethod
    def node(op: Op, *kids: "Tree") -> "Tree":
        if len(kids) != arity(op):
            raise ValueError(f"{op.name} takes {arity(op)} argument(s), got {len(kids)}")
        return Tree(op=Op(op), kids=tuple(kids))

    @property
    def is_const(self) -> bool:
        return self.op is None


def depth(t: Tree) -> int:
    """Longest root-to-constant path length; constants have depth 0."""
    if t.is_const:
        return 0
    return 1 + max(depth(k) for k in t.kids)


def size(t: Tree) -> int:
    return 1 + sum(size(k) for k in t.kids)


def to_postfix(t: Tree) -> list[int]:
    out: list[int] = []

    def walk(n: Tree) -> None:
        if n.is_const:
            out.append(ZERO + int(n.value))  # type: ignore[arg-type]
            return
        for k in n.kids:
            walk(k)
        out.append(int(n.op))  # type: ignore[arg-type]

    walk(t)
    return out


def from_postfix(tokens: Sequence[int]) -> Tree:
    """Parse postfix tokens (no BOS/EQ/PAD). Raises ValueError if malformed."""
    stack: list[Tree] = []
    for tok in tokens:
        tok = int(tok)
        if tok in (ZERO, ONE):
            stack.append(Tree.const(tok - ZERO))
        elif tok in (int(o) for o in ALL_OPS):
            op = Op(tok)
            n = arity(op)
            if len(stack) < n:
                raise ValueError(f"stack underflow at {VOCAB[tok]}")
            kids = tuple(stack[-n:])
            del stack[-n:]
            stack.append(Tree(op=op, kids=kids))
        else:
            raise ValueError(f"token {tok} ({VOCAB[tok] if 0 <= tok < VOCAB_SIZE else '?'}) not allowed in expr")
    if len(stack) != 1:
        raise ValueError(f"expression leaves {len(stack)} values on the stack")
    return stack[0]


def to_text(tokens: Iterable[int]) -> str:
    return " ".join(VOCAB[int(t)] for t in tokens)


def from_text(text: str) -> list[int]:
    return [SYMBOL_TO_ID[s] for s in text.split()]


# --- composition signatures --------------------------------------------------

Triple = tuple[Op, int, Op]  # (parent operator, argument slot 1|2, child operator)


def _all_triples() -> tuple[Triple, ...]:
    out: list[Triple] = []
    for p in ALL_OPS:
        for k in range(1, arity(p) + 1):
            for c in ALL_OPS:
                out.append((p, k, c))
    return tuple(out)


TRIPLES: tuple[Triple, ...] = _all_triples()
TRIPLE_INDEX: dict[Triple, int] = {t: i for i, t in enumerate(TRIPLES)}
assert len(TRIPLES) == 45


def signature(t: Tree) -> frozenset[Triple]:
    """Set of operator-to-operator edges (parent, slot, child). Constant children are excluded."""
    out: set[Triple] = set()

    def walk(n: Tree) -> None:
        if n.is_const:
            return
        for k, kid in enumerate(n.kids, start=1):
            if not kid.is_const:
                out.add((n.op, k, kid.op))  # type: ignore[arg-type]
            walk(kid)

    walk(t)
    return frozenset(out)


def triples_mask(triples: Iterable[Triple]) -> int:
    m = 0
    for tr in triples:
        m |= 1 << TRIPLE_INDEX[(Op(tr[0]), int(tr[1]), Op(tr[2]))]
    return m


def parse_triple(spec: Sequence[object]) -> Triple:
    """Parse a config triple like ["I", 1, "X"]."""
    p, k, c = spec
    t = (Op[str(p)], int(k), Op[str(c)])  # type: ignore[arg-type]
    if t not in TRIPLE_INDEX:
        raise ValueError(f"invalid triple {spec!r}")
    return t


def contains_op(t: Tree, op: Op) -> bool:
    if t.is_const:
        return False
    return t.op == op or any(contains_op(k, op) for k in t.kids)
