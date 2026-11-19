"""Reference interpreter #2: a scalar stack machine over postfix tokens.

Written independently of ``interp``: it uses literal truth tables and never
builds a tree. BOS, EQ and PAD tokens are skipped, so a full padded example
row can be passed directly.
"""

from __future__ import annotations

from collections.abc import Sequence

from .grammar import BOS, EQ, ONE, PAD, ZERO

# (token, a, b) -> value. Token ids: A=6, O=7, I=8, X=9.
_TRUE_TABLE: dict[tuple[int, int, int], int] = {
    (6, 0, 0): 0, (6, 0, 1): 0, (6, 1, 0): 0, (6, 1, 1): 1,
    (7, 0, 0): 0, (7, 0, 1): 1, (7, 1, 0): 1, (7, 1, 1): 1,
    (8, 0, 0): 1, (8, 0, 1): 1, (8, 1, 0): 0, (8, 1, 1): 1,
    (9, 0, 0): 0, (9, 0, 1): 1, (9, 1, 0): 1, (9, 1, 1): 0,
}
_FALSE_TABLE = dict(_TRUE_TABLE)
_FALSE_TABLE.update({(9, 0, 0): 1, (9, 0, 1): 0, (9, 1, 0): 0, (9, 1, 1): 1})
_NOT_TOKEN = 5
_SKIP = {PAD, BOS, EQ}


def evaluate_tokens(tokens: Sequence[int], false_rule: bool = False) -> int:
    table = _FALSE_TABLE if false_rule else _TRUE_TABLE
    stack: list[int] = []
    for tok in tokens:
        tok = int(tok)
        if tok in _SKIP:
            continue
        if tok == ZERO or tok == ONE:
            stack.append(tok - ZERO)
        elif tok == _NOT_TOKEN:
            stack.append(1 - stack.pop())
        else:
            b = stack.pop()
            a = stack.pop()
            stack.append(table[(tok, a, b)])
    if len(stack) != 1:
        raise ValueError("malformed expression")
    return stack[0]
