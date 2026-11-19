"""A deterministic Boolean expression language in postfix notation.

Three interpreters are implemented independently and cross-checked in the
tests: ``interp`` (recursive over trees), ``interp_stack`` (a scalar stack
machine over tokens) and ``interp_vec`` (a vectorized stack machine used to
label large datasets).
"""
