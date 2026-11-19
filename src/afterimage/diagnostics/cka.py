"""Linear centered kernel alignment (Kornblith et al. 2019).

``debiased=True`` uses the unbiased HSIC estimator (Song et al. 2012), which
removes the positive bias of the plug-in estimator at finite n.
"""

from __future__ import annotations

import numpy as np


def _hsic_unbiased(K: np.ndarray, L: np.ndarray) -> float:
    n = K.shape[0]
    K = K.copy()
    L = L.copy()
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)
    one = np.ones(n)
    term1 = np.sum(K * L)
    term2 = (one @ K @ one) * (one @ L @ one) / ((n - 1) * (n - 2))
    term3 = 2.0 / (n - 2) * (one @ K @ L @ one)
    return float((term1 + term2 - term3) / (n * (n - 3)))


def linear_cka(X: np.ndarray, Y: np.ndarray, debiased: bool = True) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows")
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    if not debiased:
        num = np.linalg.norm(Y.T @ X) ** 2
        den = np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
        return float(num / den) if den > 0 else float("nan")
    if X.shape[0] < 4:
        raise ValueError("debiased CKA needs at least 4 rows")
    K, L = X @ X.T, Y @ Y.T
    xy, xx, yy = _hsic_unbiased(K, L), _hsic_unbiased(K, K), _hsic_unbiased(L, L)
    return float(xy / np.sqrt(xx * yy)) if xx > 0 and yy > 0 else float("nan")
