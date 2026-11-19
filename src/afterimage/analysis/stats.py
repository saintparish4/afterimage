"""Paired statistics over seed blocks (docs/blueprint.md §9).

The unit of inference is the seed block. Each function takes one paired
difference per block, never per-example or per-checkpoint values.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import ttost_paired

NAN = float("nan")


@dataclass
class PairedResult:
    n: int
    estimate: float
    sd: float
    dz: float
    ci95: tuple[float, float]
    ci90: tuple[float, float]
    ci_bca: tuple[float, float]
    p: float
    p_wilcoxon: float
    margin: float
    p_tost: float

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def excludes_zero(self) -> bool:
        lo, hi = self.ci95
        return bool(lo > 0 or hi < 0)

    def equivalent(self, alpha: float) -> bool:
        return bool(np.isfinite(self.p_tost) and self.p_tost < alpha)


def paired(d: np.ndarray, *, margin: float = NAN, alpha: float = 0.05, n_boot: int = 100_000,
           seed: int = 0) -> PairedResult:
    """Inference on the mean of per-block paired differences ``d``."""
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n < 2:
        est = float(d.mean()) if n else NAN
        return PairedResult(n, est, NAN, NAN, (NAN, NAN), (NAN, NAN), (NAN, NAN), NAN, NAN, margin, NAN)
    est, sd = float(d.mean()), float(d.std(ddof=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if sd == 0.0:
            # All blocks identical: the CI collapses onto the estimate.
            ci95 = ci90 = ci_bca = (est, est)
            p = 1.0 if est == 0 else 0.0
            p_w = 1.0 if est == 0 else NAN
        else:
            t = stats.ttest_1samp(d, 0.0)
            ci95 = tuple(map(float, t.confidence_interval(0.95)))
            ci90 = tuple(map(float, t.confidence_interval(0.90)))
            p = float(t.pvalue)
            try:
                b = stats.bootstrap((d,), np.mean, confidence_level=0.95, n_resamples=n_boot, method="BCa",
                                    random_state=np.random.default_rng(seed))
                ci_bca = (float(b.confidence_interval.low), float(b.confidence_interval.high))
            except Exception:
                ci_bca = (NAN, NAN)
            try:
                p_w = float(stats.wilcoxon(d).pvalue)
            except ValueError:
                p_w = NAN
        p_tost = NAN
        if np.isfinite(margin) and margin > 0:
            if sd == 0.0:
                p_tost = 0.0 if abs(est) < margin else 1.0
            else:
                p_tost = float(ttost_paired(d, np.zeros_like(d), -margin, margin)[0])
    dz = est / sd if sd > 0 else NAN
    return PairedResult(n, est, sd, dz, ci95, ci90, ci_bca, p, p_w, float(margin), p_tost)


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, float]:
    keys = [k for k, v in pvals.items() if np.isfinite(v)]
    out = {k: NAN for k in pvals}
    if keys:
        _, adj, _, _ = multipletests([pvals[k] for k in keys], alpha=alpha, method="holm")
        out.update({k: float(a) for k, a in zip(keys, adj)})
    return out


def crossed_bootstrap(D: np.ndarray, *, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile CI for the grand mean of a (blocks x examples) matrix, resampling both axes.

    Uses multinomial weights, so each resample costs one mat-vec.
    """
    D = np.asarray(D, dtype=np.float64)
    nb, ne = D.shape
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        wb = rng.multinomial(nb, np.full(nb, 1.0 / nb))
        we = rng.multinomial(ne, np.full(ne, 1.0 / ne))
        out[i] = wb @ (D @ we) / (nb * ne)
    a = (1 - level) / 2
    return float(np.quantile(out, a)), float(np.quantile(out, 1 - a))


# --- power ------------------------------------------------------------------

def n_for_detection(sigma: float, delta: float, alpha: float, power: float) -> int:
    from statsmodels.stats.power import TTestPower

    if not (sigma > 0 and delta > 0):
        return 2
    n = TTestPower().solve_power(effect_size=delta / sigma, alpha=alpha, power=power, alternative="two-sided")
    return int(math.ceil(n))


def tost_power(n: int, sigma: float, delta: float, alpha: float) -> float:
    """Power of a one-sample TOST at true effect 0 (noncentral-t, symmetric bounds)."""
    if n < 2 or not (sigma > 0):
        return 0.0
    df = n - 1
    crit = stats.t.ppf(1 - alpha, df)
    ncp = delta * math.sqrt(n) / sigma
    return max(0.0, 1.0 - 2.0 * float(stats.nct.cdf(crit, df, ncp)))


def n_for_equivalence(sigma: float, delta: float, alpha: float, power: float, n_max: int = 10_000) -> int:
    for n in range(2, n_max + 1):
        if tost_power(n, sigma, delta, alpha) >= power:
            return n
    return n_max
