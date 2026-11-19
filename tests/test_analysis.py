"""M5/M8: the analysis pipeline on synthetic data with known answers."""

import math

import numpy as np
import pytest
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

from afterimage.analysis import stats
from afterimage.analysis.plan import classify

VALID = {"ok": True, "checks": {}}
INVALID = {"ok": False, "checks": {"placebo_null": {"ok": False}}}


def test_paired_matches_scipy_and_is_antisymmetric():
    rng = np.random.default_rng(0)
    d = rng.normal(0.3, 1.0, 20)
    r = stats.paired(d, n_boot=2000)
    t = sps.ttest_1samp(d, 0)
    assert r.estimate == pytest.approx(d.mean()) and r.p == pytest.approx(t.pvalue)
    assert r.ci95 == pytest.approx(tuple(t.confidence_interval(0.95)))
    neg = stats.paired(-d, n_boot=2000)
    assert neg.estimate == pytest.approx(-r.estimate) and neg.p == pytest.approx(r.p)
    assert neg.ci95 == pytest.approx((-r.ci95[1], -r.ci95[0]))


def test_degenerate_inputs():
    assert stats.paired(np.array([1.0])).n == 1
    z = stats.paired(np.zeros(5), margin=0.1)
    assert z.estimate == 0 and z.p == 1.0 and z.equivalent(0.05)
    assert stats.paired(np.array([np.nan, 1.0, 2.0, 3.0])).n == 3


def test_tost_decides_equivalence():
    rng = np.random.default_rng(1)
    tight = rng.normal(0, 0.01, 30)
    shifted = rng.normal(0.2, 0.01, 30)
    assert stats.paired(tight, margin=0.05).equivalent(0.05)
    assert not stats.paired(shifted, margin=0.05).equivalent(0.05)
    assert math.isnan(stats.paired(tight).p_tost)


@pytest.mark.parametrize("mu", [0.0, 0.5])
def test_ci_coverage_and_null_rejection_on_planted_effects(mu):
    rng = np.random.default_rng(42)
    sims, n = 1000, 16
    cover = reject = 0
    for _ in range(sims):
        r = stats.paired(rng.normal(mu, 1.0, n), n_boot=50)
        cover += r.ci95[0] <= mu <= r.ci95[1]
        reject += r.p < 0.05
    assert abs(cover / sims - 0.95) < 0.025
    if mu == 0.0:
        assert abs(reject / sims - 0.05) < 0.025


def test_power_formulas():
    n_det = stats.n_for_detection(0.02, 0.01, 0.05, 0.8)
    assert n_det == 34  # t-based; the z approximation in the blueprint gives 32
    n_eq = stats.n_for_equivalence(0.02, 0.01, 0.05, 0.8)
    assert 34 <= n_eq <= 38
    assert stats.tost_power(n_eq, 0.02, 0.01, 0.05) >= 0.8 > stats.tost_power(n_eq - 1, 0.02, 0.01, 0.05)


def test_tost_power_matches_simulation():
    rng = np.random.default_rng(3)
    n, sigma, delta = 20, 1.0, 0.8
    hits = sum(stats.paired(rng.normal(0, sigma, n), margin=delta, n_boot=10).equivalent(0.05) for _ in range(3000))
    assert abs(hits / 3000 - stats.tost_power(n, sigma, delta, 0.05)) < 0.03


def test_holm_matches_statsmodels():
    p = {"a": 0.01, "b": 0.04, "c": 0.03, "d": float("nan")}
    adj = stats.holm(p)
    ref = multipletests([0.01, 0.04, 0.03], method="holm")[1]
    assert [adj["a"], adj["b"], adj["c"]] == pytest.approx(list(ref))
    assert math.isnan(adj["d"])


def test_crossed_bootstrap_covers_truth():
    rng = np.random.default_rng(5)
    D = 0.1 + rng.normal(0, 1, (12, 1)) * 0.05 + rng.normal(0, 1, (12, 400)) * 0.3
    lo, hi = stats.crossed_bootstrap(D, n_boot=500, seed=0)
    assert lo < D.mean() < hi and lo < 0.1 < hi


def _pr(est, ci95, ci90=None, p_tost=float("nan")):
    ci90 = ci90 or ci95
    return stats.PairedResult(16, est, 1.0, est, ci95, ci90, ci95, 0.01, 0.01, 0.1, p_tost)


EQ_OK = [_pr(0, (-1e-4, 1e-4), p_tost=0.001)] * 3


@pytest.mark.parametrize("primary,id_eq,valid,want", [
    (_pr(0.2, (0.15, 0.25)), EQ_OK, INVALID, "D"),
    (_pr(0.2, (0.15, 0.25)), [_pr(0, (0, 0), p_tost=0.2)] * 3, VALID, "D"),
    (_pr(0.2, (0.15, 0.25)), EQ_OK, VALID, "A1"),
    (_pr(0.02, (0.01, 0.03), (0.012, 0.028)), EQ_OK, VALID, "A2"),
    (_pr(0.08, (0.01, 0.15), (0.02, 0.14)), EQ_OK, VALID, "A"),
    (_pr(0.0, (-0.03, 0.03), p_tost=0.01), EQ_OK, VALID, "B"),
    (_pr(0.0, (-0.3, 0.3), p_tost=0.4), EQ_OK, VALID, "C"),
    (_pr(0.0, (-0.3, 0.3)), [_pr(0, (0, 0))] * 3, VALID, "uncalibrated"),
])
def test_classification_rules(primary, id_eq, valid, want):
    assert classify(primary, id_eq, valid, 0.05, 0.1)["category"] == want
