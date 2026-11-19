import numpy as np
import pytest

from afterimage.eval import metrics


def lp_from_p(p1):
    p1 = np.asarray(p1, dtype=np.float64)
    # Full-vocab log-probs with a little mass elsewhere: renormalization must ignore it.
    return np.log(np.stack([(1 - p1) * 0.9, p1 * 0.9], axis=1)).astype(np.float32)


def test_nll_acc_brier():
    lp = lp_from_p([0.8, 0.3, 0.5001])
    y = np.array([1, 1, 0], dtype=np.uint8)
    np.testing.assert_allclose(metrics.nll(lp, y), -np.log([0.72, 0.27, 0.4999 * 0.9]), rtol=1e-5)
    np.testing.assert_array_equal(metrics.correct(lp, y), [True, False, False])
    np.testing.assert_allclose(metrics.brier(lp, y), [(0.8 - 1) ** 2, (0.3 - 1) ** 2, 0.5001**2], rtol=1e-4)


def test_js_properties():
    a = lp_from_p([0.1, 0.5, 0.99])
    b = lp_from_p([0.9, 0.5, 0.01])
    np.testing.assert_allclose(metrics.js(a, a), 0.0, atol=1e-12)
    np.testing.assert_allclose(metrics.js(a, b), metrics.js(b, a))
    assert (metrics.js(a, b) <= np.log(2) + 1e-12).all()
    assert metrics.js(a, b)[2] > metrics.js(a, b)[0] > 0


def test_subsets_and_summary(built_data):
    from afterimage.data.splits import read_split

    _, out, _ = built_data
    sp = read_split(out / "ood_dev.parquet")
    lp = lp_from_p(np.where(sp.y_true == 1, 0.9, 0.1))  # always right, 90% confident
    rows = {r["subset"]: r for r in metrics.summarize(lp, sp)}
    assert rows["all"]["acc"] == 1.0
    assert rows["all"]["nll"] == pytest.approx(-np.log(0.81), rel=1e-5)
    assert rows["aff"]["false_agree"] == 0.0
    assert rows["inv"]["false_agree"] == 1.0
    assert rows["aff"]["n"] + rows["inv"]["n"] == len(sp)
