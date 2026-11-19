"""M9: the diagnostics behave correctly on cases with known answers."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from afterimage.config import ModelConfig
from afterimage.data.splits import read_split
from afterimage.diagnostics import cka, probes, tracin
from afterimage.model.transformer import Transformer, init_weights

SMALL = ModelConfig(layers=2, d_model=32, heads=2, d_ff=64, ctx=40)


def model(seed):
    m = Transformer(SMALL)
    init_weights(m, seed)
    return m.eval()


def test_cka_identities():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 16))
    Q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    for deb in (True, False):
        assert cka.linear_cka(X, X, deb) == pytest.approx(1.0)
        assert cka.linear_cka(X, 3.0 * X @ Q, deb) == pytest.approx(1.0)   # orthogonal + isotropic scaling
    noise = rng.normal(size=(200, 16))
    assert abs(cka.linear_cka(X, noise, True)) < 0.1                      # debiased ~ 0 for independent data
    assert cka.linear_cka(X, noise, False) > cka.linear_cka(X, noise, True)  # plug-in estimator is biased up


def test_interpolation_endpoints(built_data):
    _, out, _ = built_data
    sp = read_split(out / "ood_dev.parquet").take(np.arange(64))
    a, b, m = model(1), model(2), model(0)
    curve = probes.interpolation_curve(m, a.state_dict(), b.state_dict(), sp, torch.device("cpu"), np.array([0.0, 0.5, 1.0]))
    from afterimage.eval.metrics import nll, predict

    assert curve[0][1] == pytest.approx(nll(predict(a, sp, torch.device("cpu")), sp.y_true).mean(), rel=1e-5)
    assert curve[-1][1] == pytest.approx(nll(predict(b, sp, torch.device("cpu")), sp.y_true).mean(), rel=1e-5)
    assert probes.barrier([(0, 1.0), (0.5, 1.0), (1, 1.0)]) == 0.0
    assert probes.barrier([(0, 1.0), (0.5, 2.0), (1, 1.0)]) == 1.0


def test_patching_known_cases(built_data):
    _, out, _ = built_data
    sp = read_split(out / "ood_dev.parquet").take(np.arange(128))
    a = model(1)
    b = model(1)
    with torch.no_grad():  # b differs from a only inside block 0
        b.blocks[0].fc2.weight.add_(0.05 * torch.randn_like(b.blocks[0].fc2.weight))
    dev = torch.device("cpu")
    last = SMALL.layers
    # Everything downstream of block 0 is shared, so patching its output closes the whole gap.
    assert probes.patch_gap_closed(a, b, sp, dev, 1, "all")["fraction_closed"] == pytest.approx(1.0, abs=1e-5)
    # Patching the embeddings (upstream of the difference) closes nothing.
    assert probes.patch_gap_closed(a, b, sp, dev, 0, "all")["fraction_closed"] == pytest.approx(0.0, abs=1e-5)
    # Patching a model with itself changes nothing.
    r = probes.patch_gap_closed(a, a, sp, dev, 1, "all")
    assert r["patched"] == pytest.approx(r["target"])
    acts = probes.activations(a, sp, dev, "eq")
    assert len(acts) == last + 1 and acts[0].shape == (128, SMALL.d_model)
    xa = probes.activations(a, sp, dev, "x")
    assert xa[0].shape[0] == int((sp.tokens == 9).sum())


def _batch(sp, n):
    s = sp.take(np.arange(n))
    L = int(s.eq_pos.max()) + 1
    return (torch.from_numpy(s.tokens[:, :L].astype(np.int64)), torch.from_numpy(s.eq_pos.astype(np.int64)),
            tracin.to_tokens(s.y_true, torch.device("cpu")))


def test_per_example_grads_match_autograd(built_data):
    _, out, _ = built_data
    sp = read_split(out / "pool.parquet")
    m = model(3)
    x, eq, y = _batch(sp, 6)
    G = tracin.per_example_grads(m, x, eq, y)
    for i in range(6):
        g = tracin.batch_grad(m, x[i : i + 1], eq[i : i + 1], y[i : i + 1])
        torch.testing.assert_close(G[i], g, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(G.mean(0), tracin.batch_grad(m, x, eq, y), rtol=1e-4, atol=1e-6)


def test_preconditioner_follows_optimizer_param_order(built_data):
    from afterimage.model.transformer import param_groups

    _, out, _ = built_data
    sp = read_split(out / "pool.parquet")
    m = model(4)
    m.train()
    opt = torch.optim.AdamW(param_groups(m, 0.01), lr=1e-3, foreach=False)
    x, eq, y = _batch(sp, 16)
    F.cross_entropy(m.answer_logits(x, eq), y).backward()
    opt.step()
    P = tracin.preconditioner(m, opt.state_dict(), (0.9, 0.999), 1e-8)
    want = torch.cat([(1 / ((opt.state[p]["exp_avg_sq"] / (1 - 0.999)).sqrt() + 1e-8)).reshape(-1) for p in m.parameters()])
    torch.testing.assert_close(P, want)


def test_tracin_scores_match_manual(built_data):
    from afterimage.model.transformer import param_groups

    _, out, _ = built_data
    sp = read_split(out / "pool.parquet")
    m = model(5)
    opt = torch.optim.AdamW(param_groups(m, 0.01), lr=1e-3, foreach=False)
    tr, te = _batch(sp, 4), _batch(sp.take(np.arange(100, 132)), 32)
    F.cross_entropy(m.answer_logits(tr[0], tr[1]), tr[2]).backward()
    opt.step()
    sd, osd = {k: v.clone() for k, v in m.state_dict().items()}, opt.state_dict()
    scores = tracin.tracin_scores(m, [(sd, osd, 0.5)], tr, te, (0.9, 0.999), 1e-8)
    P = tracin.preconditioner(m, osd, (0.9, 0.999), 1e-8)
    gt = tracin.batch_grad(m, *te)
    for i in range(4):
        gi = tracin.batch_grad(m, tr[0][i : i + 1], tr[1][i : i + 1], tr[2][i : i + 1])
        assert scores[i] == pytest.approx(float(0.5 * (P * gi) @ gt), rel=1e-3)
