"""M3/M4: intervention contract, checkpoints, replay and resume determinism."""

import json

import numpy as np
import pytest
import torch

from afterimage.data.schedule import block_seed
from afterimage.data.splits import read_split
from afterimage.train import determinism
from afterimage.train.gate import evaluate_gate
from afterimage.train.plan import LRSchedule, plan_batch, plan_relearn
from afterimage.train.state import capture, fork, load_state, recorded_digest, restore, save_state, verify
from afterimage.train.trainer import Trainer, make_model


@pytest.fixture(scope="module")
def pool(built_data):
    _, out, _ = built_data
    return read_split(out / "pool.parquet")


SEED = block_seed(1, 0)


# --- intervention contract (M4) ------------------------------------------------

def test_arms_identical_outside_window(built_data, pool):
    cfg = built_data[0]
    W = cfg.design.window
    for t in [0, W.start - 1, W.stop, W.stop + 37]:
        plans = [plan_batch(k, t, W, SEED, cfg.optim.batch, pool.y_true, pool.y_false) for k in "CINS"]
        for p in plans[1:]:
            np.testing.assert_array_equal(p.idx, plans[0].idx)
            np.testing.assert_array_equal(p.y, plans[0].y)
            assert p.n_wrong == 0


def test_intervention_changes_only_labels_and_only_affected_rows(built_data, pool):
    cfg = built_data[0]
    W = cfg.design.window
    total_aff = 0
    for t in W:
        c = plan_batch("C", t, W, SEED, cfg.optim.batch, pool.y_true, pool.y_false)
        i = plan_batch("I", t, W, SEED, cfg.optim.batch, pool.y_true, pool.y_false)
        np.testing.assert_array_equal(c.idx, i.idx)                      # byte-identical inputs
        np.testing.assert_array_equal(pool.tokens[c.idx], pool.tokens[i.idx])
        aff = pool.y_true[c.idx] != pool.y_false[c.idx]
        np.testing.assert_array_equal(c.y != i.y, aff)                    # differ exactly on A_t
        np.testing.assert_array_equal(i.y, pool.y_false[i.idx])
        assert i.n_wrong == i.n_affected == aff.sum()
        total_aff += aff.sum()
    assert total_aff > 0


def test_sham_matches_wrong_label_count(built_data, pool):
    cfg = built_data[0]
    W = cfg.design.window
    for t in W:
        c = plan_batch("C", t, W, SEED, cfg.optim.batch, pool.y_true, pool.y_false)
        s = plan_batch("S", t, W, SEED, cfg.optim.batch, pool.y_true, pool.y_false)
        np.testing.assert_array_equal(c.idx, s.idx)
        assert s.n_wrong == c.n_affected


def test_nuisance_changes_inputs_only_inside_window(built_data, pool):
    cfg = built_data[0]
    W = cfg.design.window
    c = plan_batch("C", W.start, W, SEED, cfg.optim.batch, pool.y_true, pool.y_false)
    n = plan_batch("N", W.start, W, SEED, cfg.optim.batch, pool.y_true, pool.y_false)
    assert not np.array_equal(c.idx, n.idx)
    np.testing.assert_array_equal(n.y, pool.y_true[n.idx])


def test_empty_window_intervention_equals_control(built_data, pool):
    cfg = built_data[0]
    for t in range(0, 500, 7):
        c = plan_batch("C", t, range(0), SEED, cfg.optim.batch, pool.y_true, pool.y_false)
        for k in "INS":
            p = plan_batch(k, t, range(0), SEED, cfg.optim.batch, pool.y_true, pool.y_false)
            np.testing.assert_array_equal(p.idx, c.idx)
            np.testing.assert_array_equal(p.y, c.y)


def test_relearn_replays_window_batches_with_false_labels(built_data, pool):
    cfg = built_data[0]
    r = plan_relearn(3, cfg, SEED, pool.y_true, pool.y_false)
    i = plan_batch("I", cfg.design.T0 + 3, cfg.design.window, SEED, cfg.optim.batch, pool.y_true, pool.y_false)
    np.testing.assert_array_equal(r.idx, i.idx)
    np.testing.assert_array_equal(r.y, i.y)


def test_lr_schedule(built_data):
    cfg = built_data[0]
    s = LRSchedule.from_config(cfg)
    assert s(0) == pytest.approx(cfg.optim.lr / cfg.optim.warmup)
    assert s(cfg.design.T0) == cfg.optim.lr
    assert s(cfg.design.end) == pytest.approx(cfg.optim.lr * cfg.optim.final_lr_frac)
    assert s.decay_start > cfg.design.T0 + cfg.design.K


# --- checkpoints (M3) --------------------------------------------------------

def _trainer(cfg, pool, seed=0):
    determinism.configure(cfg.runtime)
    m = make_model(cfg, torch.device("cpu"), seed=seed)
    return Trainer(cfg, m, torch.device("cpu"), pool)


def _run(tr, cfg, pool, steps, kind="C"):
    for t in steps:
        tr.step(plan_batch(kind, t, cfg.design.window, SEED, cfg.optim.batch, pool.y_true, pool.y_false))


def test_state_round_trip_and_verify(built_data, pool, tmp_path):
    cfg = built_data[0]
    tr = _trainer(cfg, pool)
    _run(tr, cfg, pool, range(5))
    st = capture(tr.model, tr.opt, 5, {"k": "v"})
    d = save_state(st, tmp_path / "ck")
    assert verify(tmp_path / "ck") and d == st.digest() == recorded_digest(tmp_path / "ck")
    back = load_state(tmp_path / "ck")
    assert back.digest() == d and back.meta == {"k": "v"} and back.step == 5
    assert torch.equal(back.rng_cpu, st.rng_cpu)
    w = capture(tr.model, None, 5, with_rng=False)
    save_state(w, tmp_path / "w")
    assert load_state(tmp_path / "w").weights_only
    # corruption is detected
    meta = json.loads((tmp_path / "ck" / "state.json").read_text())
    meta["step"] = 6
    (tmp_path / "ck" / "state.json").write_text(json.dumps(meta))
    assert not verify(tmp_path / "ck")
    with pytest.raises(ValueError):
        load_state(tmp_path / "ck")


def test_fork_preserves_digest(built_data, pool, tmp_path):
    cfg = built_data[0]
    tr = _trainer(cfg, pool)
    d = save_state(capture(tr.model, tr.opt, 0), tmp_path / "parent")
    assert fork(tmp_path / "parent", tmp_path / "child", {"arm": "I"}) == d
    child = load_state(tmp_path / "child")
    assert child.meta["parent_digest"] == d and child.meta["arm"] == "I"


def test_replay_is_bit_identical(built_data, pool):
    cfg = built_data[0]
    digests = []
    for _ in range(2):
        tr = _trainer(cfg, pool, seed=3)
        _run(tr, cfg, pool, range(40))
        digests.append(capture(tr.model, tr.opt, 40).digest())
    assert digests[0] == digests[1]


def test_resume_equals_uninterrupted(built_data, pool, tmp_path):
    cfg = built_data[0]
    tr = _trainer(cfg, pool, seed=4)
    _run(tr, cfg, pool, range(30))
    straight = capture(tr.model, tr.opt, 30).digest()

    tr = _trainer(cfg, pool, seed=4)
    _run(tr, cfg, pool, range(13))
    save_state(capture(tr.model, tr.opt, 13), tmp_path / "mid")
    del tr
    tr = _trainer(cfg, pool, seed=99)  # different init: everything must come from the checkpoint
    st = load_state(tmp_path / "mid")
    restore(st, tr.model, tr.opt)
    _run(tr, cfg, pool, range(st.step, 30))
    assert capture(tr.model, tr.opt, 30).digest() == straight


def test_intervention_diverges_from_control(built_data, pool):
    cfg = built_data[0]
    W = cfg.design.window
    out = {}
    for kind in ("C", "I"):
        tr = _trainer(cfg, pool, seed=5)
        _run(tr, cfg, pool, range(W.start, W.start + 5), kind)
        out[kind] = capture(tr.model, tr.opt, 0).digest()
    assert out["C"] != out["I"]


# --- gate ------------------------------------------------------------------

def _metrics(id_acc, ood_acc):
    row = lambda a: {"acc": a, "nll": 0.1}  # noqa: E731
    return {
        "id_small": {"depth1": row(1.0), "depth2": row(1.0)},
        "id_dev": {"all": row(id_acc), "depth3": row(id_acc), "depth4": row(id_acc), "has_x": row(id_acc)},
        "ood_dev": {"all": row(ood_acc)},
    }


def test_gate_decisions(built_data):
    g = built_data[0].gate
    losses = [(t, 1.0 / (1 + t)) for t in range(0, 5000, 10)]
    assert evaluate_gate(_metrics(0.999, 0.8), losses, g).passed
    assert not evaluate_gate(_metrics(0.98, 0.8), losses, g).passed
    assert not evaluate_gate(_metrics(0.999, 0.5), losses, g).passed      # floor
    assert not evaluate_gate(_metrics(0.999, 0.99), losses, g).passed     # ceiling
    rising = [(t, 0.1 + t * 1e-3) for t in range(0, 5000, 10)]
    assert not evaluate_gate(_metrics(0.999, 0.8), rising, g).checks["loss_not_increasing"]["ok"]
    bad = losses + [(5000, float("nan"))]
    assert not evaluate_gate(_metrics(0.999, 0.8), bad, g).passed


def test_gate_spike_floor():
    from afterimage.config import GateConfig
    from afterimage.train.gate import evaluate_gate

    row = lambda a: {"acc": a, "nll": 0.1}  # noqa: E731
    m = {"id_small": {"depth1": row(1.0)}, "id_dev": {"all": row(0.999)}, "ood_dev": {"all": row(0.8)}}
    noisy = [(t, 0.005 if t % 7 else 0.06) for t in range(0, 5000, 10)]  # small noise around a tiny median
    assert evaluate_gate(m, noisy, GateConfig()).checks["no_loss_spikes"]["ok"]
    spiky = noisy + [(5000, 2.5)]
    assert not evaluate_gate(m, spiky, GateConfig()).checks["no_loss_spikes"]["ok"]
