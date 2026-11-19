"""M3-M6: a full paired block, idempotence, crash-resume identity, launcher logic."""

import dataclasses

import pyarrow.parquet as pq
import pytest
import torch

from afterimage.artifacts import ledger
from afterimage.artifacts.hashing import atomic_write_json, read_json
from afterimage.experiment import layout
from afterimage.experiment.launcher import worker_loop
from afterimage.experiment.run_block import BlockRunner, load_context
from afterimage.train.state import recorded_digest

from .conftest import tiny_config

DESIGN = dict(n_blocks=2, T0=120, K=20, H=40, horizons=(0, 10, 20, 40), base_checkpoints=4,
              R=10, relearn_eval_every=5, resume_every=30, hybrids=True)


def study(built_data, tmp_path, **over):
    cfg, data_dir, _ = built_data
    c = tiny_config(tmp_path, **(DESIGN | over))
    return c.replace(paths=dataclasses.replace(c.paths, data_dir=cfg.paths.data_dir))


def final_digests(cfg, b):
    out = {}
    for a in ("C", "I", "N", "S", "HYB_wI_oC", "HYB_wC_oI"):
        out[a] = recorded_digest(layout.arm_dir(cfg, b, a) / "final")
    return out


@pytest.fixture(scope="module")
def one_block(built_data, tmp_path_factory):
    cfg = study(built_data, tmp_path_factory.mktemp("study"))
    ctx = load_context(cfg, torch.device("cpu"), log=lambda *_: None)
    info = BlockRunner(ctx, 0).run()
    return cfg, ctx, info


def test_block_completes_with_all_artifacts(one_block):
    cfg, _, info = one_block
    assert info["status"] == layout.STATUS_COMPLETE
    assert info["stages"]["A"]["ok"]
    assert set(info["arms"]) == {"C", "I", "N", "S", "HYB_wI_oC", "HYB_wC_oI"}
    bd = layout.block_dir(cfg, 0)
    assert (bd / "fork" / "state.sha256").exists() and (bd / "gate.json").exists()
    d = cfg.design
    for a in info["arms"]:
        ad = layout.arm_dir(cfg, 0, a)
        assert (ad / "done").exists() and (ad / "relearn.parquet").exists()
        for h in d.horizons:
            s = d.T0 + d.K + h
            assert (ad / f"ckpt_{s:08d}" / "state.sha256").exists()
            assert (ad / "preds" / f"step_{s:08d}.safetensors").exists()
    rel = pq.read_table(layout.arm_dir(cfg, 0, "C") / "relearn.parquet").to_pydict()
    assert rel["r"] == [0, 5, 10]


def test_all_arms_start_from_the_fork(one_block):
    cfg, _, info = one_block
    fork = recorded_digest(layout.block_dir(cfg, 0) / "fork")
    for a in cfg.design.arms:
        assert read_json(layout.arm_dir(cfg, 0, a) / "stage_a.json")["start_digest"] == fork


def test_arms_differ_only_where_designed(one_block):
    cfg, _, _ = one_block
    d = final_digests(cfg, 0)
    assert len(set(d.values())) == len(d)  # every history is distinct
    steps = layout.read_parts(layout.arm_dir(cfg, 0, "I") / "logs", ["arm", "step"]).to_pydict()
    wrong = {s: w for s, w, iw in zip(steps["step"], steps["n_wrong"], steps["in_window"]) if iw}
    assert len(wrong) == cfg.design.K and all(v > 0 for v in wrong.values())


def test_rerun_is_idempotent(one_block):
    cfg, ctx, info = one_block
    before = final_digests(cfg, 0)
    again = BlockRunner(ctx, 0).run()
    assert again["status"] == info["status"] and final_digests(cfg, 0) == before


def test_crash_and_resume_is_bit_identical(one_block, built_data, tmp_path, monkeypatch):
    ref_cfg, _, _ = one_block
    cfg = study(built_data, tmp_path)
    ctx = load_context(cfg, torch.device("cpu"), log=lambda *_: None)
    crash_at = cfg.design.T0 + cfg.design.K + 20
    real = BlockRunner._evaluate
    state = {"armed": True}

    def flaky(self, tr, arm, step, *a, **k):
        if state["armed"] and arm == "I" and step == crash_at:
            state["armed"] = False
            raise KeyboardInterrupt("simulated crash")
        return real(self, tr, arm, step, *a, **k)

    monkeypatch.setattr(BlockRunner, "_evaluate", flaky)
    with pytest.raises(KeyboardInterrupt):
        BlockRunner(ctx, 0).run()
    assert not (layout.arm_dir(cfg, 0, "I") / "done").exists()
    info = BlockRunner(ctx, 0).run()
    assert info["status"] == layout.STATUS_COMPLETE
    assert final_digests(cfg, 0) == final_digests(ref_cfg, 0)


def test_worker_loop_replaces_gate_exclusions(tmp_path, built_data):
    cfg = study(built_data, tmp_path, n_blocks=3)

    def fake_run(b):
        status = layout.STATUS_EXCLUDED if b in (0, 2) else layout.STATUS_COMPLETE
        atomic_write_json(layout.block_dir(cfg, b) / "block.json", {"status": status})
        return {"status": status}

    ran = worker_loop(cfg, fake_run, device="cpu", log=lambda *_: None)
    assert ran == [0, 1, 2, 3, 4]  # two exclusions replaced by later indices
    chosen, statuses = layout.analysis_blocks(cfg)
    assert chosen == [1, 3, 4]
    rows = ledger.read_rows(layout.study_dir(cfg) / "ledger.csv")
    assert len(rows) == 5 and all(r["status"] == "ok" for r in rows)
    assert worker_loop(cfg, fake_run, device="cpu", log=lambda *_: None) == []  # study already full


def test_budget_stops_claiming(tmp_path, built_data):
    cfg = study(built_data, tmp_path, n_blocks=5)
    ledger.append_row(layout.study_dir(cfg) / "ledger.csv",
                      ledger.LedgerRow("x", cfg.meta.name, 0, "cpu", "cpu", 0, 1, 3600.0, "ok"))
    ran = worker_loop(cfg, lambda b: {"status": "complete"}, device="cpu", budget_hours=1.5, log=lambda *_: None)
    assert ran == []


def test_worker_loop_does_not_overshoot_while_blocks_are_in_progress(tmp_path, built_data):
    import fcntl
    import os
    import threading

    cfg = study(built_data, tmp_path, n_blocks=2)
    locks = layout.study_dir(cfg) / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    # Another worker is running block 0 (it holds the lock and has not finished).
    fd = os.open(locks / "block_000.lock", os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)

    def fake_run(b):
        atomic_write_json(layout.block_dir(cfg, b) / "block.json", {"status": layout.STATUS_COMPLETE})
        return {"status": layout.STATUS_COMPLETE}

    def finish_block0():
        atomic_write_json(layout.block_dir(cfg, 0) / "block.json", {"status": layout.STATUS_COMPLETE})
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    timer = threading.Timer(0.5, finish_block0)
    timer.start()
    ran = worker_loop(cfg, fake_run, device="cpu", log=lambda *_: None, poll_seconds=0.05)
    timer.join()
    # Block 1 runs immediately; the worker then waits for block 0 instead of starting block 2.
    assert ran == [1]
    assert layout.block_status(cfg, 2) is None


def test_worker_loop_replaces_an_in_progress_block_that_fails_its_gate(tmp_path, built_data):
    import fcntl
    import os
    import threading

    cfg = study(built_data, tmp_path, n_blocks=2)
    locks = layout.study_dir(cfg) / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    fd = os.open(locks / "block_000.lock", os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)

    def fake_run(b):
        atomic_write_json(layout.block_dir(cfg, b) / "block.json", {"status": layout.STATUS_COMPLETE})
        return {"status": layout.STATUS_COMPLETE}

    def exclude_block0():
        atomic_write_json(layout.block_dir(cfg, 0) / "block.json", {"status": layout.STATUS_EXCLUDED})
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    timer = threading.Timer(0.5, exclude_block0)
    timer.start()
    ran = worker_loop(cfg, fake_run, device="cpu", log=lambda *_: None, poll_seconds=0.05)
    timer.join()
    assert ran == [1, 2]


def test_merge_studies(tmp_path):
    from afterimage.experiment.merge import merge_studies

    for i, blocks in enumerate(([0, 2], [1, 3])):
        src = tmp_path / f"m{i}"
        for b in blocks:
            atomic_write_json(src / f"block_{b:03d}" / "block.json", {"status": "complete"})
        ledger.append_row(src / "ledger.csv", ledger.LedgerRow(f"j{i}", "c", blocks[0], "cuda:0", "gpu", 0, 1, 1.0, "ok"))
        atomic_write_json(src / "determinism.json", {"passed": True, "device": "cuda:0", "env": {"gpu_name": f"G{i}"}})
    out = merge_studies([tmp_path / "m0", tmp_path / "m1"], tmp_path / "dest")
    assert out["blocks"] == ["block_000", "block_001", "block_002", "block_003"]
    assert out["ledger_rows"] == 2 and out["determinism"]["passed"]
    assert len(ledger.read_rows(tmp_path / "dest" / "ledger.csv")) == 2
    atomic_write_json(tmp_path / "m1" / "determinism.json", {"passed": False, "device": "cuda:0"})
    out2 = merge_studies([tmp_path / "m0", tmp_path / "m1"], tmp_path / "dest2")
    assert not out2["determinism"]["passed"]
    atomic_write_json(tmp_path / "m1" / "block_000" / "block.json", {"status": "complete"})
    with pytest.raises(ValueError):
        merge_studies([tmp_path / "m0", tmp_path / "m1"], tmp_path / "dest3")
