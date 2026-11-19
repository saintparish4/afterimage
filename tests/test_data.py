"""M1: split construction, leakage audit, sealed sets and the batch schedule."""

import dataclasses

import numpy as np
import pytest

from afterimage.data import schedule
from afterimage.data.splits import (
    DEV_SPLITS, audit, build_dev, build_pool, build_sealed, load_splits, read_split, row_hash,
    sealed_seed, verify, write_split,
)
from afterimage.lang.grammar import TRIPLES, Op

from .conftest import tiny_config


def test_manifest_matches_files(built_data):
    cfg, out, manifest = built_data
    assert verify(out) == []
    assert set(manifest["splits"]) == {*DEV_SPLITS, "pool"}
    assert manifest["splits"]["pool"]["rows"] == cfg.data.pool_size


def test_audit_passes_on_built_data(built_data):
    cfg, out, _ = built_data
    res = audit(cfg, out)
    failed = {k: v for k, v in res.items() if not v[0]}
    assert not failed, failed
    assert "pool_disjoint_from_id_dev" in res


def test_split_contents(built_data):
    cfg, out, _ = built_data
    sp = load_splits(out, [*DEV_SPLITS, "pool"])
    assert set(np.unique(sp["id_small"].depth)) == {1, 2}
    assert set(np.unique(sp["id_dev"].depth)) <= set(cfg.data.id_eval_depths)
    assert set(np.unique(sp["ood_dev"].depth)) <= set(cfg.data.ood_depths)
    assert (sp["ood_dev"].heldout_x | sp["ood_dev"].heldout_nonx).all()
    assert not (sp["pool"].heldout_x | sp["pool"].heldout_nonx).any()
    assert (sp["depth5"].depth == 5).all()
    assert (sp["depth5"].eq_pos <= cfg.data.max_expr_len + 1).all()
    # Probe = first rows of the dev splits, tagged by source.
    p = sp["probe"]
    np.testing.assert_array_equal(p.tokens[: cfg.data.n_probe_id], sp["id_dev"].tokens[: cfg.data.n_probe_id])
    assert (p.source[: cfg.data.n_probe_id] == 0).all() and (p.source[cfg.data.n_probe_id :] == 1).all()
    # The false rule changes some, but not all, labels.
    aff = sp["pool"].affected.mean()
    assert 0.05 < aff < 0.6
    assert (sp["pool"].affected <= sp["pool"].has_x).all()  # affected implies contains X


def test_heldout_flags_match_triples(built_data):
    cfg, out, _ = built_data
    hx, hn = cfg.data.heldout_triples()
    ood = read_split(out / "ood_dev.parquet")
    for i in range(len(ood)):
        present = {TRIPLES[b] for b in range(45) if int(ood.sig_mask[i]) >> b & 1}
        assert ood.heldout_x[i] == bool(present & hx)
        assert ood.heldout_nonx[i] == bool(present & hn)
    assert all(Op.X in (t[0], t[2]) for t in hx)


def test_build_is_deterministic(built_data, tmp_path):
    cfg, out, manifest = built_data
    cfg2 = cfg.replace(paths=dataclasses.replace(cfg.paths, data_dir=str(tmp_path)))
    m2 = build_dev(cfg2, cfg2.data_path(), log=lambda *_: None)
    for name, e in manifest["splits"].items():
        assert m2["splits"][name]["content_digest"] == e["content_digest"], name


def test_pool_is_independent_of_worker_count(built_data):
    cfg, out, _ = built_data
    excl = row_hash(read_split(out / "id_dev.parquet").tokens)
    a = build_pool(dataclasses.replace(cfg.data, workers=1), cfg.meta.data_seed, excl)
    b = build_pool(dataclasses.replace(cfg.data, workers=3), cfg.meta.data_seed, excl)
    assert a.content_digest() == b.content_digest()


def test_parquet_round_trip(built_data, tmp_path):
    _, out, _ = built_data
    sp = read_split(out / "ood_dev.parquet")
    write_split(sp, tmp_path / "x.parquet")
    assert read_split(tmp_path / "x.parquet").content_digest() == sp.content_digest()


def test_verify_detects_tampering(built_data, tmp_path):
    cfg, out, _ = built_data
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(out, copy)
    sp = read_split(copy / "id_dev.parquet")
    sp.y_true[0] ^= 1
    write_split(sp, copy / "id_dev.parquet")
    assert any("id_dev" in p for p in verify(copy))


def test_sealed_sets(built_data, tmp_path):
    cfg, out, _ = built_data
    import shutil

    d = tmp_path / "d"
    shutil.copytree(out, d)
    m = build_sealed(cfg, d, "deadbeef", log=lambda *_: None)
    assert m["seed"] == sealed_seed("deadbeef") != sealed_seed("deadbeee")
    res = audit(cfg, d)
    assert all(ok for ok, _ in res.values()), {k: v for k, v in res.items() if not v[0]}
    assert "pool_disjoint_from_id_conf" in res and "ood_dev_disjoint_from_ood_conf" in res
    with pytest.raises(FileExistsError):
        build_sealed(cfg, d, "deadbeef", log=lambda *_: None)
    assert verify(d) == []


# --- schedule -----------------------------------------------------------------

def test_batch_indices_pure_and_keyed():
    a = schedule.batch_indices(7, "main", 13, 1000, 64)
    b = schedule.batch_indices(7, "main", 13, 1000, 64)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, schedule.batch_indices(7, "alt", 13, 1000, 64))
    assert not np.array_equal(a, schedule.batch_indices(8, "main", 13, 1000, 64))


def test_each_epoch_visits_every_index_once():
    pool, B = 1000, 50
    idx = np.concatenate([schedule.batch_indices(3, "main", t, pool, B) for t in range(pool // B)])
    assert np.array_equal(np.sort(idx), np.arange(pool))


def test_batches_straddle_epochs():
    pool, B = 1000, 64
    t = pool // B  # step whose batch crosses the epoch boundary
    idx = schedule.batch_indices(3, "main", t, pool, B)
    assert idx.size == B
    assert np.unique(idx).size >= B - (pool % B)  # may repeat only across the boundary


def test_seeds_are_distinct_and_stable():
    seeds = {schedule.block_seed(42, b) for b in range(100)}
    assert len(seeds) == 100
    assert schedule.block_seed(42, 0) == schedule.block_seed(42, 0)
    assert schedule.derive_seed(1, 2) != schedule.derive_seed(2, 1)


def test_sham_positions():
    p = schedule.sham_positions(5, 100, 64, 17)
    assert p.size == 17 and np.unique(p).size == 17 and p.max() < 64
    np.testing.assert_array_equal(p, schedule.sham_positions(5, 100, 64, 17))


def test_tiny_config_is_valid(tmp_path):
    from afterimage.config import validate

    validate(tiny_config(tmp_path))


def test_ood_min_heldout_is_enforced(tmp_path):
    from afterimage.data.splits import _masks, sample_split

    cfg = tiny_config(tmp_path)
    d = dataclasses.replace(cfg.data, ood_min_heldout=3)
    sp = sample_split(200, name="ood", seed_keys=[1, 2], depths=(2, 3, 4), weights=None,
                      heldout="require", cfg=d, min_heldout=3)
    hx, hn = _masks(d)
    counts = [bin(int(m) & (hx | hn)).count("1") for m in sp.sig_mask]
    assert min(counts) >= 3
