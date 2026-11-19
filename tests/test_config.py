import dataclasses
from pathlib import Path

import pytest

from afterimage.config import load_config, validate

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["smoke", "pilot", "confirmatory"])
def test_shipped_configs_load(name):
    cfg = load_config(ROOT / "configs" / f"{name}.toml")
    assert cfg.meta.phase == name
    assert len(cfg.science_hash) == 64


def test_pilot_and_confirmatory_share_the_training_pool():
    p = load_config(ROOT / "configs" / "pilot.toml")
    c = load_config(ROOT / "configs" / "confirmatory.toml")
    assert p.data_hash == c.data_hash
    assert p.science_hash != c.science_hash


def test_science_hash_ignores_runtime_and_prereg_commit():
    cfg = load_config(ROOT / "configs" / "smoke.toml")
    other = cfg.replace(
        runtime=dataclasses.replace(cfg.runtime, device="cuda:3", threads=8),
        meta=dataclasses.replace(cfg.meta, prereg_commit="abc"),
    )
    assert other.science_hash == cfg.science_hash
    changed = cfg.replace(design=dataclasses.replace(cfg.design, K=cfg.design.K + 1))
    assert changed.science_hash != cfg.science_hash


def test_validation_rejects_bad_designs(tmp_path):
    cfg = load_config(ROOT / "configs" / "smoke.toml")
    bad = [
        cfg.replace(design=dataclasses.replace(cfg.design, horizons=(0, 10))),
        cfg.replace(design=dataclasses.replace(cfg.design, arms=("I", "N"))),
        cfg.replace(data=dataclasses.replace(cfg.data, heldout_x=(("I", 2, "A"),))),
        cfg.replace(data=dataclasses.replace(cfg.data, heldout_nonx=(("X", 1, "O"),))),
        cfg.replace(data=dataclasses.replace(cfg.data, ctx=20)),
    ]
    for b in bad:
        with pytest.raises(ValueError):
            validate(b)
    p = tmp_path / "x.toml"
    p.write_text((ROOT / "configs" / "smoke.toml").read_text() + "\n[bogus]\nx = 1\n")
    with pytest.raises(ValueError, match="unknown config sections"):
        load_config(p)
