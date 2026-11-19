import dataclasses
from pathlib import Path

import pytest

from afterimage.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def tiny_config(tmp: Path, **design):
    """The smoke config, shrunk further and pointed at a temporary directory."""
    cfg = load_config(ROOT / "configs" / "smoke.toml")
    data = dataclasses.replace(
        cfg.data, pool_size=6_000, chunk_size=2_000, n_id_dev=300, n_ood_dev=300,
        n_id_conf=200, n_ood_conf=200, n_depth5=100, n_probe_id=64, n_probe_ood=64, workers=2,
    )
    des = dataclasses.replace(cfg.design, **design) if design else cfg.design
    paths = dataclasses.replace(cfg.paths, data_dir=str(tmp / "data"), runs_dir=str(tmp / "runs"))
    return cfg.replace(data=data, design=des, paths=paths)


@pytest.fixture(scope="session")
def built_data(tmp_path_factory):
    from afterimage.data.splits import build_dev

    tmp = tmp_path_factory.mktemp("built")
    cfg = tiny_config(tmp)
    out = cfg.data_path()
    manifest = build_dev(cfg, out, log=lambda *_: None)
    return cfg, out, manifest
