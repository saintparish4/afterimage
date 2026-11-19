"""The CLI, its confirmatory guards, and (slow) the whole pipeline end to end."""

import json
from pathlib import Path

import pytest

from afterimage.cli import _parse_blocks, confirmatory_guards, main
from afterimage.config import load_config

ROOT = Path(__file__).resolve().parents[1]

TINY = """
[meta]
name = "tiny"
phase = "smoke"
master_seed = 3
data_seed = 3

[paths]
data_dir = "{tmp}/data"
runs_dir = "{tmp}/runs"

[data]
pool_size = 6_000
chunk_size = 3_000
n_id_dev = 300
n_ood_dev = 300
n_id_conf = 200
n_ood_conf = 200
n_depth5 = 100
n_probe_id = 64
n_probe_ood = 64
workers = 2
heldout_x = [["I", 1, "X"], ["A", 2, "X"], ["X", 1, "O"]]
heldout_nonx = [["I", 2, "A"], ["O", 1, "I"], ["A", 1, "N"]]

[model]
layers = 2
d_model = 32
heads = 2
d_ff = 64

[optim]
lr = 3e-3
warmup = 20
batch = 32

[design]
n_blocks = 2
T0 = 80
K = 20
H = 40
horizons = [0, 20, 40]
base_checkpoints = 3
R = 10
relearn_eval_every = 5
resume_every = 30
hybrids = true

[gate]
enforce = false

[analysis]
eps_id_nll = 0.05
eps_id_aff_nll = 0.1
delta_G = 0.05
n_bootstrap = 500
n_min = 2

[runtime]
device = "cpu"
threads = 1
"""


def test_parse_blocks():
    assert _parse_blocks("0-2,5") == [0, 1, 2, 5]
    assert _parse_blocks(None) is None


def test_guards_refuse_non_confirmatory_and_unfrozen_configs():
    smoke = load_config(ROOT / "configs" / "smoke.toml")
    assert any("phase" in p for p in confirmatory_guards(smoke, need_margins=True))
    conf = load_config(ROOT / "configs" / "confirmatory.toml")
    probs = confirmatory_guards(conf, need_margins=True)
    assert any("prereg_commit is empty" in p for p in probs)
    assert any("delta_G" in p for p in probs)


def test_calibrate_refuses_confirmatory(capsys):
    assert main(["calibrate", "--config", str(ROOT / "configs" / "confirmatory.toml")]) == 2
    assert "never on confirmatory data" in capsys.readouterr().out


def test_analyze_confirmatory_is_guarded():
    with pytest.raises(SystemExit):
        main(["analyze", "--config", str(ROOT / "configs" / "confirmatory.toml"), "--confirmatory"])


@pytest.mark.slow
def test_pipeline_end_to_end(tmp_path):
    cfgp = tmp_path / "tiny.toml"
    cfgp.write_text(TINY.format(tmp=tmp_path))
    c = ["--config", str(cfgp)]
    assert main(["data", "build", *c]) == 0
    assert main(["data", "verify", *c]) == 0
    assert main(["data", "audit", *c]) == 0
    assert main(["replay", *c, "--steps", "40"]) == 0
    assert main(["run", *c]) == 0
    assert main(["status", *c]) == 0
    assert main(["analyze", *c, "--audit", "--out", str(tmp_path / "res")]) == 0
    res = json.loads((tmp_path / "res" / "tiny" / "dev" / "analysis.json").read_text())
    assert res["primary"]["n"] == 2 and res["validity"]["checks"]["leakage_audit"]["ok"]
    assert res["category"]["category"] in {"A", "A1", "A2", "B", "C", "D"}
    assert main(["calibrate", *c, "--out", str(tmp_path / "res")]) == 0
    assert main(["diagnose", *c, "--blocks", "0"]) == 0
    # Sealed path (dev seeding, guards skipped explicitly): seal, evaluate, analyze sealed predictions.
    assert main(["data", "seal", *c, "--commit", "0" * 40, "--unsafe-dev"]) == 0
    assert main(["evaluate-sealed", *c]) == 0
    from afterimage.analysis.endpoints import Cube
    from afterimage.analysis.plan import run_analysis

    cfg = load_config(cfgp)
    sealed = run_analysis(Cube.load(cfg, "sealed"), determinism=None, audit=None)
    assert sealed["primary"]["n"] == 2
    assert main(["audit", *c]) == 0
