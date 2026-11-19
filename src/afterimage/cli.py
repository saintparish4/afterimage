"""Command-line interface: ``afterimage <command> --config configs/<name>.toml``.

Typical order (docs/blueprint.md §13-14)::

    data build -> data audit -> replay -> run -> analyze            (smoke / pilot)
    calibrate (pilot) -> freeze prereg -> data seal -> run
        -> evaluate-sealed -> analyze --confirmatory                (confirmatory)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from .artifacts.env import git_commit_exists, git_info, git_show, repo_root
from .artifacts.hashing import atomic_write_json, read_json
from .config import Config, load_config


def log(msg: str) -> None:
    print(msg, flush=True)


def _cfg(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config)
    if getattr(args, "device", None):
        cfg = cfg.replace(runtime=dataclasses.replace(cfg.runtime, device=args.device))
    return cfg


def _device(cfg: Config):
    from .train import determinism

    determinism.configure(cfg.runtime)  # before any CUDA call
    return determinism.resolve_device(cfg.runtime.device)


def _parse_blocks(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


# --- confirmatory guards ------------------------------------------------------

class GuardError(SystemExit):
    pass


def confirmatory_guards(cfg: Config, *, need_margins: bool) -> list[str]:
    """Return problems that forbid a confirmatory step (empty list = allowed)."""
    probs: list[str] = []
    if cfg.meta.phase != "confirmatory":
        probs.append("config phase is not 'confirmatory'")
    g = git_info()
    if g["dirty"] is not False:
        probs.append("git working tree is dirty or not a repository")
    c = cfg.meta.prereg_commit
    root = repo_root()
    if not c:
        probs.append("meta.prereg_commit is empty")
    elif root is None or not git_commit_exists(c, root):
        probs.append(f"prereg_commit {c!r} is not a commit in this repository")
    else:
        try:
            git_show(c, "prereg/preregistration.md", root)
        except Exception:
            probs.append("prereg/preregistration.md does not exist at the prereg commit")
        rel = Path(cfg.source).resolve().relative_to(root).as_posix() if cfg.source else None
        try:
            import tempfile

            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td) / "prereg_config.toml"
                tmp.write_bytes(git_show(c, rel, root))
                then = load_config(tmp)
            if then.science_hash != cfg.science_hash:
                probs.append("the scientific config differs from the one frozen at the prereg commit")
        except Exception as e:  # noqa: BLE001
            probs.append(f"cannot read {rel} at the prereg commit: {e}")
    if need_margins:
        import math

        a = cfg.analysis
        for k in ("eps_id_nll", "eps_id_aff_nll", "eps_id_acc", "delta_G"):
            if not math.isfinite(getattr(a, k)):
                probs.append(f"analysis.{k} is not set (run calibrate on the pilot)")
    return probs


def _guard(cfg: Config, need_margins: bool) -> None:
    probs = confirmatory_guards(cfg, need_margins=need_margins)
    if probs:
        for p in probs:
            log(f"refused: {p}")
        raise GuardError(2)


# --- commands -------------------------------------------------------------------

def cmd_data_build(args):
    from .data.splits import build_dev, verify

    cfg = _cfg(args)
    out = cfg.data_path()
    if (out / "manifest.json").exists() and not args.force:
        problems = verify(out)
        log(f"dataset exists at {out}; verify: {'ok' if not problems else problems}")
        return 0 if not problems else 1
    m = build_dev(cfg, out, log=log)
    for name, e in m["splits"].items():
        log(f"  {name:9s} rows={e['rows']:>9,}  affected={e['affected_rate']:.3f}  sha256={e['file_sha256'][:12]}")
    log(f"wrote {out}")
    return 0


def cmd_data_verify(args):
    from .data.splits import verify

    problems = verify(_cfg(args).data_path())
    for p in problems:
        log(f"FAIL {p}")
    log("ok" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


def cmd_data_audit(args):
    from .data.splits import audit

    res = audit(_cfg(args), _cfg(args).data_path())
    for k, (ok, detail) in res.items():
        log(f"{'ok  ' if ok else 'FAIL'} {k} {detail}")
    return 0 if all(ok for ok, _ in res.values()) else 1


def cmd_data_seal(args):
    from .data.splits import build_sealed

    cfg = _cfg(args)
    if not args.unsafe_dev:
        _guard(cfg, need_margins=True)
    commit = cfg.meta.prereg_commit or args.commit
    if not commit:
        log("refused: no commit to derive the sealed seed from")
        return 2
    m = build_sealed(cfg, cfg.data_path(), commit, log=log)
    log(f"sealed splits written (seed {m['seed']} from commit {commit[:12]})")
    return 0


def cmd_run(args):
    from .experiment.launcher import worker_loop
    from .experiment.run_block import load_context, run_block

    cfg = _cfg(args)
    if cfg.meta.phase == "confirmatory" and not args.unsafe_dev:
        _guard(cfg, need_margins=True)
    import os

    from .artifacts.ledger import GpuSampler
    from .experiment.layout import study_dir

    dev = _device(cfg)
    ctx = load_context(cfg, dev, log=log)
    sample_path = study_dir(cfg) / "logs" / f"gpu_{os.getpid()}.csv"
    with GpuSampler(sample_path, device_index=dev.index or 0) if dev.type == "cuda" else _Null():
        ran = worker_loop(cfg, lambda b: run_block(cfg, b, ctx), device=str(dev), blocks=_parse_blocks(args.blocks),
                          budget_hours=args.budget_gpu_hours, log=log)
    log(f"ran blocks: {ran}")
    return 0


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def cmd_launch(args):
    from .experiment.launcher import launch

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    return launch(args.config, workers_per_device=args.workers_per_device, devices=devices,
                  budget_hours=args.budget_gpu_hours)


def cmd_replay(args):
    from .experiment.run_block import load_context
    from .experiment.replay import check_determinism

    cfg = _cfg(args)
    dev = _device(cfg)
    r = check_determinism(load_context(cfg, dev, log=log), steps=args.steps)
    log(f"replay identical: {r['replay_identical']}  resume identical: {r['resume_identical']}  "
        f"({r['steps_per_second']:.1f} steps/s on {r['device']})")
    return 0 if r["passed"] else 1


def cmd_reproduce_r1(args):
    from .experiment.layout import study_dir
    from .experiment.replay import replay_from_released
    from .experiment.run_block import load_context

    cfg = _cfg(args)
    cfg = cfg.replace(paths=dataclasses.replace(cfg.paths, runs_dir=args.into))
    if study_dir(cfg).exists() and any(study_dir(cfg).iterdir()):
        log(f"refused: {study_dir(cfg)} is not empty")
        return 2
    dev = _device(cfg)
    blocks = _parse_blocks(args.blocks) or sorted(
        int(p.name.split("_")[1]) for p in Path(args.released).glob("block_[0-9][0-9][0-9]"))
    r = replay_from_released(load_context(cfg, dev, log=log), Path(args.released), blocks)
    for b, v in r["blocks"].items():
        log(f"block_{int(b):03d}: {'identical' if v['identical'] else 'DIFFERENT'} ({len(v['arms'])} arms)")
    log("R1 replication " + ("succeeded" if r["passed"] else "FAILED"))
    return 0 if r["passed"] else 1


def cmd_evaluate_sealed(args):
    from .experiment.sealed_eval import evaluate_sealed

    cfg = _cfg(args)
    n = evaluate_sealed(cfg, _device(cfg), log=log)
    log(f"{n} checkpoint(s) evaluated on the sealed splits")
    return 0


def _results_dir(cfg: Config, args) -> Path:
    base = Path(args.out) if args.out else Path("results")
    return base / cfg.meta.name


def cmd_analyze(args):
    from .analysis.endpoints import Cube
    from .analysis.plan import run_analysis
    from .analysis.report import render
    from .data.splits import audit, sealed_available
    from .experiment.layout import study_dir

    cfg = _cfg(args)
    out = _results_dir(cfg, args)
    mode = "sealed" if args.confirmatory else "dev"
    if args.confirmatory:
        _guard(cfg, need_margins=True)
        if (out / "analysis.json").exists():
            log(f"refused: {out / 'analysis.json'} exists; the confirmatory analysis runs once")
            return 2
        if not sealed_available(cfg.data_path()):
            log("refused: sealed splits not built")
            return 2
        sm = read_json(cfg.data_path() / "sealed_manifest.json")
        if sm["prereg_commit"] != cfg.meta.prereg_commit:
            log("refused: sealed splits were derived from a different commit")
            return 2
    else:
        out = out / "dev"
    det_p = study_dir(cfg) / "determinism.json"
    det = read_json(det_p) if det_p.exists() else None
    aud = audit(cfg, cfg.data_path()) if (args.confirmatory or args.audit) else None
    res = run_analysis(Cube.load(cfg, mode), determinism=det, audit=aud)
    res["git"] = git_info()
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "analysis.json", res)
    (out / "report.md").write_text(render(res))
    c = res["category"]
    log(f"outcome: {c['category']} ({c['label']})  primary estimate {res['primary']['estimate']:.4g} "
        f"95% CI [{res['primary']['ci95'][0]:.4g}, {res['primary']['ci95'][1]:.4g}]  n={res['primary']['n']}")
    log(f"wrote {out / 'report.md'}")
    return 0


def cmd_calibrate(args):
    from .analysis.calibrate import calibrate
    from .analysis.endpoints import Cube

    cfg = _cfg(args)
    if cfg.meta.phase == "confirmatory":
        log("refused: calibrate runs on pilot data, never on confirmatory data")
        return 2
    res = calibrate(Cube.load(cfg, "dev"))
    out = _results_dir(cfg, args)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "calibration.json", res)
    log(json.dumps({k: res[k] for k in ("eps", "delta_G", "sigma_d", "n_detection", "n_equivalence", "n_blocks")}, indent=1))
    log("paste into configs/confirmatory.toml:\n" + res["toml"])
    return 0


def cmd_diagnose(args):
    from .diagnostics.diagnose import run_diagnostics
    from .experiment.layout import analysis_blocks

    cfg = _cfg(args)
    dev = _device(cfg)
    blocks = _parse_blocks(args.blocks) or analysis_blocks(cfg)[0]
    for b in blocks:
        run_diagnostics(cfg, b, dev, with_tracin=not args.no_tracin, log=log)
    return 0


def cmd_status(args):
    from .artifacts import ledger
    from .experiment.layout import analysis_blocks, study_dir

    cfg = _cfg(args)
    chosen, statuses = analysis_blocks(cfg)
    for b, s in sorted(statuses.items()):
        log(f"block_{b:03d}  {s}{'  (analysed)' if b in chosen else ''}")
    log(f"{len(chosen)}/{cfg.design.n_blocks} blocks available for analysis")
    for k, v in ledger.totals(study_dir(cfg) / "ledger.csv").items():
        log(f"device-hours {k}: {v:.3f}")
    return 0


def cmd_audit(args):
    from .analysis.stats import paired
    from .data.splits import audit, verify
    from .experiment.layout import analysis_blocks, block_dir, study_dir

    cfg = _cfg(args)
    ok = True

    def report(name: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= good
        log(f"{'ok  ' if good else 'FAIL'} {name} {detail}")

    problems = verify(cfg.data_path())
    report("data manifest hashes", not problems, "; ".join(problems))
    for k, (good, detail) in audit(cfg, cfg.data_path()).items():
        report(f"leakage: {k}", good, detail)
    det = study_dir(cfg) / "determinism.json"
    if det.exists():
        d = read_json(det)
        report("determinism replay", bool(d["passed"]), d["device"])
    else:
        report("determinism replay", cfg.meta.phase != "confirmatory", "not run (afterimage replay)")
    chosen, _ = analysis_blocks(cfg)
    for b in chosen:
        st = block_dir(cfg, b) / "stages.json"
        if st.exists():
            report(f"block_{b:03d} stage A", bool(read_json(st)["A"]["ok"]))
    import numpy as np

    x = np.random.default_rng(0).normal(size=12)
    a, b2 = paired(x, n_boot=200), paired(-x, n_boot=200)
    report("analysis antisymmetry (swap arms flips sign)", a.estimate == -b2.estimate and a.p == b2.p)
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="afterimage", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", required=True)
        sp.set_defaults(fn=fn)
        return sp

    data = sub.add_parser("data", help="build, verify, audit or seal datasets")
    dsub = data.add_subparsers(dest="data_cmd", required=True)
    for name, fn, h in (("build", cmd_data_build, "build pool and development splits"),
                        ("verify", cmd_data_verify, "check files against the manifest"),
                        ("audit", cmd_data_audit, "exact leakage audit"),
                        ("seal", cmd_data_seal, "build sealed confirmatory splits")):
        sp = dsub.add_parser(name, help=h)
        sp.add_argument("--config", required=True)
        sp.set_defaults(fn=fn)
        if name == "build":
            sp.add_argument("--force", action="store_true", help="rebuild even if a dataset exists")
        if name == "seal":
            sp.add_argument("--commit", help="dev only: seed source when phase is not confirmatory")
            sp.add_argument("--unsafe-dev", action="store_true", help="skip confirmatory guards (testing only)")

    r = add("run", cmd_run, "run blocks in this process")
    r.add_argument("--blocks", help="e.g. 0-3,7 (default: fill the study)")
    r.add_argument("--device")
    r.add_argument("--budget-gpu-hours", type=float, default=0.0)
    r.add_argument("--unsafe-dev", action="store_true", help="skip confirmatory guards (testing only)")

    la = add("launch", cmd_launch, "spawn worker processes across devices")
    la.add_argument("--devices", default="cuda:0")
    la.add_argument("--workers-per-device", type=int, default=4)
    la.add_argument("--budget-gpu-hours", type=float, default=0.0)

    rp = add("replay", cmd_replay, "check bitwise replay and resume on this device")
    rp.add_argument("--steps", type=int, default=500)
    rp.add_argument("--device")

    r1 = add("reproduce-r1", cmd_reproduce_r1, "tier R1: rerun forks from released fork states, compare bits")
    r1.add_argument("--released", required=True, help="released study directory (runs/<name>)")
    r1.add_argument("--into", required=True, help="empty runs directory to replay into")
    r1.add_argument("--blocks")
    r1.add_argument("--device")

    es = add("evaluate-sealed", cmd_evaluate_sealed, "predict sealed splits at saved horizons")
    es.add_argument("--device")

    an = add("analyze", cmd_analyze, "run the pre-registered analysis")
    an.add_argument("--confirmatory", action="store_true")
    an.add_argument("--audit", action="store_true", help="include the leakage audit in dev mode")
    an.add_argument("--out")

    ca = add("calibrate", cmd_calibrate, "derive margins and n from pilot results")
    ca.add_argument("--out")

    dg = add("diagnose", cmd_diagnose, "exploratory diagnostics per block")
    dg.add_argument("--blocks")
    dg.add_argument("--device")
    dg.add_argument("--no-tracin", action="store_true")

    add("status", cmd_status, "block statuses and device-hours")
    add("audit", cmd_audit, "data, leakage, determinism and Stage A checks")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
