"""Stage A (identical starts) and Stage B (manipulation check), per block.

Stage C (familiar-task equivalence and the held-out endpoints) is a
population-level analysis and lives in ``afterimage.analysis``.
"""

from __future__ import annotations

from ..config import Config


def stage_a(fork_digest: str, arm_starts: dict[str, str]) -> dict:
    """Every non-hybrid arm must start from the fork state, bit for bit."""
    checks = {arm: {"start_digest": d, "ok": d == fork_digest} for arm, d in arm_starts.items()}
    return {"fork_digest": fork_digest, "arms": checks, "ok": all(c["ok"] for c in checks.values())}


def stage_b(cfg: Config, metrics_at_window_end: dict[str, dict[str, dict]]) -> dict:
    """Manipulation check at T0+K on ``id_dev``.

    ``metrics_at_window_end[arm][subset]`` holds summarize() rows for id_dev.
    Blocks that fail stay in the primary analysis (intention to treat).
    """
    sb, g = cfg.stage_b, cfg.gate
    out: dict = {"checks": {}}
    I, C = metrics_at_window_end.get("I"), metrics_at_window_end.get("C")
    if I is None or C is None or "aff" not in I:
        out["ok"] = None
        out["note"] = "arm I or C missing; manipulation check not applicable"
        return out
    fa = I["aff"]["false_agree"]
    gap = C["aff"]["acc"] - I["aff"]["acc"]
    out["checks"]["I_adopts_false_rule"] = {"value": fa, "ok": fa >= sb.false_agree_min, "rule": f">= {sb.false_agree_min}"}
    out["checks"]["C_minus_I_aff_acc_gap"] = {"value": gap, "ok": gap >= sb.aff_gap_min, "rule": f">= {sb.aff_gap_min}"}
    c_acc = C["all"]["acc"]
    out["checks"]["C_still_learned"] = {"value": c_acc, "ok": c_acc >= g.id_min_acc, "rule": f">= {g.id_min_acc}"}
    out["ok"] = all(c["ok"] for c in out["checks"].values())
    return out
