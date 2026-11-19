"""Render an analysis result as Markdown."""

from __future__ import annotations

import math
from typing import Any


def _f(x: Any, nd: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_f(v, nd) for v in x) + "]"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.{nd}g}"
    return str(x)


def render(res: dict) -> str:
    cat = res["category"]
    L = [
        "# AFTERIMAGE analysis",
        "",
        f"- mode: `{res['mode']}` (split roles: {'sealed confirmatory sets' if res['mode'] == 'sealed' else 'development sets'})",
        f"- blocks analysed: {len(res['blocks'])} of {res['n_target']} targeted",
        f"- science hash: `{res['science_hash'][:16]}`",
        "",
        f"## Outcome: **{cat['category']}**, {cat['label']}",
        "",
    ]
    if cat.get("reasons"):
        L += [f"- reason: {r}" for r in cat["reasons"]] + [""]
    p = res["primary"]
    L += [
        "## Primary endpoint",
        "",
        f"{p['contrast']} at horizon H: estimate {_f(p['estimate'])} nats, 95% CI {_f(p['ci95'])}, "
        f"BCa {_f(p['ci_bca'])}, p = {_f(p['p'])}, Wilcoxon p = {_f(p['p_wilcoxon'])}, d_z = {_f(p['dz'])}, "
        f"n = {p['n']}. TOST at delta_G = {_f(p['margin'])}: p = {_f(p['p_tost'])}.",
        "",
        "## Familiar-task equivalence (precondition)",
        "",
        "| metric | estimate | 90% CI | margin | TOST p | equivalent |",
        "|---|---|---|---|---|---|",
    ]
    for k, r in res["id_equivalence"].items():
        L.append(f"| {k} | {_f(r['estimate'])} | {_f(r['ci90'])} | {_f(r['margin'])} | {_f(r['p_tost'])} | {r['equivalent']} |")
    L += ["", "## Secondary family (Holm)", "", "| id | contrast | estimate | 95% CI | p | p (Holm) |", "|---|---|---|---|---|---|"]
    for k, r in res["secondary"].items():
        L.append(f"| {k} | {r['contrast']} | {_f(r['estimate'])} | {_f(r['ci95'])} | {_f(r['p'])} | {_f(r['p_holm'])} |")
    pl = res["placebo"]
    L += ["", "## Placebo (N - C)", "", f"estimate {_f(pl['estimate'])}, 95% CI {_f(pl['ci95'])}, p = {_f(pl['p'])}", "",
          "## Persistence curve (descriptive)", "", "| h | estimate | 95% CI |", "|---|---|---|"]
    for h, r in res["persistence"].items():
        L.append(f"| {h} | {_f(r['estimate'])} | {_f(r['ci95'])} |")
    L += ["", "## Validity", ""]
    for k, c in res["validity"]["checks"].items():
        L.append(f"- {'ok ' if c['ok'] else 'FAIL'} {k}: {_f(c.get('value'))}")
    s = res["sensitivity"]
    L += ["", "## Sensitivity (not confirmatory)", "",
          f"- Brier: estimate {_f(s['brier']['estimate'])}, 95% CI {_f(s['brier']['ci95'])}, sign agrees: {s['brier']['sign_agrees_with_primary']}"]
    if "crossed_bootstrap_ci95" in s:
        L.append(f"- crossed (blocks x examples) bootstrap 95% CI: {_f(s['crossed_bootstrap_ci95'])}")
    pp = s["per_protocol_stage_b"]
    L.append(f"- per-protocol (Stage B passes only, {pp['n_excluded']} excluded): estimate {_f(pp['estimate'])}, 95% CI {_f(pp['ci95'])}")
    if "worst_case_imputation" in s:
        w = s["worst_case_imputation"]
        L.append(f"- worst-case imputation ({w['n_imputed']} imputed): estimate {_f(w['estimate'])}, 95% CI {_f(w['ci95'])}")
    L += ["", "## Descriptive", ""]
    for k, v in res["descriptive"].items():
        L.append(f"- {k}: {_f(v)}")
    return "\n".join(L) + "\n"
