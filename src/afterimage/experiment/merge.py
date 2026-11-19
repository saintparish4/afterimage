"""Merge study directories produced on separate machines.

The launcher coordinates workers through file locks on one shared
filesystem. When blocks are split across machines without shared storage
(explicit, disjoint ``--blocks`` lists per machine), each machine has its
own study directory. ``merge_studies`` combines them:

* block directories are copied as they are; a block present in two sources
  is an error, because the blocks must be disjoint;
* ledgers are concatenated;
* determinism records are combined into ``determinism.json``, which passes
  only if every device's replay passed, with each device's record kept
  under ``devices``.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from ..artifacts.hashing import atomic_write_json, read_json
from ..artifacts.ledger import LEDGER_FIELDS


def merge_studies(sources: list[Path], dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    seen: dict[str, Path] = {}
    for src in sources:
        for bd in sorted(src.glob("block_[0-9][0-9][0-9]")):
            if bd.name in seen:
                raise ValueError(f"{bd.name} appears in both {seen[bd.name]} and {src}")
            seen[bd.name] = src
    if any((dest / name).exists() for name in seen):
        raise FileExistsError("destination already contains some of these blocks")
    for name, src in seen.items():
        shutil.copytree(src / name, dest / name)
    rows: list[dict] = []
    for src in sources:
        p = src / "ledger.csv"
        if p.exists():
            with open(p, newline="") as f:
                rows += [r for r in csv.DictReader(f) if r.get("job_id") != "job_id"]
    if rows:
        with open(dest / "ledger.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
            w.writeheader()
            w.writerows(rows)
    devices = []
    for src in sources:
        p = src / "determinism.json"
        if p.exists():
            d = read_json(p)
            devices.append({"source": str(src), "device": d.get("device"), "passed": bool(d.get("passed")),
                            "gpu_name": d.get("env", {}).get("gpu_name"), "digests": d.get("digests")})
    det = {
        "passed": bool(devices) and all(d["passed"] for d in devices),
        "device": ", ".join(sorted({f"{d['device']} ({d['gpu_name']})" for d in devices})),
        "devices": devices,
        "note": "merged from per-machine replay checks; every device must pass",
    }
    atomic_write_json(dest / "determinism.json", det)
    return {"blocks": sorted(seen), "ledger_rows": len(rows), "determinism": det}
