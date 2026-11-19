from pathlib import Path

import numpy as np
import pytest

from afterimage.artifacts import hashing, ledger


def test_array_digest_is_order_independent_and_sensitive():
    a = {"x": np.arange(6, dtype=np.float32), "y": np.zeros(3, dtype=np.uint8)}
    b = {"y": np.zeros(3, dtype=np.uint8), "x": np.arange(6, dtype=np.float32)}
    assert hashing.array_digest(a) == hashing.array_digest(b)
    # dtype, shape, name and value each change the digest
    assert hashing.array_digest({"x": np.arange(6, dtype=np.float64)}) != hashing.array_digest(
        {"x": np.arange(6, dtype=np.float32)}
    )
    assert hashing.array_digest({"x": np.arange(6, dtype=np.float32).reshape(2, 3)}) != (
        hashing.array_digest({"x": np.arange(6, dtype=np.float32)})
    )
    assert hashing.array_digest({"z": np.arange(6, dtype=np.float32)}) != (
        hashing.array_digest({"x": np.arange(6, dtype=np.float32)})
    )
    c = {"x": np.arange(6, dtype=np.float32)}
    c["x"][0] = np.nextafter(np.float32(0), np.float32(1))
    assert hashing.array_digest(c) != hashing.array_digest({"x": np.arange(6, dtype=np.float32)})


def test_array_digest_handles_torch_tensors():
    torch = pytest.importorskip("torch")
    t = torch.arange(5, dtype=torch.float32)
    assert hashing.array_digest({"t": t}) == hashing.array_digest({"t": t.clone()})
    u = t.clone()
    u[0] = 1e-30
    assert hashing.array_digest({"t": t}) != hashing.array_digest({"t": u})


def test_canonical_json_is_stable_and_rejects_nan():
    assert hashing.canonical_json({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'
    with pytest.raises(ValueError):
        hashing.canonical_json({"x": float("nan")})


def test_atomic_write_and_publish_dir(tmp_path: Path):
    p = tmp_path / "a" / "b.json"
    hashing.atomic_write_json(p, {"k": 1})
    assert hashing.read_json(p) == {"k": 1}
    assert not (p.parent / "b.json.tmp").exists()

    tmp = tmp_path / "d.tmp"
    tmp.mkdir()
    (tmp / "f").write_text("x")
    final = tmp_path / "d"
    final.mkdir()
    (final / "stale").write_text("old")
    hashing.publish_dir(tmp, final)
    assert (final / "f").read_text() == "x"
    assert not (final / "stale").exists()
    assert not tmp.exists()


def test_ledger_totals(tmp_path: Path):
    path = tmp_path / "ledger.csv"
    for i, status in enumerate(["ok", "ok", "failed"]):
        ledger.append_row(
            path,
            ledger.LedgerRow(f"j{i}", "pilot", i, "cpu", "cpu", 0.0, 1.0, 1800.0, status),
        )
    t = ledger.totals(path)
    assert t["pilot"] == pytest.approx(1.0)
    assert t["pilot:failed"] == pytest.approx(0.5)
    assert len(ledger.read_rows(path)) == 3
