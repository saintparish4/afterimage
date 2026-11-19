"""Content hashing and crash-safe file writes.

Two kinds of digest are used throughout the project:

* ``sha256_file`` hashes the bytes of a file. It detects corruption and
  identifies a released artifact exactly.
* ``array_digest`` hashes named arrays by (name, dtype, shape, bytes) in
  sorted name order. It is independent of the container format, so two
  training states are "the same" exactly when their digests match, even if
  they were serialized by different library versions.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

_CHUNK = 1 << 20


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, NaN rejected."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_default)


def _default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def json_digest(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode())


def _as_numpy(x: Any) -> np.ndarray:
    # Torch tensors are converted without importing torch here.
    if hasattr(x, "detach"):
        t = x.detach().cpu().contiguous()
        # View as raw bytes so dtypes NumPy lacks (e.g. bfloat16) still hash.
        import torch  # local import: torch is optional for data-only use

        return t.reshape(-1).view(torch.uint8).numpy()
    return np.ascontiguousarray(x)


def array_digest(arrays: Mapping[str, Any]) -> str:
    """Format-independent digest of named arrays or tensors."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        x = arrays[name]
        dtype = str(getattr(x, "dtype", type(x).__name__))
        shape = tuple(getattr(x, "shape", ()))
        h.update(name.encode())
        h.update(b"\0")
        h.update(dtype.encode())
        h.update(repr(shape).encode())
        h.update(_as_numpy(x).tobytes())
    return h.hexdigest()


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: str | os.PathLike[str], obj: Any) -> None:
    text = json.dumps(obj, sort_keys=True, indent=2, allow_nan=True, default=_default) + "\n"
    atomic_write_bytes(path, text.encode())


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path) as f:
        return json.load(f)


def publish_dir(tmp: Path, final: Path) -> None:
    """Atomically move a fully written directory into place.

    A directory either exists complete at ``final`` or not at all. A stale
    ``final`` (e.g. from a crash after a partial manual copy) is replaced.
    """
    for p in tmp.rglob("*"):
        if p.is_file():
            with open(p, "rb") as f:
                os.fsync(f.fileno())
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
