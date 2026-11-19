#!/usr/bin/env bash
# Reproduction entry points (docs/blueprint.md §14).
#
#   scripts/reproduce.sh smoke
#       All tests (fast and slow) plus the smoke pipeline on CPU.
#   scripts/reproduce.sh r1 <released runs/<name> dir> [config]
#       Tier R1: rerun every arm from the released fork states; final-state
#       digests must match bit for bit (same GPU model and image).
#   scripts/reproduce.sh r2 [config]
#       Tier R2: full rerun with the released seeds, then the pre-registered
#       confirmatory analysis into results_r2/.
#
# Set AFTERIMAGE_TORCH=cu130 to use the CUDA build of torch (default: cpu).
set -euo pipefail
cd "$(dirname "$0")/.."

extra="${AFTERIMAGE_TORCH:-cpu}"
uv sync --frozen --extra "$extra"
ai() { uv run --frozen --extra "$extra" afterimage "$@"; }

tier="${1:?usage: reproduce.sh smoke | r1 <released-study-dir> [config] | r2 [config]}"
case "$tier" in
  smoke)
    uv run --frozen --extra "$extra" pytest -q
    uv run --frozen --extra "$extra" pytest -q -m slow
    cfg=configs/smoke.toml
    ai data build --config "$cfg"
    ai data audit --config "$cfg"
    ai replay --config "$cfg" --steps 200
    ai run --config "$cfg"
    ai analyze --config "$cfg" --audit
    ;;
  r1)
    released="${2:?r1 needs the released study directory, e.g. artifacts/runs/confirmatory}"
    cfg="${3:-configs/confirmatory.toml}"
    ai data build --config "$cfg"
    ai data verify --config "$cfg"
    ai reproduce-r1 --config "$cfg" --released "$released" --into runs_r1
    ;;
  r2)
    cfg="${2:-configs/confirmatory.toml}"
    ai data build --config "$cfg"
    ai data verify --config "$cfg"
    ai data seal --config "$cfg"
    ai replay --config "$cfg"
    ai run --config "$cfg"
    ai evaluate-sealed --config "$cfg"
    ai analyze --config "$cfg" --confirmatory --out results_r2
    ;;
  *)
    echo "unknown tier: $tier" >&2
    exit 2
    ;;
esac
