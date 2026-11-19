#!/usr/bin/env bash
# Cost control for on-demand GPU nodes (docs/blueprint.md §11.8).
#
#   AFTERIMAGE_SHUTDOWN_CMD="<provider terminate call>" \
#     scripts/run_then_shutdown.sh configs/pilot.toml [launch options...]
#
# Runs `afterimage launch`, then executes $AFTERIMAGE_SHUTDOWN_CMD whether the
# run succeeded or not, so an idle node never keeps billing. Without that
# variable nothing is shut down. Checkpoints must live on storage that
# survives the node; every block is resumable.
set -uo pipefail
cd "$(dirname "$0")/.."
cfg="${1:?usage: run_then_shutdown.sh <config> [launch options]}"
shift
extra="${AFTERIMAGE_TORCH:-cu130}"
uv run --frozen --extra "$extra" afterimage launch --config "$cfg" "$@"
rc=$?
echo "launch exited with status $rc"
if [ -n "${AFTERIMAGE_SHUTDOWN_CMD:-}" ]; then
  echo "running shutdown command"
  sh -c "$AFTERIMAGE_SHUTDOWN_CMD"
fi
exit "$rc"
