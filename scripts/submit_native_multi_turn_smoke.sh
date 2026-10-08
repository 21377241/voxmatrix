#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VOXMATRIX_ROOT="$ROOT"
JOB="${JOB_BIN:-/mnt/afs/users/wangyl/job}"
cd "$ROOT"
exec "$JOB" submit "$ROOT/scripts/run_native_multi_turn_smoke.sh" \
  -g 2 \
  -N vox-native-multi-turn-smoke \
  --project-root "$ROOT" \
  --output-root "$ROOT/outputs/native_runs" \
  "$@"
