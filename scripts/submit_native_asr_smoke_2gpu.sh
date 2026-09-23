#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VOXMATRIX_ROOT="$ROOT"
JOB="${JOB_BIN:-/mnt/afs/users/wangyl/job}"
cd "$ROOT"
exec "$JOB" submit "$ROOT/scripts/run_native_asr_smoke_2gpu.sh" \
  -g 2 \
  -N vox-native-asr-smoke-2gpu \
  --project-root "$ROOT" \
  --output-root "$ROOT/outputs/native_asr_smoke_rebuilt_2gpu" \
  "$@"
