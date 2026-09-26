#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VOXMATRIX_ROOT="$ROOT"
JOB="${JOB_BIN:-/mnt/afs/users/wangyl/job}"
cd "$ROOT"
chmod +x "$ROOT/scripts/run_native_spoken_slu_smoke.sh"
exec "$JOB" submit "$ROOT/scripts/run_native_spoken_slu_smoke.sh" \
  -g "${SLU_SMOKE_GPUS:-1}" \
  -N vox-native-spoken-slu-smoke \
  --project-root "$ROOT" \
  --output-root "$ROOT/outputs/native_spoken_slu_smoke" \
  -e "NATIVE_CAPABILITY_FILTER=${NATIVE_CAPABILITY_FILTER:-spoken_slu}" \
  -e "NATIVE_LIMIT=${NATIVE_LIMIT:-10}" \
  "$@"
