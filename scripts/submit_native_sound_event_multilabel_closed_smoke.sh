#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VOXMATRIX_ROOT="$ROOT"
JOB="${JOB_BIN:-/mnt/afs/users/wangyl/job}"
cd "$ROOT"
exec "$JOB" submit "$ROOT/scripts/run_native_sound_event_multilabel_closed_smoke.sh" \
  -g 1 \
  -N vox-native-se-ml-closed \
  --project-root "$ROOT" \
  --output-root "$ROOT/outputs/native_sound_event_ml_closed_smoke" \
  "$@"
