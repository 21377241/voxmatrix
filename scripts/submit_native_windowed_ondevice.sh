#!/usr/bin/env bash
# Submit the windowed on-device meeting eval. Examples:
#   scripts/submit_native_windowed_ondevice.sh                       # formal, all 12 slots
#   WINDOWED_LIMIT=8 scripts/submit_native_windowed_ondevice.sh      # smoke (2 families/slot)
#   WINDOWED_SLOTS_FILTER=diarization__ami,diarization__chime_6 scripts/submit_native_windowed_ondevice.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VOXMATRIX_ROOT="$ROOT"
JOB="${JOB_BIN:-/mnt/afs/users/wangyl/job}"
cd "$ROOT"
chmod +x "$ROOT/scripts/run_native_windowed_ondevice.sh"
ENV_ARGS=()
[[ -n "${WINDOWED_LIMIT:-}" ]] && ENV_ARGS+=(-e "WINDOWED_LIMIT=$WINDOWED_LIMIT")
[[ -n "${WINDOWED_SLOTS_FILTER:-}" ]] && ENV_ARGS+=(-e "WINDOWED_SLOTS_FILTER=$WINDOWED_SLOTS_FILTER")
exec "$JOB" submit "$ROOT/scripts/run_native_windowed_ondevice.sh" \
  -g "${WINDOWED_GPUS:-1}" \
  -N vox-native-windowed \
  --project-root "$ROOT" \
  --output-root "$ROOT/outputs/native_runs" \
  "${ENV_ARGS[@]+"${ENV_ARGS[@]}"}" \
  "$@"
