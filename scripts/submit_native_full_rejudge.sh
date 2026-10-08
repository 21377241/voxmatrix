#!/usr/bin/env bash
# Usage: scripts/submit_native_full_rejudge.sh <finished run dir> [job submit args...]
#   REJUDGE_BENCHMARKS=native-multi-turn-dialogue--vocalbench RUN_NAME=mt_vb_rejudge \
#     scripts/submit_native_full_rejudge.sh outputs/native_full/multi_turn_dialogue_20261008_155029
#   REJUDGE_LIMIT=20 ...   (smoke check on the first 20 rows)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB="${JOB_BIN:-/mnt/afs/users/wangyl/job}"
SRC="$(cd "${1:?finished full run directory}" && pwd)"
shift
NAME="${RUN_NAME:-$(basename "$SRC" | sed -E 's/_[0-9]{8}_[0-9]{6}$//')_rejudge}"
GPUS="${GPUS:-1}"
EXTRA=()
[[ -n "${REJUDGE_BENCHMARKS:-}" ]] && EXTRA+=(-e "REJUDGE_BENCHMARKS=$REJUDGE_BENCHMARKS")
[[ -n "${REJUDGE_LIMIT:-}" ]] && EXTRA+=(-e "REJUDGE_LIMIT=$REJUDGE_LIMIT")
exec "$JOB" submit "$ROOT/scripts/run_native_full_rejudge.sh" \
  -g "$GPUS" \
  -N "vox-${NAME//_/-}" \
  --project-root "$ROOT" \
  --output-root "$ROOT/outputs/native_full" \
  -e "VOXMATRIX_ROOT=$ROOT" \
  -e "REJUDGE_SRC=$SRC" \
  -e "NATIVE_OUTPUT_DIR=$ROOT/outputs/native_full/${NAME}_$(date +%Y%m%d_%H%M%S)" \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  "$@"
