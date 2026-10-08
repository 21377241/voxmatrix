#!/usr/bin/env bash
# Usage: scripts/submit_native_full.sh <capability> [job submit args...]
#   GPUS=4 scripts/submit_native_full.sh acoustic_scene
#   NATIVE_SLOT_FILTER=asr__librispeech scripts/submit_native_full.sh asr
#   RUN_NAME=small_batch1 scripts/submit_native_full.sh sound_event,speaker_count   (one model load)
#   NATIVE_CONCURRENCY=8 scripts/submit_native_full.sh asr   (8 in-flight requests per text replica)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VOXMATRIX_ROOT="$ROOT"
JOB="${JOB_BIN:-/mnt/afs/users/wangyl/job}"
CAP="${1:?capability id(s), comma-separated, e.g. acoustic_scene}"
shift
NAME="${RUN_NAME:-${CAP//,/-}}"
GPUS="${GPUS:-4}"
cd "$ROOT"
chmod +x "$ROOT/scripts/run_native_full.sh"
EXTRA=()
[[ -n "${NATIVE_SLOT_FILTER:-}" ]] && EXTRA+=(-e "NATIVE_SLOT_FILTER=$NATIVE_SLOT_FILTER")
[[ -n "${NATIVE_TTS_REPLICAS:-}" ]] && EXTRA+=(-e "NATIVE_TTS_REPLICAS=$NATIVE_TTS_REPLICAS")
[[ -n "${NATIVE_REPLICAS:-}" ]] && EXTRA+=(-e "NATIVE_REPLICAS=$NATIVE_REPLICAS")
[[ -n "${NATIVE_CONCURRENCY:-}" ]] && EXTRA+=(-e "NATIVE_CONCURRENCY=$NATIVE_CONCURRENCY")
[[ -n "${NATIVE_FULL_LIMIT:-}" ]] && EXTRA+=(-e "NATIVE_FULL_LIMIT=$NATIVE_FULL_LIMIT")
exec "$JOB" submit "$ROOT/scripts/run_native_full.sh" \
  -g "$GPUS" \
  -N "vox-full-${NAME//_/-}" \
  --project-root "$ROOT" \
  --output-root "$ROOT/outputs/native_full" \
  -e "NATIVE_CAPABILITY_FILTER=$CAP" \
  -e "NATIVE_OUTPUT_DIR=$ROOT/outputs/native_full/${NAME}_$(date +%Y%m%d_%H%M%S)" \
  -e "QWEN3_OMNI_DISABLE_CUSTOM_ALL_REDUCE=true" \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  "$@"
