#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VOXMATRIX_ROOT="$ROOT"
JOB="${JOB_BIN:-/mnt/afs/users/wangyl/job}"
GPUS="${GPUS:-2}"
export NATIVE_OUTPUT_DIR="${NATIVE_OUTPUT_DIR:-$ROOT/outputs/native_runs/$(date +%Y%m%d_%H%M%S)_speech_infer_$$}"
echo "[submit] run dir: $NATIVE_OUTPUT_DIR"
cd "$ROOT"
chmod +x "$ROOT/scripts/run_native_speech_vllm_omni_infer.sh"
exec "$JOB" submit "$ROOT/scripts/run_native_speech_vllm_omni_infer.sh" \
  -g "$GPUS" \
  -N vox-omni-speech-infer \
  --project-root "$ROOT" \
  --output-root "$ROOT/outputs/native_speech_vllm_omni_infer" \
  -e "NATIVE_SPEECH_MODEL=${NATIVE_SPEECH_MODEL:-qwen3-omni-speech-vllm-omni-local}" \
  -e "NATIVE_LIMIT=${NATIVE_LIMIT:-2}" \
  -e "NATIVE_SLOT_FILTER=${NATIVE_SLOT_FILTER:-speech_generation__libritts,instruction_following__vocalbench}" \
  -e "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.88}" \
  -e "VOXMATRIX_SESSION_PHASE=inference" \
  -e "NATIVE_OUTPUT_DIR=$NATIVE_OUTPUT_DIR" \
  "$@"
