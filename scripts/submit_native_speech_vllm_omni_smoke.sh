#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VOXMATRIX_ROOT="$ROOT"
JOB="${JOB_BIN:-/mnt/afs/users/wangyl/job}"
GPUS="${GPUS:-4}"
cd "$ROOT"
chmod +x "$ROOT/scripts/run_native_speech_vllm_omni_smoke.sh"
exec "$JOB" submit "$ROOT/scripts/run_native_speech_vllm_omni_smoke.sh" \
  -g "$GPUS" \
  -N vox-native-speech-vllm-omni-smoke \
  --project-root "$ROOT" \
  --output-root "$ROOT/outputs/native_speech_vllm_omni_smoke" \
  -e "NATIVE_SPEECH_MODEL=${NATIVE_SPEECH_MODEL:-qwen3-omni-speech-vllm-omni-local}" \
  -e "NATIVE_LIMIT=${NATIVE_LIMIT:-2}" \
  -e "NATIVE_SLOT_FILTER=${NATIVE_SLOT_FILTER:-speech_generation__libritts,instruction_following__vocalbench}" \
  -e "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.88}" \
  "$@"
