#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VOXMATRIX_ROOT="$ROOT"
JOB="${JOB_BIN:-/mnt/afs/users/wangyl/job}"
GPUS="${GPUS:-4}"
: "${NATIVE_OUTPUT_DIR:?set NATIVE_OUTPUT_DIR to the infer run dir (outputs/native_runs/<run>)}"
[[ -d "$NATIVE_OUTPUT_DIR/tts" ]] || { echo "no tts outputs under $NATIVE_OUTPUT_DIR" >&2; exit 2; }
cd "$ROOT"
chmod +x "$ROOT/scripts/run_native_speech_vllm_omni_judge.sh"
exec "$JOB" submit "$ROOT/scripts/run_native_speech_vllm_omni_judge.sh" \
  -g "$GPUS" \
  -N vox-omni-speech-judge \
  --project-root "$ROOT" \
  --output-root "$ROOT/outputs/native_speech_vllm_omni_judge" \
  -e "NATIVE_SPEECH_MODEL=${NATIVE_SPEECH_MODEL:-qwen3-omni-speech-vllm-omni-local}" \
  -e "NATIVE_LIMIT=${NATIVE_LIMIT:-2}" \
  -e "NATIVE_SLOT_FILTER=${NATIVE_SLOT_FILTER:-speech_generation__libritts,instruction_following__vocalbench}" \
  -e "VOXMATRIX_SESSION_PHASE=evaluation" \
  -e "NATIVE_OUTPUT_DIR=$NATIVE_OUTPUT_DIR" \
  "$@"
