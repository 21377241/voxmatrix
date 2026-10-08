#!/usr/bin/env bash
# GPU-node: Full-index run for the slots selected by NATIVE_CAPABILITY_FILTER / NATIVE_SLOT_FILTER.
# Clip slots read mesh_eval/data/full/slot_map.yaml, every slot runs without a row limit.
set -euo pipefail
ROOT="${VOXMATRIX_ROOT:-/mnt/afs/users/wangyl/VoxMatrix-clean}"
HERE="$ROOT/suites/native"
if [[ -z "${NATIVE_CAPABILITY_FILTER:-}" && -z "${NATIVE_SLOT_FILTER:-}" ]]; then
  echo "[native-full] set NATIVE_CAPABILITY_FILTER or NATIVE_SLOT_FILTER (one capability per job)" >&2
  exit 2
fi
export NATIVE_DATA=full
export NATIVE_RUNS_ROOT="${NATIVE_RUNS_ROOT:-$ROOT/outputs/native_full}"
# shellcheck disable=SC1091
source "$HERE/env.sh"
cd "$ROOT"
export QWEN3_OMNI_DISABLE_CUSTOM_ALL_REDUCE="${QWEN3_OMNI_DISABLE_CUSTOM_ALL_REDUCE:-true}"
GPU_COUNT="$("$VOXMATRIX_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "no GPU"; print(torch.cuda.device_count())')"
export NATIVE_REPLICAS="${NATIVE_REPLICAS:-$GPU_COUNT}"
PY="${VOXMATRIX_PYTHON}"

echo "[native-full] caps=${NATIVE_CAPABILITY_FILTER:-} slots=${NATIVE_SLOT_FILTER:-} gpus=$GPU_COUNT replicas=$NATIVE_REPLICAS concurrency=${NATIVE_CONCURRENCY:-1} full_limit=${NATIVE_FULL_LIMIT:-0} output=$NATIVE_OUTPUT_DIR"
"$PY" "$HERE/preflight.py"
SUITE_MAIN="$NATIVE_OUTPUT_DIR/suite.runtime.yaml"
SUITE_TTS="$NATIVE_OUTPUT_DIR/suite_tts.runtime.yaml"
[[ -f "$SUITE_MAIN" || -f "$SUITE_TTS" ]] || {
  echo "[native-full] no runtime suite generated" >&2
  exit 1
}
if [[ -f "$SUITE_MAIN" ]]; then
  echo "[native-full] running main suite"
  "$PY" -m audio_evals.session --config "$SUITE_MAIN"
fi
if [[ -f "$SUITE_TTS" ]]; then
  echo "[native-full] running tts suite"
  "$PY" -m audio_evals.session --config "$SUITE_TTS"
fi
echo "[native-full] done"
ls -la "$NATIVE_OUTPUT_DIR/main" "$NATIVE_OUTPUT_DIR/tts" 2>/dev/null || true
