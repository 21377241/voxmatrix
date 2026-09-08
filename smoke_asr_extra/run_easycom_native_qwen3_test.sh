#!/usr/bin/env bash
set -euo pipefail
ROOT="/mnt/afs/users/wangyl/VoxMatrix"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="$ROOT/envs/qwen3-omni/bin/python"
OUT="$ROOT/smoke_asr_extra/results_native_qwen3/easycom_test_10_00_00_000.json"

echo "[native-qwen3-easycom] python=$PY cuda=$CUDA_VISIBLE_DEVICES"
"$PY" "$ROOT/smoke_asr_extra/scripts/native_qwen3_easycom_test.py" \
  --output "$OUT"
echo "[native-qwen3-easycom] done"
cat "$OUT"
