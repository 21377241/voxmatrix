#!/usr/bin/env bash
set -euo pipefail
ROOT="/mnt/afs/users/wangyl/VoxMatrix"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="$ROOT/envs/qwen3-omni/bin/python"
OUT_DIR="$ROOT/smoke_asr_extra/results_native_qwen3/duration_sweep_easycom_000"

echo "[duration-sweep] python=$PY cuda=$CUDA_VISIBLE_DEVICES"
"$PY" "$ROOT/smoke_asr_extra/scripts/native_qwen3_duration_sweep.py" \
  --output "$OUT_DIR/results.json"
echo "[duration-sweep] done"
cat "$OUT_DIR/results.json"
