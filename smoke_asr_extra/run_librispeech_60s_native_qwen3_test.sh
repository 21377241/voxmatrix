#!/usr/bin/env bash
set -euo pipefail
ROOT="/mnt/afs/users/wangyl/VoxMatrix"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="$ROOT/envs/qwen3-omni/bin/python"
OUT_DIR="$ROOT/smoke_asr_extra/results_native_qwen3"
AUDIO="$OUT_DIR/librispeech_60s_single.wav"
META="$OUT_DIR/librispeech_60s_single.meta.json"
RESULT="$OUT_DIR/librispeech_60s_single.json"

echo "[native-qwen3-librispeech-60s] prepare clip"
"$PY" "$ROOT/smoke_asr_extra/scripts/prepare_librispeech_60s_clip.py"

echo "[native-qwen3-librispeech-60s] python=$PY cuda=$CUDA_VISIBLE_DEVICES"
"$PY" "$ROOT/smoke_asr_extra/scripts/native_qwen3_asr_test.py" \
  --audio "$AUDIO" \
  --meta "$META" \
  --output "$RESULT" \
  --label "librispeech_60s_single_speaker" \
  --prompt-mode all

echo "[native-qwen3-librispeech-60s] done"
cat "$RESULT"
