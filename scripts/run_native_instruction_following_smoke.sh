#!/usr/bin/env bash
# GPU-node: Native instruction_following — VoiceBench (text IFEval) + VocalBench/zh (speech IF).
# Suites are split: main = VoiceBench; tts = VocalBench speech+extract_audio+audio rubric.
set -euo pipefail
ROOT="${VOXMATRIX_ROOT:-/mnt/afs/users/wangyl/VoxMatrix-clean}"
HERE="$ROOT/suites/native"
# shellcheck disable=SC1091
source "$HERE/env.sh"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NATIVE_SLOT_FILTER="${NATIVE_SLOT_FILTER:-instruction_following__voicebench,instruction_following__vocalbench,instruction_following__vocalbench_zh}"
unset NATIVE_CAPABILITY_FILTER || true
PY="${VOXMATRIX_PYTHON}"

echo "[native-if] python=$PY cuda=$CUDA_VISIBLE_DEVICES slots=$NATIVE_SLOT_FILTER"
"$PY" "$HERE/preflight.py"
SUITE_MAIN="$NATIVE_OUTPUT_DIR/suite.runtime.yaml"
SUITE_TTS="$NATIVE_OUTPUT_DIR/suite_tts.runtime.yaml"
[[ -f "$SUITE_MAIN" || -f "$SUITE_TTS" ]] || {
  echo "[native-if] no runtime suite generated" >&2
  exit 1
}
"$PY" -c 'import torch; assert torch.cuda.is_available(), "no GPU"; print("cuda", torch.cuda.device_count())'
if [[ -f "$SUITE_MAIN" ]]; then
  echo "[native-if] running VoiceBench text suite"
  "$PY" -m audio_evals.session --config "$SUITE_MAIN"
fi
if [[ -f "$SUITE_TTS" ]]; then
  echo "[native-if] running VocalBench speech suite (two_phase)"
  "$PY" -m audio_evals.session --config "$SUITE_TTS"
fi
echo "[native-if] done"
ls -la "$NATIVE_OUTPUT_DIR/main"/native-instruction* "$NATIVE_OUTPUT_DIR/tts"/native-instruction* 2>/dev/null || true
