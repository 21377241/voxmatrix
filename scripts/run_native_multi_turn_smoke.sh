#!/usr/bin/env bash
# GPU-node: Native multi_turn_dialogue — official-protocol judges with the local Thinking judge.
# main = VocalBench/zh, VoiceBench MTBench, AudioAgentBench (text SUT); tts = MTalk (speech SUT).
set -euo pipefail
ROOT="${VOXMATRIX_ROOT:-/mnt/afs/users/wangyl/VoxMatrix-clean}"
HERE="$ROOT/suites/native"
# shellcheck disable=SC1091
source "$HERE/env.sh"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NATIVE_SLOT_FILTER="${NATIVE_SLOT_FILTER:-multi_turn_dialogue__vocalbench,multi_turn_dialogue__vocalbench_zh,multi_turn_dialogue__voicebench,multi_turn_dialogue__audioagentbench_suite,multi_turn_dialogue__mtalk_bench}"
export NATIVE_DATA=full
export NATIVE_FULL_LIMIT="${NATIVE_FULL_LIMIT:-${NATIVE_LIMIT:-10}}"
unset NATIVE_CAPABILITY_FILTER || true
PY="${VOXMATRIX_PYTHON}"

echo "[native-mt] python=$PY cuda=$CUDA_VISIBLE_DEVICES slots=$NATIVE_SLOT_FILTER limit=$NATIVE_LIMIT output=$NATIVE_OUTPUT_DIR"
"$PY" "$HERE/preflight.py"
SUITE_MAIN="$NATIVE_OUTPUT_DIR/suite.runtime.yaml"
SUITE_TTS="$NATIVE_OUTPUT_DIR/suite_tts.runtime.yaml"
[[ -f "$SUITE_MAIN" || -f "$SUITE_TTS" ]] || {
  echo "[native-mt] no runtime suite generated" >&2
  exit 1
}
"$PY" -c 'import torch; assert torch.cuda.is_available(), "no GPU"; print("cuda", torch.cuda.device_count())'
if [[ -f "$SUITE_MAIN" ]]; then
  echo "[native-mt] running text multi-turn suite (two_phase)"
  "$PY" -m audio_evals.session --config "$SUITE_MAIN"
fi
if [[ -f "$SUITE_TTS" ]]; then
  echo "[native-mt] running MTalk speech suite (two_phase)"
  "$PY" -m audio_evals.session --config "$SUITE_TTS"
fi
echo "[native-mt] done"
ls -la "$NATIVE_OUTPUT_DIR/main" "$NATIVE_OUTPUT_DIR/tts" 2>/dev/null || true
