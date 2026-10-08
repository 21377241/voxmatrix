#!/usr/bin/env bash
# Phase 1: speech SUT inference only (vLLM-Omni). Writes jsonl + AFS wavs; no judge.
set -euo pipefail
ROOT="${VOXMATRIX_ROOT:-/mnt/afs/users/wangyl/VoxMatrix-clean}"
HERE="$ROOT/suites/native"
# shellcheck disable=SC1091
source "$HERE/env.sh"
cd "$ROOT"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi
export NATIVE_SPEECH_MODEL="${NATIVE_SPEECH_MODEL:-qwen3-omni-speech-vllm-omni-local}"
export NATIVE_LIMIT="${NATIVE_LIMIT:-2}"
export NATIVE_SLOT_FILTER="${NATIVE_SLOT_FILTER:-speech_generation__libritts,instruction_following__vocalbench}"
unset NATIVE_CAPABILITY_FILTER || true
export VOXMATRIX_SESSION_PHASE=inference
PY="${VOXMATRIX_PYTHON}"

ENV_OMNI="${QWEN3_VLLM_OMNI_ENV:-/mnt/afs/users/wangyl/qwen3_omni_long_audio/envs/vllm_omni}"
if [[ ! -x "$ENV_OMNI/bin/python" ]]; then
  echo "missing vllm-omni env: $ENV_OMNI" >&2
  exit 2
fi

WAV_DIR="$NATIVE_OUTPUT_DIR/tts/speech_wavs"
mkdir -p "$WAV_DIR"

echo "[native-speech-infer] speech_model=$NATIVE_SPEECH_MODEL limit=$NATIVE_LIMIT"
echo "[native-speech-infer] slots=$NATIVE_SLOT_FILTER phase=inference wav_dir=$WAV_DIR"
"$ENV_OMNI/bin/python" -c 'import vllm_omni,vllm,transformers; print("vllm_omni", getattr(vllm_omni,"__version__", "?"), "vllm", vllm.__version__, "tf", transformers.__version__)'
"$PY" "$HERE/preflight.py"
SUITE_TTS="$NATIVE_OUTPUT_DIR/suite_tts.runtime.yaml"
[[ -f "$SUITE_TTS" ]] || { echo "missing $SUITE_TTS" >&2; exit 1; }
"$PY" -c 'import torch; assert torch.cuda.is_available(); print("cuda", torch.cuda.device_count())'
"$PY" -m audio_evals.session --config "$SUITE_TTS" --phase inference
echo "[native-speech-infer] done"
ls -la "$NATIVE_OUTPUT_DIR/tts" | head -30
ls -la "$WAV_DIR" | head -30
