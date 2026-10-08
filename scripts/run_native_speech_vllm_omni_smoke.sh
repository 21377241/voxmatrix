#!/usr/bin/env bash
# Smoke: speech_generation (LibriTTS) + VocalBench IF via vLLM-Omni speech backend.
# Does not change default HF speech_model; override via NATIVE_SPEECH_MODEL.
set -euo pipefail
ROOT="${VOXMATRIX_ROOT:-/mnt/afs/users/wangyl/VoxMatrix-clean}"
HERE="$ROOT/suites/native"
# shellcheck disable=SC1091
source "$HERE/env.sh"
cd "$ROOT"

# Do not pin CUDA_VISIBLE_DEVICES=0 — allocator should expose all -g cards.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi
export NATIVE_SPEECH_MODEL="${NATIVE_SPEECH_MODEL:-qwen3-omni-speech-vllm-omni-local}"
export NATIVE_LIMIT="${NATIVE_LIMIT:-2}"
export NATIVE_SLOT_FILTER="${NATIVE_SLOT_FILTER:-speech_generation__libritts,instruction_following__vocalbench}"
unset NATIVE_CAPABILITY_FILTER || true
PY="${VOXMATRIX_PYTHON}"

ENV_OMNI="${QWEN3_VLLM_OMNI_ENV:-/mnt/afs/users/wangyl/qwen3_omni_long_audio/envs/vllm_omni}"
if [[ ! -x "$ENV_OMNI/bin/python" ]]; then
  echo "missing vllm-omni env: $ENV_OMNI (run qwen3_omni_long_audio/jobs/prepare_vllm_omni_env.sh)" >&2
  exit 2
fi

echo "[native-speech-vllm-omni] speech_model=$NATIVE_SPEECH_MODEL limit=$NATIVE_LIMIT slots=$NATIVE_SLOT_FILTER"
echo "[native-speech-vllm-omni] omni_env=$ENV_OMNI"
echo "[native-speech-vllm-omni] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
"$ENV_OMNI/bin/python" -c 'import vllm_omni,vllm,transformers; print("vllm_omni", getattr(vllm_omni,"__version__", "?"), "vllm", vllm.__version__, "tf", transformers.__version__)'
"$PY" "$HERE/preflight.py"
SUITE_TTS="$NATIVE_OUTPUT_DIR/suite_tts.runtime.yaml"
[[ -f "$SUITE_TTS" ]] || { echo "missing $SUITE_TTS" >&2; exit 1; }
"$PY" -c 'import torch; assert torch.cuda.is_available(); print("cuda", torch.cuda.device_count())'
echo "[native-speech-vllm-omni] running tts suite (LibriTTS + VocalBench IF)"
"$PY" -m audio_evals.session --config "$SUITE_TTS"
echo "[native-speech-vllm-omni] done"
ls -la "$NATIVE_OUTPUT_DIR/tts" 2>/dev/null | head -30
