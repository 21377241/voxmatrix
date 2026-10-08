#!/usr/bin/env bash
# GPU-node: windowed on-device meeting eval (long_form_asr / diarization / speaker_attribution, S/M/L/XL).
# Defaults run the formal set (20 families x 4 buckets = 80 rows per slot).
# Smoke: WINDOWED_LIMIT=8 (= 2 complete families). Output dir comes from env.sh (one per run).
set -euo pipefail
ROOT="${VOXMATRIX_ROOT:-/mnt/afs/users/wangyl/VoxMatrix-clean}"
HERE="$ROOT/suites/native"
# shellcheck disable=SC1091
source "$HERE/env.sh"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset NATIVE_CAPABILITY_FILTER || true
WINDOWED_SLOTS="long_form_asr__aishell_4,long_form_asr__alimeeting,long_form_asr__ami,long_form_asr__chime_6,diarization__aishell_4,diarization__alimeeting,diarization__ami,diarization__chime_6,speaker_attribution__aishell_4,speaker_attribution__alimeeting,speaker_attribution__ami,speaker_attribution__chime_6"
export NATIVE_LIMIT="${WINDOWED_LIMIT:-80}"
export NATIVE_SLOT_FILTER="${WINDOWED_SLOTS_FILTER:-$WINDOWED_SLOTS}"
PY="${VOXMATRIX_PYTHON}"

ENV_OMNI="${QWEN3_VLLM_OMNI_ENV:-/mnt/afs/users/wangyl/qwen3_omni_long_audio/envs/vllm_omni}"
echo "[windowed] python=$PY cuda=$CUDA_VISIBLE_DEVICES limit=$NATIVE_LIMIT"
echo "[windowed] slots=$NATIVE_SLOT_FILTER"
echo "[windowed] output=$NATIVE_OUTPUT_DIR"
[[ -x "$ENV_OMNI/bin/python" ]] || {
  echo "missing vllm-omni env: $ENV_OMNI" >&2
  exit 2
}
"$ENV_OMNI/bin/python" -c 'import vllm_omni,vllm; print("vllm_omni", getattr(vllm_omni,"__version__","?"), "vllm", vllm.__version__)'
"$PY" -c 'import torch; assert torch.cuda.is_available(), "no GPU"; print("cuda", torch.cuda.device_count())'
"$PY" "$HERE/preflight.py"
SUITE_MAIN="$NATIVE_OUTPUT_DIR/suite.runtime.yaml"
[[ -f "$SUITE_MAIN" ]] || {
  echo "[windowed] missing $SUITE_MAIN" >&2
  exit 1
}
echo "[windowed] phase=infer+eval"
"$PY" -m audio_evals.session --config "$SUITE_MAIN"
echo "[windowed] done -> $NATIVE_OUTPUT_DIR"
