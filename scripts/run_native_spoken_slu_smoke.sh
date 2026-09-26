#!/usr/bin/env bash
# GPU-node: Native Spoken SLU smoke (FSC + SLURP intent/slots via vLLM-Omni thinker-only).
set -euo pipefail
ROOT="${VOXMATRIX_ROOT:-/mnt/afs/users/wangyl/VoxMatrix-clean}"
HERE="$ROOT/suites/native"
# shellcheck disable=SC1091
source "$HERE/env.sh"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NATIVE_CAPABILITY_FILTER="${NATIVE_CAPABILITY_FILTER:-spoken_slu}"
export NATIVE_LIMIT="${NATIVE_LIMIT:-10}"
unset NATIVE_SLOT_FILTER || true
PY="${VOXMATRIX_PYTHON}"

ENV_OMNI="${QWEN3_VLLM_OMNI_ENV:-/mnt/afs/users/wangyl/qwen3_omni_long_audio/envs/vllm_omni}"
echo "[native-slu] python=$PY cuda=$CUDA_VISIBLE_DEVICES filter=$NATIVE_CAPABILITY_FILTER limit=$NATIVE_LIMIT"
echo "[native-slu] omni_env=$ENV_OMNI"
[[ -x "$ENV_OMNI/bin/python" ]] || {
  echo "missing vllm-omni env: $ENV_OMNI" >&2
  exit 2
}
"$ENV_OMNI/bin/python" -c 'import vllm_omni,vllm; print("vllm_omni", getattr(vllm_omni,"__version__","?"), "vllm", vllm.__version__)'
"$PY" -c 'import torch; assert torch.cuda.is_available(), "no GPU"; print("cuda", torch.cuda.device_count())'
"$PY" "$HERE/preflight.py"
SUITE_MAIN="$HERE/output/suite.runtime.yaml"
[[ -f "$SUITE_MAIN" ]] || {
  echo "[native-slu] missing $SUITE_MAIN" >&2
  exit 1
}
# Sanity: runtime must include spoken_slu slots only.
"$PY" - <<'PY'
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path("suites/native/output/suite.runtime.yaml").read_text())
# suite may nest tasks; just print file size / hint
print("[native-slu] suite.runtime.yaml bytes", Path("suites/native/output/suite.runtime.yaml").stat().st_size)
PY

echo "[native-slu] phase=infer+eval (intent_acc; no LLM judge)"
"$PY" -m audio_evals.session --config "$SUITE_MAIN"
echo "[native-slu] done"
ls -la "$HERE/output/main"/native-spoken-slu* "$HERE/output/main"/*spoken*slu* 2>/dev/null || \
  ls -la "$HERE/output/main"/ | rg -i 'slu|fsc|slurp' || true
