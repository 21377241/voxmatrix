#!/usr/bin/env bash
# GPU-node: Native paralinguistic_recognition (MMSU closed-label) only.
set -euo pipefail
ROOT="${VOXMATRIX_ROOT:-/mnt/afs/users/wangyl/VoxMatrix-clean}"
HERE="$ROOT/suites/native"
# shellcheck disable=SC1091
source "$HERE/env.sh"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NATIVE_CAPABILITY_FILTER="${NATIVE_CAPABILITY_FILTER:-paralinguistic_recognition}"
unset NATIVE_SLOT_FILTER || true
PY="${VOXMATRIX_PYTHON}"

echo "[native-pr] python=$PY cuda=$CUDA_VISIBLE_DEVICES filter=$NATIVE_CAPABILITY_FILTER"
"$PY" "$HERE/preflight.py"
SUITE_MAIN="$HERE/output/suite.runtime.yaml"
[[ -f "$SUITE_MAIN" ]] || {
  echo "[native-pr] missing suite.runtime.yaml" >&2
  exit 1
}
"$PY" -c 'import torch; assert torch.cuda.is_available(), "no GPU"; print("cuda", torch.cuda.device_count())'
echo "[native-pr] running main suite"
"$PY" -m audio_evals.session --config "$SUITE_MAIN"
echo "[native-pr] done"
ls -la "$HERE/output/main"/native-paralinguistic* 2>/dev/null || true
