#!/usr/bin/env bash
# GPU-node: Native sound_event MAVD + SONYC only (closed multilabel rerun).
set -euo pipefail
ROOT="${VOXMATRIX_ROOT:-/mnt/afs/users/wangyl/VoxMatrix-clean}"
HERE="$ROOT/suites/native"
# shellcheck disable=SC1091
source "$HERE/env.sh"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NATIVE_CAPABILITY_FILTER="${NATIVE_CAPABILITY_FILTER:-sound_event}"
export NATIVE_SLOT_FILTER="${NATIVE_SLOT_FILTER:-sound_event__mavd_traffic,sound_event__sonyc_ust_v23}"
PY="${VOXMATRIX_PYTHON}"

echo "[native-se-ml] python=$PY cuda=$CUDA_VISIBLE_DEVICES cap=$NATIVE_CAPABILITY_FILTER slots=$NATIVE_SLOT_FILTER"
"$PY" "$HERE/preflight.py"
SUITE_MAIN="$HERE/output/suite.runtime.yaml"
[[ -f "$SUITE_MAIN" ]] || {
  echo "[native-se-ml] missing suite.runtime.yaml" >&2
  exit 1
}
"$PY" -c 'import torch; assert torch.cuda.is_available(), "no GPU"; print("cuda", torch.cuda.device_count())'
echo "[native-se-ml] running main suite"
"$PY" -m audio_evals.session --config "$SUITE_MAIN"
echo "[native-se-ml] done"
ls -la "$HERE/output/main"/native-sound-event--{mavd,sonyc}* 2>/dev/null || true
