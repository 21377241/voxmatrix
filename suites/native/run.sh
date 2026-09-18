#!/usr/bin/env bash
# GPU-node one-click native suite. Do not run on the Cursor login pod.
set -euo pipefail
ROOT="${VOXMATRIX_ROOT:-/mnt/afs/users/wangyl/VoxMatrix-clean}"
HERE="$ROOT/suites/native"
# shellcheck disable=SC1091
source "$HERE/env.sh"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${VOXMATRIX_PYTHON}"

echo "[native] python=$PY cuda=$CUDA_VISIBLE_DEVICES"
"$PY" "$HERE/preflight.py"
SUITE_MAIN="$HERE/output/suite.runtime.yaml"
SUITE_TTS="$HERE/output/suite_tts.runtime.yaml"
[[ -f "$SUITE_MAIN" || -f "$SUITE_TTS" ]] || {
  echo "[native] no runtime suite generated" >&2
  exit 1
}

"$PY" -c 'import torch; assert torch.cuda.is_available(), "no GPU"; print("cuda", torch.cuda.device_count())'
if [[ -f "$SUITE_MAIN" ]]; then
  echo "[native] running main suite"
  "$PY" -m audio_evals.session --config "$SUITE_MAIN"
fi
if [[ -f "$SUITE_TTS" ]]; then
  echo "[native] running tts suite"
  "$PY" -m audio_evals.session --config "$SUITE_TTS"
fi
echo "[native] done"
ls -la "$HERE/output" "$HERE/output/main" "$HERE/output/tts" 2>/dev/null || true
