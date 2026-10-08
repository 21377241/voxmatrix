#!/usr/bin/env bash
# GPU-node: judge-only MTalk comparison (soft / strict / strict_force) on saved SUT outputs.
set -euo pipefail
ROOT="${VOXMATRIX_ROOT:-/mnt/afs/users/wangyl/VoxMatrix-clean}"
# shellcheck disable=SC1091
source "$ROOT/suites/native/env.sh"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
RUN_DIR="${MTALK_RUN_DIR:?set MTALK_RUN_DIR to a native run dir with tts/ MTalk outputs}"
OUT="${MTALK_COMPARE_OUT:-$ROOT/outputs/mtalk_judge_compare/$(date +%Y%m%d_%H%M%S)}"
echo "[mtalk-compare] run_dir=$RUN_DIR out=$OUT variants=${MTALK_VARIANTS:-all}"
"$VOXMATRIX_PYTHON" scripts/compare_mtalk_judge_variants.py \
  --run-dir "$RUN_DIR" --out "$OUT" \
  ${MTALK_VARIANTS:+--variants "$MTALK_VARIANTS"}
echo "[mtalk-compare] done"
