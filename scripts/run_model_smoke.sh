#!/usr/bin/env bash
# GPU-node: load each registry model in SMOKE_MODELS and run scripts/smoke_models.py.
set -euo pipefail
ROOT="${VOXMATRIX_ROOT:-/mnt/afs/users/wangyl/VoxMatrix-clean}"
# shellcheck disable=SC1091
source "$ROOT/suites/native/env.sh"
cd "$ROOT"
: "${SMOKE_MODELS:?set SMOKE_MODELS=name1,name2}"
OUT="${SMOKE_OUTPUT_DIR:-$ROOT/outputs/model_smoke/$(date +%Y%m%d_%H%M%S)_${SMOKE_MODELS//,/+}}"
echo "[model-smoke] models=$SMOKE_MODELS out=$OUT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
exec "$VOXMATRIX_PYTHON" -u scripts/smoke_models.py --models "$SMOKE_MODELS" --out "$OUT"
