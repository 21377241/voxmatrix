#!/usr/bin/env bash
set -euo pipefail
ROOT="${VOXMATRIX_ROOT:-/mnt/afs/users/wangyl/VoxMatrix-clean}"
cd "$ROOT"
source suites/native/env.sh
exec "$VOXMATRIX_PYTHON" -m audio_evals.session --config examples/native_code_switch_smoke.yaml
