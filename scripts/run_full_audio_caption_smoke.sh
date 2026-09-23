#!/usr/bin/env bash
set -euo pipefail
ROOT="${VOXMATRIX_ROOT:-/mnt/afs/users/wangyl/VoxMatrix-clean}"
cd "$ROOT"
export HF_HOME="${HF_HOME:-/mnt/afs/users/wangyl/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export FENSE_DATA="${FENSE_DATA:-/mnt/afs/users/wangyl/.cache/fense}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$FENSE_DATA"
source suites/native/env.sh
exec "$VOXMATRIX_PYTHON" -m audio_evals.session --config examples/full_audio_caption_smoke.yaml
