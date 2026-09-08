#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/afs/users/wangyl/VoxMatrix
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${VOXMATRIX_PYTHON:-/mnt/afs/conda/envs/audioeval/bin/python}"
echo "[spoof-smoke] python=$PY"
"$PY" "$ROOT/smoke_spoof_all/build_manifests.py"
"$PY" -m audio_evals.session --config "$ROOT/smoke_spoof_all/suite.yaml"
echo "[spoof-smoke] done"
ls -la "$ROOT/smoke_spoof_all/results" || true
