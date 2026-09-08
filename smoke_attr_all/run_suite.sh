#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/afs/users/wangyl/VoxMatrix
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${VOXMATRIX_PYTHON:-/mnt/afs/conda/envs/audioeval/bin/python}"
echo "[attr-smoke] python=$PY cuda_devices=$CUDA_VISIBLE_DEVICES"
"$PY" -c 'import torch; print("cuda", torch.cuda.is_available())' || true
"$PY" "$ROOT/smoke_attr_all/build_manifests.py"
"$PY" -m audio_evals.session --config "$ROOT/smoke_attr_all/suite.yaml"
echo "[attr-smoke] done"
ls -la "$ROOT/smoke_attr_all/results" || true
