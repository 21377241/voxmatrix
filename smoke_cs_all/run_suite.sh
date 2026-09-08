#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/afs/users/wangyl/VoxMatrix
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${VOXMATRIX_PYTHON:-/mnt/afs/conda/envs/audioeval/bin/python}"
echo "[cs-smoke] python=$PY cuda_devices=$CUDA_VISIBLE_DEVICES"
"$PY" -c 'import torch; print("cuda", torch.cuda.is_available())' || true
"$PY" "$ROOT/smoke_cs_all/build_manifests.py"
"$PY" -m audio_evals.session --config "$ROOT/smoke_cs_all/suite.yaml"
echo "[cs-smoke] done"
ls -la "$ROOT/smoke_cs_all/results" || true
