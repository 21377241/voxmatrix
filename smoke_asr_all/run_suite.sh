#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/afs/users/wangyl/VoxMatrix
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${VOXMATRIX_PYTHON:-/mnt/afs/conda/envs/audioeval/bin/python}"
echo "[suite] python=$PY cuda_devices=$CUDA_VISIBLE_DEVICES"
"$PY" -c 'import torch; print("cuda", torch.cuda.is_available())' || true
"$PY" -m audio_evals.session --config "$ROOT/smoke_asr_all/suite.yaml"
echo "[suite] done"
ls -la "$ROOT/smoke_asr_all/results" || true
