#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/afs/users/wangyl/VoxMatrix
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${VOXMATRIX_PYTHON:-/mnt/afs/conda/envs/audioeval/bin/python}"
echo "[mmsu-open] python=$PY cuda_devices=$CUDA_VISIBLE_DEVICES"
"$PY" -c 'import torch; print("cuda", torch.cuda.is_available())' || true
"$PY" "$ROOT/smoke_st_all/build_manifests.py"
head -1 "$ROOT/smoke_st_all/mmsu_open/manifest.jsonl"
"$PY" -m audio_evals.session --config "$ROOT/smoke_st_all/suite_mmsu_open.yaml"
echo "[mmsu-open] done"
cat "$ROOT/smoke_st_all/results_mmsu_open/mmsu_open-overall.json" || true
