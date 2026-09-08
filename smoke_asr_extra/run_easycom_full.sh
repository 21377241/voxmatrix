#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/afs/users/wangyl/VoxMatrix
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${VOXMATRIX_PYTHON:-/mnt/afs/conda/envs/audioeval/bin/python}"
echo "[easycom-full-prompt] python=$PY cuda=$CUDA_VISIBLE_DEVICES"
"$PY" -c 'import torch; print("cuda", torch.cuda.is_available())' || true
"$PY" -m audio_evals.session --config "$ROOT/smoke_asr_extra/suite_easycom_full.yaml"
echo "[easycom-full-prompt] done"
ls -la "$ROOT/smoke_asr_extra/results_full_prompt" || true
cat "$ROOT/smoke_asr_extra/results_full_prompt/easycom-overall.json" 2>/dev/null || true
