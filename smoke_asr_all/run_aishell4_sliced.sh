#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/afs/users/wangyl/VoxMatrix
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${VOXMATRIX_PYTHON:-/mnt/afs/conda/envs/audioeval/bin/python}"
echo "[aishell4-sliced] python=$PY cuda=$CUDA_VISIBLE_DEVICES"
"$PY" -c 'import torch; print("cuda", torch.cuda.is_available())' || true
"$PY" -m audio_evals.session --config "$ROOT/smoke_asr_all/suite_aishell4_sliced.yaml"
echo "[aishell4-sliced] done"
ls -la "$ROOT/smoke_asr_all/results_utt_clips" || true
cat "$ROOT/smoke_asr_all/results_utt_clips/aishell_4-overall.json" || true
