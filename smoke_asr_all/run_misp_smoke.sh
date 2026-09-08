#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/afs/users/wangyl/VoxMatrix
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${VOXMATRIX_PYTHON:-/mnt/afs/conda/envs/audioeval/bin/python}"
echo "[misp-smoke] python=$PY cuda=$CUDA_VISIBLE_DEVICES"
echo "[misp-smoke] manifest=$ROOT/smoke_asr_all/misp/manifest.jsonl"
head -1 "$ROOT/smoke_asr_all/misp/manifest.jsonl"
"$PY" -c 'import torch; print("cuda", torch.cuda.is_available())' || true
"$PY" -m audio_evals.session --config "$ROOT/smoke_asr_all/suite_misp_smoke.yaml"
echo "[misp-smoke] done"
ls -la "$ROOT/smoke_asr_all/results_misp_reslice" || true
echo "===== misp ====="
cat "$ROOT/smoke_asr_all/results_misp_reslice/misp-overall.json" 2>/dev/null || true
