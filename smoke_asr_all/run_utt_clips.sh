#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/afs/users/wangyl/VoxMatrix
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${VOXMATRIX_PYTHON:-/mnt/afs/conda/envs/audioeval/bin/python}"
echo "[utt-clips-smoke] python=$PY cuda=$CUDA_VISIBLE_DEVICES"
"$PY" -c 'import torch; print("cuda", torch.cuda.is_available())' || true
"$PY" -m audio_evals.session --config "$ROOT/smoke_asr_all/suite_utt_clips.yaml"
echo "[utt-clips-smoke] done"
ls -la "$ROOT/smoke_asr_all/results_utt_clips" || true
for f in aishell_5 alimeeting magicdata_ramc misp sbcsae; do
  echo "===== $f ====="
  cat "$ROOT/smoke_asr_all/results_utt_clips/${f}-overall.json" 2>/dev/null || true
done
