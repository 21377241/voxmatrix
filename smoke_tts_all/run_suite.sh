#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/afs/users/wangyl/VoxMatrix
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${VOXMATRIX_PYTHON:-/mnt/afs/conda/envs/audioeval/bin/python}"
echo "[tts-smoke] python=$PY"
mkdir -p "$ROOT/smoke_tts_all/generated"
"$PY" "$ROOT/smoke_tts_all/build_manifests.py"
"$PY" -m audio_evals.session --config "$ROOT/smoke_tts_all/suite.yaml"
echo "[tts-smoke] done"
ls -la "$ROOT/smoke_tts_all/results" || true
