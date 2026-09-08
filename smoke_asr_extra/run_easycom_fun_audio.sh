#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/afs/users/wangyl/VoxMatrix
cd "$ROOT"
bash "$ROOT/smoke_asr_extra/prepare_fun_audio_chat_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${VOXMATRIX_PYTHON:-/mnt/afs/conda/envs/audioeval/bin/python}"
echo "[easycom-fun-audio-chat] python=$PY cuda=$CUDA_VISIBLE_DEVICES"
"$PY" -c 'import torch; print("cuda", torch.cuda.is_available())' || true
"$PY" -m audio_evals.session --config "$ROOT/smoke_asr_extra/suite_easycom_fun_audio.yaml"
echo "[easycom-fun-audio-chat] done"
ls -la "$ROOT/smoke_asr_extra/results_fun_audio_chat" || true
cat "$ROOT/smoke_asr_extra/results_fun_audio_chat/easycom-overall.json" 2>/dev/null || true
