#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/afs/users/wangyl/VoxMatrix
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${VOXMATRIX_PYTHON:-/mnt/afs/conda/envs/audioeval/bin/python}"
echo "[cv-en] python=$PY"
"$PY" -c 'import torch; print("cuda", torch.cuda.is_available())' || true
# archive previous zh-HK results if present
RES="$ROOT/smoke_asr_all/results"
if [[ -f "$RES/common_voice.jsonl" ]] && ! [[ -f "$RES/common_voice_zhHK_bak.jsonl" ]]; then
  cp -a "$RES/common_voice.jsonl" "$RES/common_voice_zhHK_bak.jsonl"
  [[ -f "$RES/common_voice-overall.json" ]] && cp -a "$RES/common_voice-overall.json" "$RES/common_voice-overall_zhHK_bak.json"
  echo "[cv-en] archived previous common_voice results as *_zhHK_bak"
fi
"$PY" -m audio_evals.session --config "$ROOT/smoke_asr_all/suite_common_voice_en.yaml"
echo "[cv-en] done"
cat "$RES/common_voice-overall.json" || true
