#!/usr/bin/env bash
# CHiME-6 diar smoke with continuous ref-array crops (not utt overlay).
set -euo pipefail
ROOT=/mnt/afs/users/wangyl/VoxMatrix
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${VOXMATRIX_PYTHON:-/mnt/afs/conda/envs/audioeval/bin/python}"
OUT="$ROOT/smoke_diar_all/results_chime6_crop"
echo "[diar-chime6] python=$PY cuda=$CUDA_VISIBLE_DEVICES out=$OUT"
"$PY" -c 'import torch; print("cuda", torch.cuda.is_available())' || true
"$PY" "$ROOT/smoke_diar_all/build_manifests.py" chime_6
mkdir -p "$OUT"
"$PY" -m audio_evals.session --config "$ROOT/smoke_diar_all/suite_chime6.yaml"
echo "[diar-chime6] done"
ls -la "$OUT" || true
for f in "$OUT"/*-overall.json; do
  [ -f "$f" ] || continue
  echo "=== $(basename "$f") ==="
  "$PY" -c "import json,sys; o=json.load(open(sys.argv[1])); print({k:o[k] for k in o if 'der' in k.lower() or 'fail' in k.lower()})" "$f"
done
