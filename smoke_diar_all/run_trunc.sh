#!/usr/bin/env bash
# Re-infer alimeeting + chime_6 with duration-constrained prompt (truncation check).
set -euo pipefail
ROOT=/mnt/afs/users/wangyl/VoxMatrix
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${VOXMATRIX_PYTHON:-/mnt/afs/conda/envs/audioeval/bin/python}"
OUT="$ROOT/smoke_diar_all/results_trunc"
echo "[diar-trunc] python=$PY cuda_devices=$CUDA_VISIBLE_DEVICES out=$OUT"
"$PY" -c 'import torch; print("cuda", torch.cuda.is_available())' || true
# clips/manifests already exist; only ensure duration field is present
"$PY" - <<'PY'
import json
from pathlib import Path
root = Path("/mnt/afs/users/wangyl/VoxMatrix/smoke_diar_all")
for name in ("alimeeting", "chime_6"):
    p = root / name / "manifest.jsonl"
    rows = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        o["audio_duration_seconds"] = float(o.get("audio_duration_seconds") or 30.0)
        rows.append(o)
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"manifest ok {name}: {len(rows)}", flush=True)
PY
mkdir -p "$OUT"
"$PY" -m audio_evals.session --config "$ROOT/smoke_diar_all/suite_trunc.yaml"
echo "[diar-trunc] done"
ls -la "$OUT" || true
for f in "$OUT"/*-overall.json; do
  [ -f "$f" ] || continue
  echo "=== $(basename "$f") ==="
  "$PY" -c "import json,sys; o=json.load(open(sys.argv[1])); print({k:o[k] for k in o if 'der' in k or 'fail' in k or 'trunc' in k or 'salv' in k})" "$f"
done
