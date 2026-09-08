#!/usr/bin/env bash
set -euo pipefail
ROOT="/mnt/afs/users/wangyl/VoxMatrix"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="$ROOT/envs/qwen3-omni/bin/python"
OUT="$ROOT/smoke_asr_extra/results_native_qwen3/easycom_min_new_tokens.json"

echo "[native-qwen3-min-tokens] python=$PY cuda=$CUDA_VISIBLE_DEVICES"
"$PY" "$ROOT/smoke_asr_extra/scripts/native_qwen3_min_tokens_test.py" \
  --output "$OUT"
echo "[native-qwen3-min-tokens] done"
# compact summary
"$PY" - <<'PY'
import json
from pathlib import Path
p = Path("/mnt/afs/users/wangyl/VoxMatrix/smoke_asr_extra/results_native_qwen3/easycom_min_new_tokens.json")
data = json.loads(p.read_text())
for s in data["samples"]:
    print(f"\n== {s['sample_id']} ref={s['ref_words']} ==")
    for r in s["runs"]:
        print(
            f"  {r['case']:12s} min={r['min_new_tokens']:<4} "
            f"tokens={r['generated_tokens']:<4} words={r['pred_words']:<4} "
            f"cov={r['coverage_pct']}%  tail={r['pred'][-80:]!r}"
        )
PY
