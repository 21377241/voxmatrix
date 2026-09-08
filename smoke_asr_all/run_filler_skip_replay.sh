#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/afs/users/wangyl/VoxMatrix
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${VOXMATRIX_PYTHON:-/mnt/afs/conda/envs/audioeval/bin/python}"
REG="$ROOT/smoke_asr_all/registry"
OUT="$ROOT/smoke_asr_all/results_filler_skip"
mkdir -p "$OUT/replay_src"

filter_replay_jsonl() {
  local src="$1"
  local dst="$2"
  "$PY" - <<PY
import json
from pathlib import Path
src = Path(${src@Q})
dst = Path(${dst@Q})
keep = {"prompt", "inference", "post_process"}
with src.open() as fin, dst.open("w") as fout:
    for line in fin:
        ev = json.loads(line)
        if ev.get("type") in keep:
            fout.write(json.dumps(ev, ensure_ascii=False) + "\n")
print(f"filtered {src.name} -> {dst} ({dst.stat().st_size} bytes)")
PY
}

run_replay() {
  local name="$1"
  local dataset="$2"
  local task="$3"
  local src_jsonl="$4"
  local replay_src="$OUT/replay_src/${name}.replay.jsonl"
  local save="$OUT/${name}.jsonl"

  echo "===== replay $name ====="
  filter_replay_jsonl "$src_jsonl" "$replay_src"
  "$PY" -m audio_evals.main \
    --registry_path "$REG" \
    --dataset "$dataset" \
    --task "$task" \
    --model qwen3-omni-local \
    --save "$save" \
    --resume "$replay_src" \
    --replay-only \
    --limit 10
  echo "--- overall ---"
  cat "${save%.jsonl}-overall.json"
  echo
}

run_replay ami smoke-ami-asr10 smoke-task-ami "$ROOT/smoke_asr_all/results/ami.jsonl"
run_replay sbcsae smoke-sbcsae-sliced-asr10 smoke-task-sbcsae-sliced "$ROOT/smoke_asr_all/results/sbcsae.jsonl"
run_replay misp smoke-misp-asr10 smoke-task-misp "$ROOT/smoke_asr_all/results/misp.jsonl"

echo "[filler-skip-replay] done -> $OUT"
