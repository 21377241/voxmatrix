#!/usr/bin/env bash
set -euo pipefail
# Two-phase judge (A1B-style):
#   1) ASR prefill once
#   2) release GPU memory
#   3) shard Omni Thinking across visible GPUs
echo "[ar-judge-job] start $(date -Is) host=$(hostname) pid=$$"
ROOT="${VOXMATRIX_ROOT:-/mnt/afs/users/wangyl/VoxMatrix-clean}"
cd "$ROOT"
echo "[ar-judge-job] ROOT=$ROOT"
source suites/native/env.sh
export PATH="/mnt/afs/conda/envs/audioeval/bin:/mnt/afs/users/wangyl/bin:${PATH}"
export PYTHONUNBUFFERED=1
export QWEN3_OMNI_DEBUG_STAGE="${QWEN3_OMNI_DEBUG_STAGE:-1}"
echo "[ar-judge-job] python=$VOXMATRIX_PYTHON"
echo "[ar-judge-job] which uv=$(command -v uv || echo MISSING)"
echo "[ar-judge-job] uv version=$(uv --version 2>/dev/null || echo MISSING)"
if ! command -v uv >/dev/null 2>&1; then
  echo "[ar-judge-job] FATAL: uv not on PATH" >&2
  exit 127
fi
if [[ ! -x "${ROOT}/envs/qwen3-omni/bin/python" ]]; then
  echo "[ar-judge-job] FATAL: missing envs/qwen3-omni/bin/python" >&2
  ls -la "${ROOT}/envs" >&2 || true
  exit 127
fi

export JUDGE_ASR_BACKEND="local"
export JUDGE_ASR_DEVICE="cuda"
export JUDGE_WHISPER_PATH="/mnt/afs/models/whisper-large-v3"
export JUDGE_PARAFORMER_PATH="/mnt/afs/models/paraformer-zh"

OUT_DIR="${ROOT}/outputs/native_audio_reasoning_smoke"
PACK="${ROOT}/outputs/audio_reasoning_judge_pack.jsonl"
ASR_CACHE="${OUT_DIR}/judge_asr_cache"
SHARD_DIR="${OUT_DIR}/shards"
FINAL_OUT="${OUT_DIR}/ar-local-llm-judge.jsonl"
mkdir -p "${ASR_CACHE}" "${SHARD_DIR}"

if [[ ! -f "${PACK}" ]]; then
  echo "[ar-judge-job] FATAL: missing pack ${PACK}" >&2
  exit 1
fi

echo "[ar-judge-job] phase=asr-prefill pack=${PACK} cache=${ASR_CACHE} force=${JUDGE_ASR_FORCE:-0}"
PREFILL_ARGS=(--pack "${PACK}" --asr-cache "${ASR_CACHE}")
if [[ "${JUDGE_ASR_FORCE:-0}" == "1" ]]; then
  PREFILL_ARGS+=(--force)
fi
"$VOXMATRIX_PYTHON" -u scripts/prefill_asr_for_judge_pack.py "${PREFILL_ARGS[@]}"

echo "[ar-judge-job] phase=cuda-cleanup"
"$VOXMATRIX_PYTHON" -u - <<'PY'
import gc
try:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"[ar-judge-job] cuda empty_cache ok devices={torch.cuda.device_count()}", flush=True)
    else:
        print("[ar-judge-job] cuda unavailable", flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"[ar-judge-job] cuda cleanup skip: {exc}", flush=True)
PY

export JUDGE_ASR_CACHE_ONLY="${JUDGE_ASR_CACHE_ONLY:-1}"
NGPU=$("$VOXMATRIX_PYTHON" - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)
REPLICAS="${JUDGE_REPLICAS:-${NGPU}}"
if [[ "${REPLICAS}" -lt 1 ]]; then REPLICAS=1; fi
if [[ "${NGPU}" -gt 0 && "${REPLICAS}" -gt "${NGPU}" ]]; then
  echo "[ar-judge-job] clamp replicas ${REPLICAS} -> ${NGPU}"
  REPLICAS="${NGPU}"
fi
echo "[ar-judge-job] phase=llm-judge cache_only=${JUDGE_ASR_CACHE_ONLY} replicas=${REPLICAS} ngpu=${NGPU}"

if [[ "${REPLICAS}" -eq 1 ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  exec "$VOXMATRIX_PYTHON" -u scripts/run_audio_reasoning_local_judge.py \
    --shard-index 0 --num-shards 1 --out "${FINAL_OUT}"
fi

PIDS=()
SHARDS=()
STAGGER_SEC="${JUDGE_SHARD_STAGGER_SEC:-20}"
for ((i=0; i<REPLICAS; i++)); do
  SHARD_OUT="${SHARD_DIR}/ar-local-llm-judge.shard${i}.jsonl"
  SHARDS+=("${SHARD_OUT}")
  LOG_I="${SHARD_DIR}/shard${i}.log"
  if [[ "${i}" -gt 0 && "${STAGGER_SEC}" -gt 0 ]]; then
    echo "[ar-judge-job] stagger ${STAGGER_SEC}s before shard ${i}"
    sleep "${STAGGER_SEC}"
  fi
  echo "[ar-judge-job] launch shard ${i}/${REPLICAS} gpu=${i} -> ${SHARD_OUT}"
  (
    export CUDA_VISIBLE_DEVICES="${i}"
    export JUDGE_ASR_CACHE_ONLY=1
    "$VOXMATRIX_PYTHON" -u scripts/run_audio_reasoning_local_judge.py \
      --shard-index "${i}" --num-shards "${REPLICAS}" --out "${SHARD_OUT}"
  ) >"${LOG_I}" 2>&1 &
  PIDS+=("$!")
done

FAIL=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  if ! wait "${pid}"; then
    echo "[ar-judge-job] shard ${idx} failed pid=${pid}; tail:" >&2
    tail -40 "${SHARD_DIR}/shard${idx}.log" >&2 || true
    FAIL=1
  else
    echo "[ar-judge-job] shard ${idx} ok"
  fi
done
if [[ "${FAIL}" -ne 0 ]]; then
  echo "[ar-judge-job] one or more shards failed" >&2
  exit 1
fi

echo "[ar-judge-job] merge shards -> ${FINAL_OUT}"
printf '%s\n' "${SHARDS[@]}" > "${SHARD_DIR}/shard_list.txt"
"$VOXMATRIX_PYTHON" -u - <<PY
import json
from pathlib import Path
shard_list = Path("${SHARD_DIR}/shard_list.txt")
shards = [ln.strip() for ln in shard_list.read_text(encoding="utf-8").splitlines() if ln.strip()]
rows = []
for path in shards:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
out = Path("${FINAL_OUT}")
out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
print(json.dumps({"records": len(rows), "output": str(out), "shards": len(shards)}, ensure_ascii=False, indent=2), flush=True)
PY
echo "[ar-judge-job] done"
