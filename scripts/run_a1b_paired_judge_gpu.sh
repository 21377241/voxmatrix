#!/usr/bin/env bash
# GPU node entry: fill missing Instruct preds → join → local ASR + A1/B Thinking judge.
# A1B_PHASE=all|asr|judge  (default all). asr stops after cache prefill; judge skips undertest/join.
# A1B_REPLICAS=N launches N sharded judge workers (one GPU each via CUDA_VISIBLE_DEVICES).
set -euo pipefail

VOXMATRIX_ROOT="${VOXMATRIX_ROOT:-/mnt/afs/users/wangyl/VoxMatrix}"
cd "${VOXMATRIX_ROOT}"
export PYTHONPATH="${VOXMATRIX_ROOT}"

PYTHON="${PYTHON:-${VOXMATRIX_ROOT}/envs/qwen3-omni/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3 || command -v python)"
fi

export MESH_JUDGE_MODEL="${MESH_JUDGE_MODEL:-qwen3-omni-thinking}"
export MESH_MM_JUDGE_MODEL="${MESH_JUDGE_MODEL}"
export JUDGE_ASR_BACKEND="${JUDGE_ASR_BACKEND:-local}"
export JUDGE_ASR_DEVICE="${JUDGE_ASR_DEVICE:-cuda}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATA_DIR="${DATA_DIR:-${VOXMATRIX_ROOT}/scripts/data}"
PACK_BASE="${PACK_BASE:-${DATA_DIR}/semantic_judge_stratified_pack.jsonl}"
PACK="${PACK:-${DATA_DIR}/semantic_judge_stratified_pack.with_pred.jsonl}"
NEED="${NEED:-${DATA_DIR}/semantic_judge_stratified_pack.need_pred.jsonl}"
UNDERTEST_OUT="${UNDERTEST_OUT:-${DATA_DIR}/semantic_judge_undertest_preds.jsonl}"

OUT_DIR="${OUT_DIR:-${VOXMATRIX_ROOT}/output/a1b_paired_judge}"
mkdir -p "${OUT_DIR}"
OUT="${OUT:-${OUT_DIR}/semantic_judge_a1b_full.json}"
REPORT_AUTO="${REPORT_AUTO:-/mnt/afs/users/wangyl/8p31/交付/开放语义_A1B成对Judge自动稿.md}"
# Formal case-study report is locked; job merge must NOT overwrite it.
REPORT="${REPORT:-${REPORT_AUTO}}"
ASR_CACHE="${ASR_CACHE:-${OUT_DIR}/asr_cache}"
mkdir -p "${ASR_CACHE}"
echo "[a1b] auto report -> ${REPORT} (formal case-study report is locked separately)"

PHASE="${A1B_PHASE:-all}"  # all | asr | judge

echo "[a1b] python=${PYTHON}"
echo "[a1b] judge=${MESH_JUDGE_MODEL} asr_backend=${JUDGE_ASR_BACKEND} asr_device=${JUDGE_ASR_DEVICE}"
echo "[a1b] phase=${PHASE}"
NGPU=$("${PYTHON}" - <<'PY'
import torch
n = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(n)
if n < 1:
    raise SystemExit("CUDA required for undertest Omni / Thinking judge")
PY
)
echo "[a1b] cuda_n=${NGPU}"

# 1-2) undertest + join (skip for judge-only)
if [[ "${PHASE}" != "judge" ]]; then
  if [[ -f "${NEED}" ]] && [[ "$(wc -l < "${NEED}" | tr -d ' ')" -gt 0 ]]; then
    echo "[a1b] undertest residual lines=$(wc -l < "${NEED}" | tr -d ' ')"
    "${PYTHON}" scripts/run_undertest_for_stratified_pack.py \
      --pack "${NEED}" \
      --out "${UNDERTEST_OUT}" \
      --model qwen3-omni-audio \
      --skip-existing
  else
    echo "[a1b] no residual need_pred (or empty); skip undertest"
  fi

  echo "[a1b] join preds (incl. undertest)"
  "${PYTHON}" scripts/join_preds_into_stratified_pack.py \
    --pack "${PACK_BASE}" \
    --out "${PACK}" \
    --residual-out "${NEED}" \
    --undertest "${UNDERTEST_OUT}"
fi

# 3) Prefill ASR
if [[ "${PHASE}" == "asr" || "${PHASE}" == "all" ]]; then
  echo "[a1b] prefill ASR cache -> ${ASR_CACHE}"
  "${PYTHON}" scripts/prefill_asr_for_judge_pack.py \
    --pack "${PACK}" \
    --asr-cache "${ASR_CACHE}"
  if [[ "${PHASE}" == "asr" ]]; then
    echo "[a1b] phase=asr done cache=${ASR_CACHE}"
    exit 0
  fi
fi

"${PYTHON}" - <<'PY'
import gc
try:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("cuda empty_cache ok", flush=True)
except Exception as exc:  # noqa: BLE001
    print("cuda cleanup skip:", exc, flush=True)
PY

REPLICAS="${A1B_REPLICAS:-${NGPU}}"
if [[ "${REPLICAS}" -lt 1 ]]; then
  REPLICAS=1
fi
if [[ "${REPLICAS}" -gt "${NGPU}" ]]; then
  echo "[a1b] clamp replicas ${REPLICAS} -> ${NGPU}"
  REPLICAS="${NGPU}"
fi

echo "[a1b] compare A1/B with ${MESH_JUDGE_MODEL} replicas=${REPLICAS}"

if [[ "${REPLICAS}" -eq 1 ]]; then
  COMPARE_ARGS=(
    --pack "${PACK}"
    --tracks A1,B
    --judge-model "${MESH_JUDGE_MODEL}"
    --transcript-mode asr
    --asr-cache "${ASR_CACHE}"
    --out "${OUT}"
    --report-md "${REPORT}"
  )
  if [[ "${A1B_RESUME:-1}" == "1" ]]; then
    COMPARE_ARGS+=(--resume)
  fi
  "${PYTHON}" scripts/compare_semantic_llm_judge_tracks.py "${COMPARE_ARGS[@]}"
else
  SHARD_DIR="${OUT_DIR}/shards"
  mkdir -p "${SHARD_DIR}"
  PIDS=()
  SHARDS=()
  for ((i=0; i<REPLICAS; i++)); do
    SHARD_OUT="${SHARD_DIR}/semantic_judge_a1b.shard${i}.json"
    SHARDS+=("${SHARD_OUT}")
    LOG_I="${SHARD_DIR}/shard${i}.log"
    # Stagger cold starts to avoid Errno 11 / transient path races on shared FS.
    STAGGER_SEC="${A1B_SHARD_STAGGER_SEC:-20}"
    if [[ "${i}" -gt 0 && "${STAGGER_SEC}" -gt 0 ]]; then
      echo "[a1b] stagger ${STAGGER_SEC}s before shard ${i}"
      sleep "${STAGGER_SEC}"
    fi
    echo "[a1b] launch shard ${i}/${REPLICAS} gpu=${i} -> ${SHARD_OUT}"
    (
      export CUDA_VISIBLE_DEVICES="${i}"
      ARGS=(
        --pack "${PACK}"
        --tracks A1,B
        --judge-model "${MESH_JUDGE_MODEL}"
        --transcript-mode asr
        --asr-cache "${ASR_CACHE}"
        --out "${SHARD_OUT}"
        --report-md "${REPORT}"
        --shard-index "${i}"
        --num-shards "${REPLICAS}"
        --skip-report
      )
      if [[ "${A1B_RESUME:-1}" == "1" ]]; then
        ARGS+=(--resume)
      fi
      "${PYTHON}" scripts/compare_semantic_llm_judge_tracks.py "${ARGS[@]}"
    ) >"${LOG_I}" 2>&1 &
    PIDS+=("$!")
  done

  FAIL=0
  for idx in "${!PIDS[@]}"; do
    pid="${PIDS[$idx]}"
    if ! wait "${pid}"; then
      echo "[a1b] shard ${idx} failed pid=${pid}; tail:" >&2
      tail -40 "${SHARD_DIR}/shard${idx}.log" >&2 || true
      FAIL=1
    else
      echo "[a1b] shard ${idx} ok"
    fi
  done
  if [[ "${FAIL}" -ne 0 ]]; then
    echo "[a1b] one or more shards failed" >&2
    exit 1
  fi

  echo "[a1b] merge shards -> ${OUT}"
  "${PYTHON}" scripts/merge_a1b_judge_shards.py \
    --out "${OUT}" \
    --report-md "${REPORT}" \
    --pack "${PACK}" \
    --shards "${SHARDS[@]}"
fi

echo "[a1b] done out=${OUT} report=${REPORT}"
