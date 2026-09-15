#!/usr/bin/env bash
# GPU node entry: ASR (local) + A1/B paired judge with qwen3-omni-thinking.
set -euo pipefail
VOXMATRIX_ROOT="${VOXMATRIX_ROOT:-/mnt/afs/users/wangyl/VoxMatrix}"
cd "${VOXMATRIX_ROOT}"
export PYTHONPATH="${VOXMATRIX_ROOT}"
export MESH_JUDGE_MODEL="${MESH_JUDGE_MODEL:-qwen3-omni-thinking}"
export JUDGE_ASR_BACKEND="${JUDGE_ASR_BACKEND:-local}"
PACK="${PACK:-${VOXMATRIX_ROOT}/scripts/data/semantic_judge_stratified_pack.with_pred.jsonl}"
OUT="${OUT:-/tmp/semantic_judge_a1b_full.json}"
REPORT="${REPORT:-/mnt/afs/users/wangyl/8p31/交付/开放语义_A1B音频增益实验报告.md}"
ASR_CACHE="${ASR_CACHE:-/tmp/semantic_judge_asr_cache}"

python - <<'PY'
import torch
print("cuda", torch.cuda.is_available(), "n", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA required for Thinking judge")
PY

# Fill missing preds if residual exists and CUDA ok
NEED="${VOXMATRIX_ROOT}/scripts/data/semantic_judge_stratified_pack.need_pred.jsonl"
if [[ -f "${NEED}" ]] && [[ "$(wc -l < "${NEED}")" -gt 0 ]]; then
  python scripts/run_undertest_for_stratified_pack.py --pack "${NEED}" --skip-existing || true
  python scripts/join_preds_into_stratified_pack.py \
    --pack "${VOXMATRIX_ROOT}/scripts/data/semantic_judge_stratified_pack.jsonl" \
    --out "${PACK}"
fi

python scripts/compare_semantic_llm_judge_tracks.py \
  --pack "${PACK}" \
  --tracks A1,B \
  --judge-model "${MESH_JUDGE_MODEL}" \
  --asr-cache "${ASR_CACHE}" \
  --out "${OUT}" \
  --report-md "${REPORT}"

echo "done out=${OUT} report=${REPORT}"
