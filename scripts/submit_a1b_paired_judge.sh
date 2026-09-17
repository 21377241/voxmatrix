#!/usr/bin/env bash
# Submit A1/B formal paired judge (+ undertest fill) to the cluster.
set -euo pipefail

VOXMATRIX_ROOT="${VOXMATRIX_ROOT:-/mnt/afs/users/wangyl/VoxMatrix}"
JOB_BIN="${JOB_BIN:-/mnt/afs/users/wangyl/job}"
GPUS="${SCO_GPU_NUMS:-${GPUS:-2}}"
NAME="${JOB_NAME:-a1b-thinking-$(date +%Y%m%d_%H%M%S)}"
RUN_SCRIPT="${VOXMATRIX_ROOT}/scripts/run_a1b_paired_judge_gpu.sh"
OUT_ROOT="${SCO_OUTPUT_ROOT:-${VOXMATRIX_ROOT}/output}"

if [[ ! -x "${JOB_BIN}" ]]; then
  echo "job CLI not executable: ${JOB_BIN}" >&2
  exit 127
fi
if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "missing run script: ${RUN_SCRIPT}" >&2
  exit 2
fi

cd "${VOXMATRIX_ROOT}"
exec "${JOB_BIN}" submit "${RUN_SCRIPT}" \
  -g "${GPUS}" \
  -N "${NAME}" \
  --project-root "${VOXMATRIX_ROOT}" \
  --output-root "${OUT_ROOT}" \
  -e "VOXMATRIX_ROOT=${VOXMATRIX_ROOT}" \
  -e "MESH_JUDGE_MODEL=${MESH_JUDGE_MODEL:-qwen3-omni-thinking}" \
  -e "MESH_MM_JUDGE_MODEL=${MESH_JUDGE_MODEL:-qwen3-omni-thinking}" \
  -e "JUDGE_ASR_BACKEND=${JUDGE_ASR_BACKEND:-local}" \
  -e "OUT_DIR=${OUT_ROOT}/a1b_paired_judge" \
  "$@"
