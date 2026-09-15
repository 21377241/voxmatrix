#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
WORK_ROOT="${CAP_T4_WORK_ROOT:?CAP_T4_WORK_ROOT is required}"
PYTHON_BIN="${CAP_T4_PYTHON:-/mnt/afs/conda/envs/audioeval/bin/python}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export EVALUATION_UTILS_ROOT="${EVALUATION_UTILS_ROOT:-/mnt/afs/users/shizs/eval_framework/evaluation_utils}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "${PYTHON_BIN}" -u "${PROJECT_ROOT}/tools/capability_t4_smoke.py" infer \
  --work-root "${WORK_ROOT}" "$@"
