#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VOXMATRIX_ROOT="$ROOT"
JOB="${JOB_BIN:-/mnt/afs/users/wangyl/job}"
cd "$ROOT"
# 2 GPUs: SUT uses one; judge shards across both after unload.
exec "$JOB" submit "$ROOT/scripts/run_native_qa_vllm_retest.sh" \
  -g "${QA_RETEST_GPUS:-2}" \
  -N vox-native-qa-vllm-retest \
  --project-root "$ROOT" \
  --output-root "$ROOT/outputs/native_qa_smoke" \
  "$@"
