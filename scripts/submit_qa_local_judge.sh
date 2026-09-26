#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VOXMATRIX_ROOT="$ROOT"
JOB="${JOB_BIN:-/mnt/afs/users/wangyl/job}"
cd "$ROOT"
# Default 2 GPUs for A1B-style sample sharding (override with -g N).
exec "$JOB" submit "$ROOT/scripts/run_qa_local_judge_job.sh" \
  -g "${JUDGE_GPUS:-2}" \
  -N vox-qa-local-judge \
  --project-root "$ROOT" \
  --output-root "$ROOT/outputs/native_qa_smoke" \
  "$@"
