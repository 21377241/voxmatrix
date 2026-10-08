#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VOXMATRIX_ROOT="$ROOT"
JOB="${JOB_BIN:-/mnt/afs/users/wangyl/job}"
: "${NATIVE_OUTPUT_DIR:?set NATIVE_OUTPUT_DIR to the native run dir (outputs/native_runs/<run>)}"
[[ -d "$NATIVE_OUTPUT_DIR/main" ]] || { echo "no native results under $NATIVE_OUTPUT_DIR/main" >&2; exit 2; }
cd "$ROOT"
# Default 2 GPUs for A1B-style sample sharding (override with -g N).
exec "$JOB" submit "$ROOT/scripts/run_qa_local_judge_job.sh" \
  -g "${JUDGE_GPUS:-2}" \
  -N vox-qa-local-judge \
  --project-root "$ROOT" \
  --output-root "$ROOT/outputs/native_qa_smoke" \
  -e "NATIVE_OUTPUT_DIR=$NATIVE_OUTPUT_DIR" \
  "$@"
