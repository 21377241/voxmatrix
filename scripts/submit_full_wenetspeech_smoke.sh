#!/usr/bin/env bash
# Submit the WenetSpeech Full Profile smoke regression to one GPU node.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VOXMATRIX_ROOT="$ROOT"
JOB="${JOB_BIN:-/mnt/afs/users/wangyl/job}"

cd "$ROOT"
exec "$JOB" submit "$ROOT/scripts/run_full_wenetspeech_smoke.sh" \
  -g 1 \
  -N vox-full-wenetspeech-smoke \
  --project-root "$ROOT" \
  --output-root "$ROOT/outputs/full_wenetspeech_smoke" \
  "$@"
