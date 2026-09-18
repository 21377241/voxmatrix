#!/usr/bin/env bash
# Submit the native all-capability job. Does not start until you run this.
set -euo pipefail
ROOT="${VOXMATRIX_ROOT:-/mnt/afs/users/wangyl/VoxMatrix-clean}"
JOB="${JOB_BIN:-/mnt/afs/users/wangyl/job}"
cd "$ROOT"
exec "$JOB" submit "$ROOT/suites/native/run.sh" \
  -g 1 \
  -N vox-native \
  --project-root "$ROOT" \
  --output-root "$ROOT/suites/native/output" \
  "$@"
