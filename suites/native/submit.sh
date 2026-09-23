#!/usr/bin/env bash
# Submit the native all-capability job. Does not start until you run this.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
export VOXMATRIX_ROOT="$ROOT"
JOB="${JOB_BIN:-/mnt/afs/users/wangyl/job}"
cd "$ROOT"
exec "$JOB" submit "$HERE/run.sh" \
  -g 1 \
  -N vox-native \
  --project-root "$ROOT" \
  --output-root "$ROOT/suites/native/output" \
  "$@"
