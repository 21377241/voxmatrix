#!/usr/bin/env bash
# bash scripts/submit_model_smoke.sh step-audio-2-mini,step-audio-2-mini-speech [extra job args]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB="${JOB_BIN:-/mnt/afs/users/wangyl/job}"
MODELS="${1:?usage: $0 model1,model2 [job args]}"
shift
cd "$ROOT"
exec "$JOB" submit "$ROOT/scripts/run_model_smoke.sh" \
  -g "${SMOKE_GPUS:-1}" \
  -N "vox-smoke-${MODELS%%,*}" \
  --project-root "$ROOT" \
  --output-root "$ROOT/outputs/model_smoke" \
  -e "SMOKE_MODELS=$MODELS" \
  "$@"
