#!/usr/bin/env bash
# Submit infer, wait until success, then submit judge. Requires job CLI.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VOXMATRIX_ROOT="$ROOT"
JOB="${JOB_BIN:-/mnt/afs/users/wangyl/job}"
cd "$ROOT"
chmod +x \
  "$ROOT/scripts/submit_native_speech_vllm_omni_infer.sh" \
  "$ROOT/scripts/submit_native_speech_vllm_omni_judge.sh"

export NATIVE_OUTPUT_DIR="${NATIVE_OUTPUT_DIR:-$ROOT/outputs/native_runs/$(date +%Y%m%d_%H%M%S)_speech_split_$$}"
echo "[split] run dir: $NATIVE_OUTPUT_DIR"
echo "[split] submitting inference job..."
INF_OUT="$("$ROOT/scripts/submit_native_speech_vllm_omni_infer.sh" "$@" 2>&1 | tee /dev/stderr)"
# Parse numeric id from: "[OK] 已入队  id=24071  display_name=..."
INF_ID="$(printf '%s\n' "$INF_OUT" | rg -o 'id=[0-9]+' | head -1 | cut -d= -f2 || true)"
if [[ -z "${INF_ID}" ]]; then
  # Fallback: newest job dir under infer output-root (prefix is numeric id).
  INF_ROOT="$ROOT/outputs/native_speech_vllm_omni_infer/jobs"
  INF_DIR="$(ls -1t "$INF_ROOT" 2>/dev/null | head -1 || true)"
  INF_ID="$(printf '%s\n' "$INF_DIR" | rg -o '^[0-9]+' || true)"
fi
if [[ -z "${INF_ID}" ]]; then
  echo "[split] could not determine infer job id; submit judge manually after infer finishes" >&2
  exit 1
fi
echo "[split] tracking infer job: $INF_ID"
"$JOB" track "$INF_ID" || {
  echo "[split] infer job failed or track ended non-zero" >&2
  exit 1
}

MARKER="$NATIVE_OUTPUT_DIR/tts/session.json"
if [[ ! -f "$MARKER" ]]; then
  echo "[split] missing $MARKER after infer" >&2
  exit 1
fi
"$ROOT/scripts/submit_native_speech_vllm_omni_judge.sh" "$@"
echo "[split] judge submitted"
