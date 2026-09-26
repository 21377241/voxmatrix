#!/usr/bin/env bash
# GPU-node: Native QA smoke (vLLM-Omni thinker-only listen→text) then Thinking judge.
# Verifies qwen3-omni-local / qwen3-omni-thinking on envs/vllm_omni.
set -euo pipefail
ROOT="${VOXMATRIX_ROOT:-/mnt/afs/users/wangyl/VoxMatrix-clean}"
HERE="$ROOT/suites/native"
# shellcheck disable=SC1091
source "$HERE/env.sh"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NATIVE_CAPABILITY_FILTER="${NATIVE_CAPABILITY_FILTER:-qa}"
unset NATIVE_SLOT_FILTER || true
PY="${VOXMATRIX_PYTHON}"

echo "[native-qa-vllm] python=$PY cuda=$CUDA_VISIBLE_DEVICES filter=$NATIVE_CAPABILITY_FILTER"
"$PY" -c 'import torch; assert torch.cuda.is_available(), "no GPU"; print("cuda", torch.cuda.device_count())'
"$PY" "$HERE/preflight.py"
SUITE_MAIN="$HERE/output/suite.runtime.yaml"
[[ -f "$SUITE_MAIN" ]] || {
  echo "[native-qa-vllm] missing $SUITE_MAIN" >&2
  exit 1
}

echo "[native-qa-vllm] phase=sut-infer (vLLM Instruct)"
"$PY" -m audio_evals.session --config "$SUITE_MAIN"

echo "[native-qa-vllm] phase=cuda-cleanup after SUT"
"$PY" -u - <<'PY'
import gc
try:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"[native-qa-vllm] cuda empty_cache ok devices={torch.cuda.device_count()}", flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"[native-qa-vllm] cuda cleanup skip: {exc}", flush=True)
PY

echo "[native-qa-vllm] phase=build-judge-pack"
"$PY" -u scripts/build_qa_judge_pack.py

echo "[native-qa-vllm] phase=local-judge (vLLM Thinking + short-output prompts)"
# Reuse the existing two-phase judge job body (ASR cache → judge shards).
bash "$ROOT/scripts/run_qa_local_judge_job.sh"

echo "[native-qa-vllm] done"
ls -la "$HERE/output/main"/native-qa--* "$ROOT/outputs/native_qa_smoke/qa-local-llm-judge.jsonl" 2>/dev/null || true
