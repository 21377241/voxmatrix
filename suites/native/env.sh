# Source before preflight / GPU run. Does not load the 30B model.
VOXMATRIX_ROOT="${VOXMATRIX_ROOT:-/mnt/afs/users/wangyl/VoxMatrix-clean}"
export VOXMATRIX_ROOT
export PYTHONPATH="${VOXMATRIX_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export VOXMATRIX_FULL_EVAL_ROOT="${VOXMATRIX_FULL_EVAL_ROOT:-/mnt/afs/users/wangyl/vocal_bench/work/full_eval}"
export MESH_FULL_EVAL_ROOT="${MESH_FULL_EVAL_ROOT:-$VOXMATRIX_FULL_EVAL_ROOT}"
export VOXMATRIX_SUBSET_MANIFEST="${VOXMATRIX_SUBSET_MANIFEST:-${VOXMATRIX_ROOT}/annotation/benchmark_subset_manifest.jsonl}"
export VOXMATRIX_PYTHON="${VOXMATRIX_PYTHON:-/mnt/afs/conda/envs/audioeval/bin/python}"
# isolate.py shells out to `uv`; keep the audioeval toolchain on PATH for GPU jobs.
_AUDIOEVAL_BIN="$(dirname "$VOXMATRIX_PYTHON")"
case ":${PATH}:" in
  *":${_AUDIOEVAL_BIN}:"*) ;;
  *) export PATH="${_AUDIOEVAL_BIN}:${PATH}" ;;
esac
unset _AUDIOEVAL_BIN
# One directory per run so concurrent/duplicate jobs never append to the same
# result files. Pass NATIVE_OUTPUT_DIR explicitly to resume or post-process a run.
export NATIVE_RUNS_ROOT="${NATIVE_RUNS_ROOT:-${VOXMATRIX_ROOT}/outputs/native_runs}"
export NATIVE_OUTPUT_DIR="${NATIVE_OUTPUT_DIR:-${NATIVE_RUNS_ROOT}/$(date +%Y%m%d_%H%M%S)_${HOSTNAME:-node}_$$}"
export NATIVE_LIMIT="${NATIVE_LIMIT:-${W1_LIMIT:-10}}"
export W1_LIMIT="${NATIVE_LIMIT}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export EVALUATION_UTILS_ROOT="${EVALUATION_UTILS_ROOT:-/mnt/afs/users/shizs/eval_framework/evaluation_utils}"
export VOXMATRIX_AUDIO_AGENT_ROOT="${VOXMATRIX_AUDIO_AGENT_ROOT:-/mnt/afs/eval_data/benchmarks/AudioAgentBench}"
export VOXMATRIX_IHBENCH_ROOT="${VOXMATRIX_IHBENCH_ROOT:-/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/IHBench}"
export VOXMATRIX_EAR_ROOT="${VOXMATRIX_EAR_ROOT:-/mnt/afs/eval_data/benchmarks/EAR}"
export VOXMATRIX_FSC_ROOT="${VOXMATRIX_FSC_ROOT:-/mnt/afs/eval_data/04_dialogue_slu/fluent_speech_commands_dataset}"
export VOXMATRIX_SLURP_ROOT="${VOXMATRIX_SLURP_ROOT:-/mnt/afs/oss_data/datasets/04_dialogue_slu/SLURP_repo}"
export VOXMATRIX_MMSU_ROOT="${VOXMATRIX_MMSU_ROOT:-/mnt/afs/eval_data/benchmarks/MMSU}"
