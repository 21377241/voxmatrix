#!/usr/bin/env bash
# Pre-build isolated envs on a CPU node so GPU jobs do not install packages.
#   bash scripts/build_model_envs.sh kimiaudio|minicpmo4_5
# uv's directory renames hang on the /mnt/afs FUSE mount, so each env is built
# relocatable on local disk and then copied to envs/<name>.
# kimiaudio: torch 2.6 + the cluster-local flash-attn wheel (GitHub releases are
#   unreachable and the Kimi whisper encoder imports flash_attn); the registry
#   uses it with requirements_path "".
# minicpmo4_5: gets the audio_evals.isolate prepare sentinel for its registry
#   requirements file, so replicas reuse it instead of reinstalling.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/mnt/afs/conda/envs/audioeval/bin:$PATH"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
# PyPI torch 2.6.0 wheels are cu124 builds; download.pytorch.org is ~60 KB/s here.
unset UV_TORCH_BACKEND
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
PY=/mnt/afs/conda/envs/audioeval/bin/python
LOCAL_BUILD_ROOT="${LOCAL_BUILD_ROOT:-/tmp/voxmatrix-envs}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-/mnt/afs/users/wangyl/vocal_bench/vendor/wheels/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"

NAME="${1:-}"
case "$NAME" in
  kimiaudio) REQ=audio_evals/lib/Kimi-Audio/requirements.txt ;;
  minicpmo4_5) REQ=audio_evals/lib/minicpm/requirements-o4_5.txt ;;
  *) echo "usage: $0 kimiaudio|minicpmo4_5" >&2; exit 2 ;;
esac
BUILD="$LOCAL_BUILD_ROOT/$NAME"
TARGET="$ROOT/envs/$NAME"
[[ -e "$TARGET" ]] && { echo "$TARGET already exists; remove it first" >&2; exit 1; }

rm -rf "$BUILD"
uv venv "$BUILD" --python "$PY" --relocatable
install=(uv pip install --python "$BUILD/bin/python" --index-strategy unsafe-best-match)
"${install[@]}" 'setuptools<81'
if [[ "$NAME" == kimiaudio ]]; then
  # torchvision must match torch 2.6 (0.21.x); unpinned deps pull 0.29+ and break nms.
  printf 'torch==2.6.0\ntorchaudio==2.6.0\ntorchvision==0.21.0\n' > "$BUILD/overrides.txt"
  "${install[@]}" --override "$BUILD/overrides.txt" -r "$REQ" "$FLASH_ATTN_WHEEL"
else
  "${install[@]}" -r "$REQ"
fi

mkdir -p "$(dirname "$TARGET")"
cp -r --preserve=mode,timestamps,links "$BUILD" "$TARGET"
if [[ "$NAME" == minicpmo4_5 ]]; then
  "$PY" - "$TARGET" "$REQ" <<'PY'
import os, sys
from audio_evals.isolate import _ENV_PREPARE_VERSION, _atomic_json, _sha256_file
env, req = sys.argv[1], os.path.abspath(sys.argv[2])
_atomic_json(os.path.join(env, ".ultraeval-prepare.json"), {
    "format_version": _ENV_PREPARE_VERSION, "python": "3.10",
    "requirements_path": req, "requirements_sha256": _sha256_file(req), "pre_command": "",
})
PY
fi
echo "built $TARGET"
