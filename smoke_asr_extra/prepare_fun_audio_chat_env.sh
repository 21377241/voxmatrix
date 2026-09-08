#!/usr/bin/env bash
# Fun-Audio-Chat env: reuse qwen3-omni torch/cuda stack, pin transformers 4.52.3 only.
#
# Usage:
#   bash smoke_asr_extra/prepare_fun_audio_chat_env.sh
#
# Mirrors:
#   PYPI_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_DIR="$ROOT/envs/fun-audio-chat"
QWEN_ENV="$ROOT/envs/qwen3-omni"
LIB="$ROOT/audio_evals/lib/fun-audio-chat"
PYPI_MIRROR="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

verify_env() {
  local py="$1"
  "$py" -c "
import sys
sys.path.insert(0, '$LIB')
from funaudiochat.register import register_funaudiochat
register_funaudiochat()
import torch, transformers, librosa, accelerate, soundfile
assert transformers.__version__.startswith('4.52'), transformers.__version__
print('env ok', torch.__version__, transformers.__version__)
"
}

if [[ ! -x "$QWEN_ENV/bin/python" ]]; then
  echo "missing $QWEN_ENV"
  exit 1
fi

echo "[prepare] venv at $ENV_DIR (system-site-packages=$QWEN_ENV, pypi=$PYPI_MIRROR)"
rm -rf "$ENV_DIR"
"$QWEN_ENV/bin/python" -m venv --system-site-packages "$ENV_DIR"
"$ENV_DIR/bin/pip" install -U pip setuptools wheel -i "$PYPI_MIRROR"
# Only pin transformers; torch/librosa/accelerate come from qwen3-omni site-packages.
"$ENV_DIR/bin/pip" install -i "$PYPI_MIRROR" 'transformers==4.52.3'
verify_env "$ENV_DIR/bin/python"
echo "[prepare] ready at $ENV_DIR"
