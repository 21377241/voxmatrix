#!/usr/bin/env bash
# Phase 2: judge/ASR only. Replays saved inference; does not load speech Omni.
set -euo pipefail
ROOT="${VOXMATRIX_ROOT:-/mnt/afs/users/wangyl/VoxMatrix-clean}"
HERE="$ROOT/suites/native"
# shellcheck disable=SC1091
source "$HERE/env.sh"
cd "$ROOT"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi
export NATIVE_SPEECH_MODEL="${NATIVE_SPEECH_MODEL:-qwen3-omni-speech-vllm-omni-local}"
export NATIVE_LIMIT="${NATIVE_LIMIT:-2}"
export NATIVE_SLOT_FILTER="${NATIVE_SLOT_FILTER:-speech_generation__libritts,instruction_following__vocalbench}"
unset NATIVE_CAPABILITY_FILTER || true
export VOXMATRIX_SESSION_PHASE=evaluation
PY="${VOXMATRIX_PYTHON}"

TTS_OUT="$NATIVE_OUTPUT_DIR/tts"
VOCAL="$TTS_OUT/native-instruction-following--vocalbench.jsonl"
LIBRI="$TTS_OUT/native-speech-generation--libritts.jsonl"
for f in "$VOCAL" "$LIBRI"; do
  if [[ ! -f "$f" ]]; then
    echo "missing inference jsonl: $f (run infer job first)" >&2
    exit 2
  fi
done

# Fail fast if replay audio paths point at vanished /tmp files.
"$PY" - "$VOCAL" "$LIBRI" <<'PY'
import json, os, sys
paths = sys.argv[1:]
missing = []
for p in paths:
    with open(p) as f:
        for line in f:
            o = json.loads(line)
            if o.get("type") != "inference":
                continue
            content = o.get("data", {}).get("content")
            audio = None
            if isinstance(content, str):
                try:
                    payload = json.loads(content)
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    audio = payload.get("audio")
            if audio and not os.path.isfile(audio):
                missing.append(audio)
if missing:
    print("missing wav(s) for replay:", *missing[:5], sep="\n  ", file=sys.stderr)
    print("re-run infer job with speech_output_dir on AFS", file=sys.stderr)
    sys.exit(3)
print("replay audio ok")
PY

echo "[native-speech-judge] phase=evaluation limit=$NATIVE_LIMIT slots=$NATIVE_SLOT_FILTER"
"$PY" "$HERE/preflight.py"
SUITE_TTS="$NATIVE_OUTPUT_DIR/suite_tts.runtime.yaml"
[[ -f "$SUITE_TTS" ]] || { echo "missing $SUITE_TTS" >&2; exit 1; }
"$PY" -c 'import torch; assert torch.cuda.is_available(); print("cuda", torch.cuda.device_count())'
"$PY" -m audio_evals.session --config "$SUITE_TTS" --phase evaluation
echo "[native-speech-judge] done"
ls -la "$TTS_OUT" | head -40
