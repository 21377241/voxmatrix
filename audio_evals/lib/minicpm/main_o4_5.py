"""JSON-line worker for MiniCPM-o 4.5 (half-duplex chat, audio understanding / speech output)."""

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

# Must run before importing transformers / torch so concurrent replicas do not
# race on writing trust_remote_code modules into one shared HF modules cache.
os.environ["HF_MODULES_CACHE"] = os.path.join(
    tempfile.gettempdir(), f"voxmatrix-minicpm-hf-modules-{os.getpid()}"
)

import librosa
import torch
from transformers import AutoModel

READY_SENTINEL = "__MINICPMO45_READY__"
DEFAULT_REF_AUDIO = str(
    Path(__file__).resolve().parents[3] / "mesh_eval" / "data" / "default.wav"
)
VOICE_PREFIX = "模仿音频样本的音色并生成新的内容。"
VOICE_SUFFIX = "请用这种声音风格来为用户提供帮助。 直接作答，不要有冗余内容"


def load(args):
    model = AutoModel.from_pretrained(
        args.path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False,
        init_audio=True,
        init_tts=args.speech,
    )
    model = model.eval().cuda()
    ref_audio = None
    if args.speech:
        model.init_tts()
        ref_audio, _ = librosa.load(args.ref_audio, sr=16000, mono=True)
    return model, ref_audio


def build_msgs(prompt, ref_audio):
    msgs = []
    system_texts = []
    for message in prompt:
        role = message.get("role") or "user"
        content = []
        for item in message.get("contents") or []:
            if item["type"] == "text":
                content.append(item["value"])
            elif item["type"] == "audio":
                content.append(librosa.load(item["value"], sr=16000, mono=True)[0])
            else:
                raise ValueError(f"unsupported content type: {item['type']!r}")
        if role == "system":
            system_texts.extend(c for c in content if isinstance(c, str))
        else:
            msgs.append({"role": role, "content": content})
    if ref_audio is not None:
        suffix = "\n".join([VOICE_SUFFIX] + system_texts)
        msgs.insert(0, {"role": "system", "content": [VOICE_PREFIX, ref_audio, suffix]})
    elif system_texts:
        msgs.insert(0, {"role": "system", "content": ["\n".join(system_texts)]})
    return msgs


def run_request(model, ref_audio, payload, args):
    kwargs = {"do_sample": False, "max_new_tokens": args.max_new_tokens}
    kwargs.update(payload.get("generate_kwargs") or {})
    msgs = build_msgs(payload["prompt"], ref_audio)
    if args.speech:
        out_dir = args.speech_output_dir or None
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=out_dir, delete=False) as f:
            audio_path = f.name
        text = model.chat(
            msgs=msgs,
            use_tts_template=True,
            enable_thinking=False,
            generate_audio=True,
            output_audio_path=audio_path,
            **kwargs,
        )
        return {"text": str(text or "").strip(), "audio": audio_path}
    text = model.chat(
        msgs=msgs, use_tts_template=True, enable_thinking=False, **kwargs
    )
    return {"text": str(text or "").strip()}


def emit(prefix, response):
    print(prefix + json.dumps(response, ensure_ascii=False), flush=True)
    ack = sys.stdin.readline().strip()
    if ack != f"{prefix}close":
        raise RuntimeError("response acknowledgement was not received")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--speech", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    parser.add_argument("--speech-output-dir", default="")
    args = parser.parse_args()
    model, ref_audio = load(args)
    print(f"{READY_SENTINEL} Model loaded from checkpoint: {args.path}", flush=True)
    for raw in sys.stdin:
        prefix = ""
        try:
            anchor = raw.find("->")
            if anchor < 0:
                raise ValueError("request is missing protocol separator")
            prefix = raw[:anchor].strip() + "->"
            payload = json.loads(raw[anchor + 2 :])
            with torch.no_grad():
                response = run_request(model, ref_audio, payload, args)
            emit(prefix, response)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            if prefix:
                emit(prefix, {"error": f"{type(exc).__name__}: {exc}"})
            else:
                print(f"Error:{type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
