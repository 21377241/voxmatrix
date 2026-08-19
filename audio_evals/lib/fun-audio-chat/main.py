import argparse
import json
import sys

import librosa
import torch
from funaudiochat.register import register_funaudiochat
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor


READY_SENTINEL = "__FUN_AUDIO_CHAT_READY__"


def load_model(path):
    register_funaudiochat()
    config = AutoConfig.from_pretrained(path)
    processor = AutoProcessor.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.sp_gen_kwargs.update({"text_greedy": True, "disable_speech": True})
    model.eval()
    return model, processor


def run_request(model, processor, payload, max_new_tokens):
    conversation = payload.get("conversation")
    audio_paths = payload.get("audio_paths") or []
    if not isinstance(conversation, list):
        raise TypeError("conversation must be a list")
    audios = [librosa.load(path, sr=16000, mono=True)[0] for path in audio_paths]
    rendered = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=rendered,
        audio=audios,
        return_tensors="pt",
        return_token_type_ids=False,
    ).to(model.device)
    with torch.inference_mode():
        generated_ids, _ = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )
    generated_ids = generated_ids[:, inputs.input_ids.size(1) :]
    return processor.decode(generated_ids[0], skip_special_tokens=True)


def emit(prefix, response):
    print(prefix + json.dumps(response, ensure_ascii=False), flush=True)
    acknowledgement = sys.stdin.readline().strip()
    if acknowledgement != f"{prefix}close":
        raise RuntimeError("response acknowledgement was not received")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    model, processor = load_model(args.path)
    print(f"{READY_SENTINEL} Model loaded from checkpoint: {args.path}", flush=True)

    for raw_request in sys.stdin:
        prefix = ""
        try:
            anchor = raw_request.find("->")
            if anchor < 0:
                raise ValueError("request is missing protocol separator")
            prefix = raw_request[:anchor].strip() + "->"
            payload = json.loads(raw_request[anchor + 2 :])
            text = run_request(model, processor, payload, args.max_new_tokens)
            emit(prefix, {"text": text})
        except Exception as exc:
            if prefix:
                emit(prefix, {"error": f"{type(exc).__name__}: {exc}"})
            else:
                print(f"Error:{type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
