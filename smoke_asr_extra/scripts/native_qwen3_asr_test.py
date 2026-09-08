#!/usr/bin/env python3
"""Standalone native Qwen3-Omni ASR inference (no VoxMatrix wrapper)."""

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

MODEL_PATH = "/mnt/afs/models/Qwen3-Omni-30B-A3B-Instruct"

PROMPTS = {
    "default": "Transcribe the English audio into text.",
    "full_single": (
        "Transcribe the entire English audio from start to finish. "
        "Do not stop early."
    ),
    "full_multi": (
        "Transcribe the entire English audio from start to finish. Include all "
        "speakers and do not stop early."
    ),
}


def word_count(text: str) -> int:
    return len(text.split())


def run_one(model, processor, audio_path: str, prompt: str, max_new_tokens: int):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    use_audio_in_video = True

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=use_audio_in_video
    )
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    generation_kwargs = {"return_audio": False, "thinker_return_dict_in_generate": True}
    if max_new_tokens > 0:
        generation_kwargs["thinker_max_new_tokens"] = max_new_tokens

    t0 = time.time()
    text_ids, _ = model.generate(
        **inputs,
        use_audio_in_video=use_audio_in_video,
        **generation_kwargs,
    )
    latency_s = time.time() - t0

    pred = processor.batch_decode(
        text_ids.sequences[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    gen_tokens = int(text_ids.sequences.shape[1] - inputs["input_ids"].shape[1])
    return {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "pred": pred,
        "pred_words": word_count(pred),
        "generated_tokens": gen_tokens,
        "latency_s": round(latency_s, 2),
    }


def load_ref(args):
    if args.meta:
        meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
        return meta.get("ref", ""), meta
    if args.ref:
        return args.ref, None
    return "", None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--meta")
    parser.add_argument("--ref", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--prompt-mode",
        choices=["default", "single", "multi", "all"],
        default="all",
    )
    args = parser.parse_args()

    ref, meta = load_ref(args)
    info = sf.info(args.audio)
    print(f"audio: {args.audio}")
    print(f"duration: {info.duration:.2f}s  sr={info.samplerate}  channels={info.channels}")
    if ref:
        print(f"ref_words: {word_count(ref)}")
    print(f"cuda: {torch.cuda.is_available()}  device_count={torch.cuda.device_count()}")

    print("loading model ...")
    t0 = time.time()
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)
    print(f"model loaded in {time.time() - t0:.1f}s")

    cases = []
    if args.prompt_mode in ("default", "all"):
        cases.append(("default_prompt", PROMPTS["default"], 0))
    if args.prompt_mode in ("single", "all"):
        cases.append(("full_prompt_single", PROMPTS["full_single"], 0))
        cases.append(("full_prompt_single_4096", PROMPTS["full_single"], 4096))
    if args.prompt_mode in ("multi", "all"):
        cases.append(("full_prompt_multi", PROMPTS["full_multi"], 0))

    results = {
        "label": args.label or Path(args.audio).stem,
        "audio": args.audio,
        "model_path": args.model_path,
        "audio_duration_s": info.duration,
        "ref": ref,
        "ref_words": word_count(ref) if ref else None,
        "meta": meta,
        "runs": [],
    }

    for name, prompt, max_new_tokens in cases:
        print(f"\n=== run: {name} (max_new_tokens={max_new_tokens or 'default'}) ===")
        out = run_one(model, processor, args.audio, prompt, max_new_tokens)
        out["case"] = name
        if ref:
            out["coverage_pct"] = round(out["pred_words"] / word_count(ref) * 100, 1)
        results["runs"].append(out)
        cov = f" coverage={out.get('coverage_pct', 'n/a')}%" if ref else ""
        print(
            f"pred_words={out['pred_words']}  gen_tokens={out['generated_tokens']}  "
            f"latency={out['latency_s']}s{cov}"
        )
        print(f"pred: {out['pred'][:300]}{'...' if len(out['pred']) > 300 else ''}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
