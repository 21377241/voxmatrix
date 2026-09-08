#!/usr/bin/env python3
"""Standalone native Qwen3-Omni inference on one EasyCom clip (no VoxMatrix wrapper)."""

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

MODEL_PATH = "/mnt/afs/models/Qwen3-Omni-30B-A3B-Instruct"
AUDIO_PATH = (
    "/mnt/afs/users/wangyl/VoxMatrix/smoke_asr_extra/easycom/audio/"
    "easycom_test_10_00_00_000.wav"
)
REF = (
    "I dunno! You guys told me you'd pay me three dollars an hour to sit here and "
    "talk to you about something. What do you want me to talk about? Oh! My name is "
    "Sophie. I- I drive big trucks. Like, I try not to hit things but, you know, "
    "every once in a while it happens. Huh? Yeah! Yeah. Drive all over the place. "
    "Yeah, I was in Tennessee- Yeah, yeah! I gotta leave in twenty minutes to head "
    "to Alaska, so figured I'd make a little bit of money for beer along the way. "
    "Yeah yeah, yeah. Yeah. Got it. Wow! Sounds cool, Sophie. Wow! Yeah! There's "
    "three dollar beer in Alaska? Wow. Ah. No wonder it's three dollars, then! I "
    "think no wonder it's three dollars! You get a big one for three dollars. "
    "gonna get here? Yeah! Can you talk about something? I dunno! Who are you? "
    "Okay. You really have a job? You really have a job? Okay. You accepted this "
    "thing for three dollars an hour? Okay. A small beer. Well, I don't think they "
    "drink beer. It's like sort of [H] a popsicle that tastes like beer. Yeah. "
    "Say again? Ah, not anymore!"
)
VOXMATRIX_PRED = (
    "I don't know. You guys told me you'd pay me three dollars an hour to sit here "
    "and talk to you about something. Uh, yeah, can you talk about something? What "
    "do you want me to talk about? I don't know. Who are you? Uh, oh, my name is "
    "Sophie. I drive big trucks. Okay. Like, I try not to hit things, but you know, "
    "every once in a while it happens. Do you really have a job? Huh? Do you really "
    "have a job? Yeah. Okay. Yeah, drive all over the place. I was in."
)

PROMPTS = {
    "default": "Transcribe the English audio into text.",
    "full": (
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--audio", default=AUDIO_PATH)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    info = sf.info(args.audio)
    print(f"audio: {args.audio}")
    print(f"duration: {info.duration:.2f}s  sr={info.samplerate}  channels={info.channels}")
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

    cases = [
        ("default_prompt", PROMPTS["default"], 0),
        ("full_prompt", PROMPTS["full"], 0),
        ("full_prompt_2048", PROMPTS["full"], 2048),
        ("full_prompt_4096", PROMPTS["full"], 4096),
    ]

    results = {
        "audio": args.audio,
        "model_path": args.model_path,
        "audio_duration_s": info.duration,
        "ref_words": word_count(REF),
        "voxmatrix_pred_words": word_count(VOXMATRIX_PRED),
        "voxmatrix_pred": VOXMATRIX_PRED,
        "ref": REF,
        "runs": [],
    }

    for name, prompt, max_new_tokens in cases:
        print(f"\n=== run: {name} (max_new_tokens={max_new_tokens or 'default'}) ===")
        out = run_one(model, processor, args.audio, prompt, max_new_tokens)
        out["case"] = name
        results["runs"].append(out)
        print(f"pred_words={out['pred_words']}  gen_tokens={out['generated_tokens']}  latency={out['latency_s']}s")
        print(f"pred: {out['pred'][:300]}{'...' if len(out['pred']) > 300 else ''}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
