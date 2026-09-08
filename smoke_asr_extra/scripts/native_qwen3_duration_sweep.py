#!/usr/bin/env python3
"""Cut EasyCom audio into 10/20/.../60s prefixes and run native Qwen3-Omni ASR."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

MODEL_PATH = "/mnt/afs/models/Qwen3-Omni-30B-A3B-Instruct"
SRC_AUDIO = (
    "/mnt/afs/users/wangyl/VoxMatrix/smoke_asr_extra/easycom/audio/"
    "easycom_test_10_00_00_000.wav"
)
OUT_DIR = Path(
    "/mnt/afs/users/wangyl/VoxMatrix/smoke_asr_extra/results_native_qwen3/"
    "duration_sweep_easycom_000"
)
PROMPT = "Transcribe the English audio into text."
DURATIONS = [10, 20, 30, 40, 50, 60]


def word_count(text: str) -> int:
    return len(text.split())


def prepare_clips():
    audio, sr = sf.read(SRC_AUDIO)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clips = []
    for sec in DURATIONS:
        n = min(len(audio), int(sec * sr))
        path = OUT_DIR / f"prefix_{sec:02d}s.wav"
        sf.write(path, audio[:n], sr)
        clips.append(
            {
                "seconds": sec,
                "path": str(path),
                "actual_duration_s": round(n / sr, 3),
            }
        )
        print(f"wrote {path}  dur={n/sr:.3f}s")
    return clips, sr


def run_one(model, processor, audio_path: str):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": PROMPT},
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

    t0 = time.time()
    text_ids, _ = model.generate(
        **inputs,
        use_audio_in_video=use_audio_in_video,
        return_audio=False,
        thinker_return_dict_in_generate=True,
        thinker_max_new_tokens=1024,
    )
    latency_s = time.time() - t0
    pred = processor.batch_decode(
        text_ids.sequences[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    gen_tokens = int(text_ids.sequences.shape[1] - inputs["input_ids"].shape[1])
    return {
        "pred": pred,
        "pred_words": word_count(pred),
        "generated_tokens": gen_tokens,
        "latency_s": round(latency_s, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--output", default=str(OUT_DIR / "results.json"))
    args = parser.parse_args()

    print("preparing prefix clips ...")
    clips, sr = prepare_clips()
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

    # full 60s ref from manifest
    ref = ""
    man = Path(
        "/mnt/afs/users/wangyl/VoxMatrix/smoke_asr_extra/easycom/manifest.jsonl"
    )
    for line in man.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["sample_id"] == "easycom_test_10_00_00_000":
            ref = row["text"]
            break

    results = {
        "source_audio": SRC_AUDIO,
        "sample_id": "easycom_test_10_00_00_000",
        "prompt": PROMPT,
        "model_path": args.model_path,
        "full_ref": ref,
        "full_ref_words": word_count(ref),
        "runs": [],
    }

    for clip in clips:
        print(f"\n=== prefix {clip['seconds']}s ===")
        out = run_one(model, processor, clip["path"])
        out.update(clip)
        results["runs"].append(out)
        print(
            f"pred_words={out['pred_words']}  gen_tokens={out['generated_tokens']}  "
            f"latency={out['latency_s']}s"
        )
        print(f"pred: {out['pred'][:300]}{'...' if len(out['pred']) > 300 else ''}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== summary =====")
    print(f"{'sec':>4} {'tokens':>7} {'words':>6} {'latency':>8}  pred_tail")
    for r in results["runs"]:
        print(
            f"{r['seconds']:>4} {r['generated_tokens']:>7} {r['pred_words']:>6} "
            f"{r['latency_s']:>7.1f}s  ...{r['pred'][-60:]!r}"
        )
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
