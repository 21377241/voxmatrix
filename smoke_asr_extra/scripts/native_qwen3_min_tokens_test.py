#!/usr/bin/env python3
"""Native Qwen3-Omni ASR with min_new_tokens ablation."""

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

MODEL_PATH = "/mnt/afs/models/Qwen3-Omni-30B-A3B-Instruct"
PROMPT = "Transcribe the English audio into text."
FULL_PROMPT = (
    "Transcribe the entire English audio from start to finish. "
    "Do not stop early."
)


def word_count(text: str) -> int:
    return len(text.split())


def run_one(
    model,
    processor,
    audio_path: str,
    prompt: str,
    max_new_tokens: int,
    min_new_tokens: int,
):
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

    generation_kwargs = {
        "return_audio": False,
        "thinker_return_dict_in_generate": True,
        "thinker_max_new_tokens": max_new_tokens,
    }
    if min_new_tokens > 0:
        generation_kwargs["thinker_min_new_tokens"] = min_new_tokens

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
        "min_new_tokens": min_new_tokens,
        "pred": pred,
        "pred_words": word_count(pred),
        "generated_tokens": gen_tokens,
        "latency_s": round(latency_s, 2),
    }


def load_manifest_refs():
    refs = {}
    path = Path(
        "/mnt/afs/users/wangyl/VoxMatrix/smoke_asr_extra/easycom/manifest.jsonl"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        refs[row["sample_id"]] = row["text"]
    return refs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    refs = load_manifest_refs()
    samples = [
        {
            "sample_id": "easycom_test_10_00_00_000",
            "audio": (
                "/mnt/afs/users/wangyl/VoxMatrix/smoke_asr_extra/easycom/audio/"
                "easycom_test_10_00_00_000.wav"
            ),
            "note": "baseline ~96 words / early stop mid-sentence",
        },
        {
            "sample_id": "easycom_test_10_03_00_332",
            "audio": (
                "/mnt/afs/users/wangyl/VoxMatrix/smoke_asr_extra/easycom/audio/"
                "easycom_test_10_03_00_332.wav"
            ),
            "note": "extreme early stop: only 6 words",
        },
    ]

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

    # (name, prompt, max_new_tokens, min_new_tokens)
    cases = [
        ("baseline", PROMPT, 1024, 0),
        ("min200", PROMPT, 1024, 200),
        ("min400", PROMPT, 1024, 400),
        ("full_min200", FULL_PROMPT, 1024, 200),
        ("full_min400", FULL_PROMPT, 1024, 400),
    ]

    results = {
        "model_path": args.model_path,
        "experiment": "min_new_tokens ablation on EasyCom",
        "samples": [],
    }

    for sample in samples:
        sid = sample["sample_id"]
        ref = refs[sid]
        info = sf.info(sample["audio"])
        sample_out = {
            "sample_id": sid,
            "audio": sample["audio"],
            "note": sample["note"],
            "audio_duration_s": info.duration,
            "ref": ref,
            "ref_words": word_count(ref),
            "runs": [],
        }
        print(f"\n########## {sid}  ref_words={sample_out['ref_words']} ##########")
        for name, prompt, max_tok, min_tok in cases:
            print(
                f"\n=== {name}: max={max_tok} min={min_tok or 0} ==="
            )
            out = run_one(
                model, processor, sample["audio"], prompt, max_tok, min_tok
            )
            out["case"] = name
            out["coverage_pct"] = round(
                out["pred_words"] / sample_out["ref_words"] * 100, 1
            )
            sample_out["runs"].append(out)
            print(
                f"pred_words={out['pred_words']}  gen_tokens={out['generated_tokens']}  "
                f"coverage={out['coverage_pct']}%  latency={out['latency_s']}s"
            )
            print(f"pred: {out['pred'][:400]}{'...' if len(out['pred']) > 400 else ''}")
        results["samples"].append(sample_out)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
