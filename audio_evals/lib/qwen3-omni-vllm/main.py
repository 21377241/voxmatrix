"""Qwen3-Omni listen→text worker backed by vLLM (thinker only; no speech).

Protocol matches audio_evals/lib/qwen3-omni/main.py stdin JSON lines so the
existing OfflineModel client can be reused. Speech/TTS is not supported here —
keep HF Qwen3Omni with --speech for VocalBench IF / speech_generation.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import time

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor
from vllm import LLM, SamplingParams


def emit_and_wait_close(prefix: str, payload: dict, retries: int = 3) -> None:
    """Print response then consume `{prefix}close` (same handshake as HF worker)."""
    line = prefix + json.dumps(payload, ensure_ascii=False)
    retry = retries
    while retry:
        retry -= 1
        print(line, flush=True)
        rlist, _, _ = select.select([sys.stdin], [], [], 1)
        if rlist:
            finish = sys.stdin.readline().strip()
            if finish == "{}close".format(prefix):
                return
        if retry:
            print("not found close signal, will emit again", flush=True)

READY_SENTINEL = "__QWEN3_OMNI_READY__"


def stage(message: str) -> None:
    startup = message.startswith(("import_", "load_")) or message == "main_start"
    if startup or os.environ.get("QWEN3_OMNI_DEBUG_STAGE") == "1":
        print(
            f"[qwen3-omni-vllm-stage] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}",
            flush=True,
        )


def _parse_bool(value: str) -> bool | None:
    text = str(value).strip().lower()
    if text in ("default", "omit", "none", ""):
        return None
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(value)


def build_vllm_inputs(processor, conversation, *, truncation, use_audio_in_video: bool):
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=use_audio_in_video
    )
    mm_kwargs = {"use_audio_in_video": use_audio_in_video}
    if truncation is not None:
        mm_kwargs["truncation"] = truncation
    inputs = {
        "prompt": text,
        "multi_modal_data": {},
        "mm_processor_kwargs": mm_kwargs,
    }
    if images is not None:
        inputs["multi_modal_data"]["image"] = images
    if videos is not None:
        inputs["multi_modal_data"]["video"] = videos
    if audios is not None:
        inputs["multi_modal_data"]["audio"] = audios
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--thinker-max-new-tokens", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--tensor-parallel-size", type=int, default=0)
    parser.add_argument("--limit-mm-audio", type=int, default=16)
    parser.add_argument("--limit-mm-image", type=int, default=1)
    parser.add_argument("--limit-mm-video", type=int, default=1)
    parser.add_argument("--truncation", type=_parse_bool, default=False)
    parser.add_argument("--use-audio-in-video", type=_parse_bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    # Accepted so callers can share CLI flags with the HF worker; speech is rejected.
    parser.add_argument("--speech", action="store_true", default=False)
    parser.add_argument("--speaker", type=str, default="Ethan")
    parser.add_argument("--speech-output-dir", type=str, default="")
    config = parser.parse_args()

    if config.speech:
        raise SystemExit(
            "qwen3-omni-vllm does not support speech/TTS output; "
            "use audio_evals.models.qwen3_omni.Qwen3Omni with speech=true"
        )

    stage("main_start")
    tp = config.tensor_parallel_size or max(1, torch.cuda.device_count())
    llm_kwargs = dict(
        model=config.path,
        trust_remote_code=True,
        gpu_memory_utilization=config.gpu_memory_utilization,
        tensor_parallel_size=tp,
        limit_mm_per_prompt={
            "image": max(1, int(config.limit_mm_image)),
            "video": max(1, int(config.limit_mm_video)),
            "audio": max(1, int(config.limit_mm_audio)),
        },
        max_num_seqs=max(1, int(config.max_num_seqs)),
        max_model_len=int(config.max_model_len),
        seed=int(config.seed),
    )
    stage(f"load_llm_start kwargs={json.dumps(llm_kwargs)}")
    llm = LLM(**llm_kwargs)
    processor = Qwen3OmniMoeProcessor.from_pretrained(config.path)
    stage("load_llm_done")

    max_tokens = (
        int(config.thinker_max_new_tokens)
        if int(config.thinker_max_new_tokens) > 0
        else 8192
    )
    sampling_params = SamplingParams(
        temperature=float(config.temperature),
        top_p=1.0,
        top_k=-1,
        max_tokens=max_tokens,
    )
    print(f"{READY_SENTINEL} vLLM Model loaded from checkpoint: {config.path}", flush=True)

    use_audio_in_video = bool(config.use_audio_in_video)
    truncation = config.truncation

    while True:
        try:
            prompt = input()
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    "Error: Invalid conversation format, must contains  ->, but {}".format(
                        prompt
                    ),
                    flush=True,
                )
                continue
            prefix = prompt[:anchor].strip() + "->"
            raw = prompt[anchor + 2 :].strip()
            # Stale/late close lines must not be treated as requests.
            if raw == "close":
                continue
            conversation = json.loads(raw)
            stage(f"request_received prefix={prefix}")

            inputs = build_vllm_inputs(
                processor,
                conversation,
                truncation=truncation,
                use_audio_in_video=use_audio_in_video,
            )
            stage("generate_start")
            outputs = llm.generate([inputs], sampling_params=sampling_params)
            stage("generate_done")
            text = outputs[0].outputs[0].text
            emit_and_wait_close(prefix, {"text": text})
        except Exception as exc:
            print(f"Error:{type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
