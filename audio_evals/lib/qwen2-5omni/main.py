import argparse
import json
import os
import sys
import tempfile
import time


def stage(message):
    startup_stage = message.startswith(("import_", "load_model_")) or message == "main_start"
    if startup_stage or os.environ.get("QWEN2_5_OMNI_DEBUG_STAGE") == "1":
        print(
            f"[qwen2.5-omni-stage] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}",
            flush=True,
        )


stage("import_soundfile_start")
import soundfile as sf
stage("import_soundfile_done")
stage("import_torch_start")
import torch
stage("import_torch_done")
stage("import_qwen_omni_utils_start")
from qwen_omni_utils import process_mm_info
stage("import_qwen_omni_utils_done")
stage("import_transformers_start")
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
stage("import_transformers_done")


READY_SENTINEL = "__QWEN2_5_OMNI_READY__"


def load_model(path, speech=False):
    stage(f"load_model_start path={path}")
    load_args = {
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda:0",
    }
    try:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            path, attn_implementation="flash_attention_2", **load_args
        )
    except Exception as exc:
        print(
            f"flash_attention_2 unavailable ({exc}); loading sdpa attention",
            flush=True,
        )
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            path, attn_implementation="sdpa", **load_args
        )
    stage("load_model_model_done")
    if not speech:
        model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(path)
    stage("load_model_processor_done")
    return model, processor


def response_audio_path(output_dir, prefix):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f"{prefix[:-2]}.wav")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        return handle.name


def emit(prefix, response):
    print(prefix + json.dumps(response, ensure_ascii=False), flush=True)
    acknowledgement = sys.stdin.readline().strip()
    if acknowledgement != f"{prefix}close":
        raise RuntimeError("response acknowledgement was not received")


def run_request(model, processor, conversation, config):
    stage("request_received")
    stage("apply_chat_template_start")
    rendered = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    stage("process_mm_info_start")
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=True
    )
    stage("processor_call_start")
    inputs = processor(
        text=rendered,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True,
    )
    inputs = inputs.to(model.device).to(model.dtype)
    generation_args = {
        "use_audio_in_video": True,
        "return_audio": config.speech,
        "thinker_do_sample": False,
    }
    if config.thinker_max_new_tokens > 0:
        generation_args["thinker_max_new_tokens"] = config.thinker_max_new_tokens
    stage("generate_start")
    result = model.generate(**inputs, speaker=config.speaker, **generation_args)
    stage("generate_done")
    if config.speech:
        text_ids, audio = result
    else:
        text_ids, audio = result, None
    generated_ids = text_ids[:, inputs["input_ids"].shape[1] :]
    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    response = {"text": text}
    if audio is not None:
        path = response_audio_path(config.speech_output_dir, config.request_prefix)
        sf.write(path, audio.reshape(-1).detach().cpu().numpy(), samplerate=24000)
        response["audio"] = path
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--speech", action="store_true")
    parser.add_argument("--speaker", default="Chelsie")
    parser.add_argument("--speech-output-dir", default="")
    parser.add_argument("--thinker-max-new-tokens", type=int, default=0)
    config = parser.parse_args()
    stage("main_start")
    model, processor = load_model(config.path, config.speech)
    print(f"{READY_SENTINEL} Model loaded from checkpoint: {config.path}", flush=True)

    for raw_request in sys.stdin:
        prefix = ""
        try:
            anchor = raw_request.find("->")
            if anchor < 0:
                raise ValueError("request is missing protocol separator")
            prefix = raw_request[:anchor].strip() + "->"
            config.request_prefix = prefix
            conversation = json.loads(raw_request[anchor + 2 :])
            emit(prefix, run_request(model, processor, conversation, config))
        except Exception as exc:
            if prefix:
                emit(prefix, {"error": f"{type(exc).__name__}: {exc}"})
            else:
                print(f"Error:{type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
