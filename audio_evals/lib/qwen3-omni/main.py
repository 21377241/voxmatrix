import argparse
import json
import os
import select
import sys
import tempfile
import time


def stage(message):
    startup_stage = message.startswith(("import_", "load_model_")) or message == "main_start"
    if startup_stage or os.environ.get("QWEN3_OMNI_DEBUG_STAGE") == "1":
        print(
            f"[qwen3-omni-stage] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}",
            flush=True,
        )


stage("import_soundfile_start")
import soundfile as sf
stage("import_soundfile_done")
stage("import_torch_start")
import torch
stage("import_torch_done")
stage("import_transformers_start")
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
stage("import_transformers_done")
stage("import_qwen_omni_utils_start")
from qwen_omni_utils import process_mm_info
stage("import_qwen_omni_utils_done")


device = "cuda"
READY_SENTINEL = "__QWEN3_OMNI_READY__"


def load_model(path, **kwargs):
    stage(f"load_model_start path={path}")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
        **kwargs,
    )
    stage("load_model_model_done")

    processor = Qwen3OmniMoeProcessor.from_pretrained(path)
    stage("load_model_processor_done")
    return model, processor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--speech",
        action="store_true",
        default=False,
        help="Whether to use speech output",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="Ethan",
        help="Speaker name for speech generation",
    )
    parser.add_argument(
        "--speech-output-dir",
        type=str,
        default="",
        help="Persistent directory for generated speech audio.",
    )
    parser.add_argument(
        "--thinker-max-new-tokens",
        type=int,
        default=0,
        help="Maximum generated text tokens; 0 keeps the model default.",
    )
    config = parser.parse_args()
    stage("main_start")
    model, processor = load_model(config.path)
    print(f"{READY_SENTINEL} Model loaded from checkpoint: {config.path}", flush=True)

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
            conversation = json.loads(prompt[anchor + 2 :])
            stage(f"request_received prefix={prefix}")

            # Set whether to use audio in video
            USE_AUDIO_IN_VIDEO = True

            stage("apply_chat_template_start")
            text = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            stage("process_mm_info_start")
            audios, images, videos = process_mm_info(
                conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
            )
            stage("processor_call_start")
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
            )
            inputs = inputs.to(model.device).to(model.dtype)

            # Inference: Generation of the output text and audio
            generation_kwargs = {}
            if config.thinker_max_new_tokens > 0:
                generation_kwargs[
                    "thinker_max_new_tokens"
                ] = config.thinker_max_new_tokens
            if config.speech:
                stage("generate_speech_start")
                text_ids, audio = model.generate(
                    **inputs,
                    speaker=config.speaker,
                    return_audio=True,
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=USE_AUDIO_IN_VIDEO,
                    **generation_kwargs,
                )
                stage("generate_speech_done")
                text = processor.batch_decode(
                    text_ids.sequences[:, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if audio is not None:
                    if config.speech_output_dir:
                        os.makedirs(config.speech_output_dir, exist_ok=True)
                        audio_path = os.path.join(
                            config.speech_output_dir, f"{prefix[:-2]}.wav"
                        )
                    else:
                        with tempfile.NamedTemporaryFile(
                            suffix=".wav", delete=False
                        ) as handle:
                            audio_path = handle.name
                    sf.write(
                        audio_path,
                        audio.reshape(-1).detach().cpu().numpy(),
                        samplerate=24000,
                    )
                    retry = 3
                    while retry:
                        retry -= 1
                        print(
                            prefix
                            + json.dumps({"text": text[0], "audio": audio_path}),
                            flush=True,
                        )
                        rlist, _, _ = select.select([sys.stdin], [], [], 1)
                        if rlist:
                            finish = sys.stdin.readline().strip()
                            if finish == "{}close".format(prefix):
                                break
                        print("not found close signal, will emit again", flush=True)
                else:
                    retry = 3
                    while retry:
                        retry -= 1
                        print(prefix + json.dumps({"text": text[0]}), flush=True)
                        rlist, _, _ = select.select([sys.stdin], [], [], 1)
                        if rlist:
                            finish = sys.stdin.readline().strip()
                            if finish == "{}close".format(prefix):
                                break
                        print("not found close signal, will emit again", flush=True)
            else:
                stage("generate_text_start")
                text_ids, _ = model.generate(
                    **inputs,
                    use_audio_in_video=USE_AUDIO_IN_VIDEO,
                    return_audio=False,
                    thinker_return_dict_in_generate=True,
                    **generation_kwargs,
                )
                stage("generate_text_done")
                text = processor.batch_decode(
                    text_ids.sequences[:, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                retry = 3
                while retry:
                    retry -= 1
                    print(prefix + json.dumps({"text": text[0]}), flush=True)
                    rlist, _, _ = select.select([sys.stdin], [], [], 1)
                    if rlist:
                        finish = sys.stdin.readline().strip()
                        if finish == "{}close".format(prefix):
                            break
                    print("not found close signal, will emit again", flush=True)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print("Error:" + str(e), flush=True)
