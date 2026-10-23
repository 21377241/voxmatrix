import argparse
import json
import logging
import os
import sys
import tempfile
import time
import traceback

import soundfile as sf
import torch
from kimia_infer.api.kimia import KimiAudio

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

READY_SENTINEL = "__KIMI_AUDIO_READY__"


def emit(prefix, response):
    print(prefix + json.dumps(response, ensure_ascii=False), flush=True)
    ack = sys.stdin.readline().strip()
    if ack != f"{prefix}close":
        raise RuntimeError("response acknowledgement was not received")


def run_request(model, payload, config, sampling_params):
    messages = payload["messages"] if "messages" in payload else payload
    output_type = "both" if config.speech else "text"
    with torch.no_grad():
        wav, text = model.generate(
            messages, **sampling_params, output_type=output_type
        )
    response = {"text": str(text or "").strip()}
    if config.speech:
        out_dir = config.speech_output_dir or None
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=out_dir, delete=False) as f:
            sf.write(
                f.name,
                (
                    wav.detach().cpu().view(-1).numpy()
                    if torch.is_tensor(wav)
                    else wav
                ),
                24000,
            )
            response["audio"] = f.name
    del wav, text
    torch.cuda.empty_cache()
    return response


def main():
    # Concurrent replicas otherwise race on writing trust_remote_code modules
    # into one shared HF modules cache (KimiAudioConfig AttributeError).
    os.environ["HF_MODULES_CACHE"] = os.path.join(
        tempfile.gettempdir(), f"voxmatrix-kimi-hf-modules-{os.getpid()}"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default="moonshotai/Kimi-Audio-7B-Instruct",
        help="Path or HF repo for Kimi-Audio model",
    )
    parser.add_argument(
        "--speech",
        action="store_true",
        default=False,
        help="Whether to use speech output",
    )
    parser.add_argument(
        "--audio_tokenizer_path",
        type=str,
        default="THUDM/glm-4-voice-tokenizer",
        help="Path or HF repo for the GLM-4-Voice audio tokenizer",
    )
    parser.add_argument(
        "--speech_output_dir",
        type=str,
        default="",
        help="Directory for persistent speech wavs (defaults to tempfile)",
    )
    config = parser.parse_args()

    start_time = time.time()
    model = KimiAudio(
        model_path=config.model_path,
        load_detokenizer=config.speech,
        audio_tokenizer_path=config.audio_tokenizer_path,
    )
    end_time = time.time()
    logger.info(f"Model loading took {end_time - start_time:.2f} seconds")
    logger.info(f"Using Kimi-Audio model: {config.model_path}")

    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.1,
        "text_repetition_window_size": 16,
        "max_new_tokens": -1,
    }

    print(
        f"{READY_SENTINEL} Model loaded from checkpoint: {config.model_path}",
        flush=True,
    )
    for raw in sys.stdin:
        prefix = ""
        try:
            anchor = raw.find("->")
            if anchor < 0:
                raise ValueError("request is missing protocol separator")
            prefix = raw[:anchor].strip() + "->"
            payload = json.loads(raw[anchor + 2 :])
            emit(prefix, run_request(model, payload, config, sampling_params))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            torch.cuda.empty_cache()
            if prefix:
                emit(prefix, {"error": f"{type(exc).__name__}: {exc}"})
            else:
                print(f"Error:{type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
