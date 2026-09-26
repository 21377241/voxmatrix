"""Qwen3-Omni worker via vLLM-Omni (listen→text + speech/TTS).

- speech=False: Omni pipeline ``qwen3_omni_moe_thinker_only``, modalities=["text"]
  (same stack as qwen3_omni_long_audio AISHELL-4 sweep).
- speech=True: thinker→talker→code2wav with deploy YAML (default 2-GPU speech).

Protocol matches audio_evals/lib/qwen3-omni/main.py stdin JSON lines.
Requires envs/vllm_omni (not stock thinker-only vLLM).
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor
from vllm import SamplingParams
from vllm_omni.entrypoints.omni import Omni

READY_SENTINEL = "__QWEN3_OMNI_READY__"
AUDIO_SR = 24000


def stage(message: str) -> None:
    startup = message.startswith(("import_", "load_")) or message == "main_start"
    if startup or os.environ.get("QWEN3_OMNI_DEBUG_STAGE") == "1":
        print(
            f"[qwen3-omni-vllm-omni-stage] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}",
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


def write_thinker_deploy(
    path: Path, *, tp: int, util: float, max_model_len: int, max_tokens: int
) -> None:
    """Thinker-only deploy; devices/TP follow allocated GPUs."""
    devices = ",".join(str(i) for i in range(max(1, tp)))
    cfg = {
        "pipeline": "qwen3_omni_moe_thinker_only",
        "distributed_executor_backend": "mp",
        "enable_prefix_caching": False,
        "stages": [
            {
                "stage_id": 0,
                "devices": devices,
                "max_num_seqs": 1,
                "max_num_batched_tokens": int(max_model_len),
                "gpu_memory_utilization": float(util),
                "enforce_eager": True,
                "async_scheduling": False,
                "tensor_parallel_size": int(max(1, tp)),
                "trust_remote_code": True,
                "default_sampling_params": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_tokens": int(max_tokens),
                    "seed": 1234,
                },
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def build_omni_inputs(
    processor,
    conversation,
    *,
    use_audio_in_video: bool,
    modalities,
    truncation: bool | None = None,
):
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
        "modalities": list(modalities),
    }
    if images is not None:
        inputs["multi_modal_data"]["image"] = images
    if videos is not None:
        inputs["multi_modal_data"]["video"] = videos
    if audios is not None:
        inputs["multi_modal_data"]["audio"] = audios
    return inputs


def _to_numpy_audio(audio_tensor) -> np.ndarray:
    if isinstance(audio_tensor, list):
        chunks = []
        for t in audio_tensor:
            if isinstance(t, torch.Tensor):
                chunks.append(t.detach().float().cpu().reshape(-1).numpy())
            else:
                chunks.append(np.asarray(t, dtype=np.float32).reshape(-1))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    if isinstance(audio_tensor, torch.Tensor):
        return audio_tensor.detach().float().cpu().reshape(-1).numpy()
    arr = np.asarray(audio_tensor, dtype=np.float32)
    return arr.reshape(-1)


def _emit(prefix: str, payload: dict) -> None:
    body = prefix + json.dumps(payload, ensure_ascii=False)
    retry = 3
    while retry:
        retry -= 1
        print(body, flush=True)
        rlist, _, _ = select.select([sys.stdin], [], [], 1)
        if rlist:
            finish = sys.stdin.readline().strip()
            if finish == f"{prefix}close":
                return
        if retry:
            print("not found close signal, will emit again", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--speech", action="store_true", default=False)
    parser.add_argument("--speaker", type=str, default="Ethan")
    parser.add_argument("--speech-output-dir", type=str, default="")
    parser.add_argument("--thinker-max-new-tokens", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--tensor-parallel-size", type=int, default=0)
    parser.add_argument("--truncation", type=_parse_bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--stage-init-timeout", type=int, default=2400)
    parser.add_argument(
        "--deploy-config",
        type=str,
        default="",
        help="Optional Omni deploy YAML; empty → auto (thinker-only or bundled speech)",
    )
    config = parser.parse_args()

    stage("main_start")
    tp = int(config.tensor_parallel_size) or max(1, torch.cuda.device_count())
    max_text_tokens = (
        int(config.thinker_max_new_tokens)
        if int(config.thinker_max_new_tokens) > 0
        else (1200 if config.speech else 8192)
    )

    deploy_cfg = (
        str(config.deploy_config).strip()
        or os.environ.get("QWEN3_OMNI_SPEECH_DEPLOY_CONFIG", "").strip()
    )
    thinker_deploy_tmp: str | None = None
    if not deploy_cfg:
        if config.speech:
            deploy_cfg = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "deploy_qwen3_omni_speech_2gpu.yaml",
            )
        else:
            fd, thinker_deploy_tmp = tempfile.mkstemp(
                prefix="qwen3_omni_thinker_only_", suffix=".yaml"
            )
            os.close(fd)
            write_thinker_deploy(
                Path(thinker_deploy_tmp),
                tp=tp,
                util=float(config.gpu_memory_utilization),
                max_model_len=int(config.max_model_len),
                max_tokens=max_text_tokens,
            )
            deploy_cfg = thinker_deploy_tmp

    # Never pass top-level tensor_parallel_size / gpu_memory_utilization:
    # Omni applies them to every stage and fights deploy YAML (#5003).
    omni_kwargs = dict(
        model=config.path,
        trust_remote_code=True,
        deploy_config=deploy_cfg,
        max_num_seqs=max(1, int(config.max_num_seqs)),
        max_model_len=int(config.max_model_len),
        seed=int(config.seed),
        stage_init_timeout=int(config.stage_init_timeout),
        init_timeout=int(config.stage_init_timeout),
    )
    stage(
        "load_omni_start speech={} deploy={} kwargs={}".format(
            bool(config.speech),
            deploy_cfg,
            json.dumps({k: v for k, v in omni_kwargs.items() if k != "model"}, default=str),
        )
    )
    try:
        omni = Omni(**omni_kwargs)
        processor = Qwen3OmniMoeProcessor.from_pretrained(config.path)
    finally:
        # Keep temp deploy file for the process lifetime (Omni may re-read).
        pass
    stage(f"load_omni_done num_stages={getattr(omni, 'num_stages', '?')}")

    if config.speech:
        thinker_sp = SamplingParams(
            temperature=float(config.temperature if config.temperature > 0 else 0.6),
            top_p=0.9,
            top_k=-1,
            max_tokens=max_text_tokens,
            repetition_penalty=1.05,
            seed=int(config.seed),
        )
        talker_sp = SamplingParams(
            temperature=0.9,
            top_k=50,
            max_tokens=4096,
            seed=int(config.seed),
            detokenize=False,
            repetition_penalty=1.05,
            stop_token_ids=[2150],
        )
        code2wav_sp = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=4096 * 16,
            seed=int(config.seed),
            detokenize=True,
            repetition_penalty=1.1,
        )
        all_sp = [thinker_sp, talker_sp, code2wav_sp]
        sampling_params_list = all_sp[: getattr(omni, "num_stages", 3)]
        modalities = ["text", "audio"]
    else:
        sampling_params_list = [
            SamplingParams(
                temperature=float(config.temperature),
                top_p=1.0,
                top_k=-1,
                max_tokens=max_text_tokens,
                seed=int(config.seed),
            )
        ]
        modalities = ["text"]

    mode = "speech" if config.speech else "thinker-only"
    print(
        f"{READY_SENTINEL} vLLM-Omni ({mode}) loaded from checkpoint: {config.path}",
        flush=True,
    )

    voice = str(config.speaker or "Ethan").strip().lower() or "ethan"
    truncation = config.truncation

    while True:
        try:
            prompt = input()
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    f"Error: Invalid conversation format, must contains  ->, but {prompt}",
                    flush=True,
                )
                continue
            prefix = prompt[:anchor].strip() + "->"
            raw = prompt[anchor + 2 :].strip()
            if raw == "close":
                continue
            conversation = json.loads(raw)
            stage(f"request_received prefix={prefix}")

            inputs = build_omni_inputs(
                processor,
                conversation,
                use_audio_in_video=False,
                modalities=modalities,
                truncation=truncation,
            )
            if config.speech:
                inputs["voice_type"] = voice

            stage("generate_start")
            text_out = ""
            audio_np = None
            # py_generator=False: keep Omni engine alive across requests.
            for stage_outputs in omni.generate(
                [inputs], sampling_params_list, py_generator=False
            ):
                out_type = getattr(stage_outputs, "final_output_type", None)
                output = getattr(stage_outputs, "request_output", stage_outputs)
                if out_type == "text" or (
                    not config.speech and out_type in (None, "text")
                ):
                    try:
                        text_out = output.outputs[0].text
                    except Exception:
                        text_out = str(getattr(output, "outputs", output))
                elif out_type == "audio":
                    try:
                        audio_tensor = output.outputs[0].multimodal_output["audio"]
                        audio_np = _to_numpy_audio(audio_tensor)
                    except Exception as exc:
                        raise RuntimeError(f"audio extract failed: {exc}") from exc
            stage("generate_done")

            if not config.speech:
                _emit(prefix, {"text": text_out or ""})
                continue

            if audio_np is None or audio_np.size == 0:
                _emit(prefix, {"text": text_out or ""})
                continue

            if config.speech_output_dir:
                os.makedirs(config.speech_output_dir, exist_ok=True)
                audio_path = os.path.join(
                    config.speech_output_dir, f"{prefix[:-2]}.wav"
                )
            else:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                    audio_path = handle.name
            sf.write(audio_path, audio_np, samplerate=AUDIO_SR, format="WAV")
            _emit(prefix, {"text": text_out or "", "audio": audio_path})
        except Exception as exc:
            print(f"Error:{type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
