"""JSON-line worker for Step-Audio-2-mini (official transformers implementation)."""

import argparse
import gc
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

READY_SENTINEL = "__STEP_AUDIO2_READY__"
HERE = Path(__file__).resolve().parent
DEFAULT_PROMPT_WAV = str(HERE / "assets" / "default_female.wav")


def load_audio_compatible(file_path, target_rate=16000, max_length=None):
    # Recent torchaudio routes every load through TorchCodec, which the
    # Step environment does not ship; decode with soundfile instead.
    import soundfile as sf
    import torch
    from torchaudio.functional import resample

    samples, sample_rate = sf.read(str(file_path), dtype="float32", always_2d=True)
    audio = torch.from_numpy(samples[:, 0].copy())
    if int(sample_rate) != int(target_rate):
        audio = resample(audio, int(sample_rate), int(target_rate))
    if max_length is not None and audio.shape[0] > int(max_length):
        audio = audio[: int(max_length)]
    return audio


def load_audio_channels_compatible(file_path, *args, **kwargs):
    import soundfile as sf
    import torch

    samples, sample_rate = sf.read(str(file_path), dtype="float32", always_2d=True)
    return torch.from_numpy(samples.T.copy()), int(sample_rate)


def save_audio_compatible(file_object, waveform, sample_rate, *args, **kwargs):
    import soundfile as sf

    fmt = str(kwargs.get("format") or "wav").upper()
    sf.write(
        file_object,
        waveform.detach().float().cpu().numpy().T,
        int(sample_rate),
        format=fmt,
        subtype="PCM_16",
    )


def speech_token_budget(messages, configured_limit):
    """Bound speech continuation by the length of the text to be read.

    Step speech tokens run at 25 Hz.  Some deterministic reading requests never
    emit EOT and grow the KV cache until the GPU runs out of memory, so cap the
    budget at four tokens per target character plus a small allowance.
    """
    target_text = ""
    for message in messages:
        if message.get("role") not in {"user", "human"}:
            continue
        parts = message.get("content")
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "text":
                    target_text = str(part.get("text") or "")
    marker = "without adding any other words:"
    if marker in target_text:
        target_text = target_text.rsplit(marker, 1)[-1]
    chars = len(target_text.strip())
    if chars < 1:
        return int(configured_limit)
    return min(int(configured_limit), max(256, 4 * chars + 64))


def cached_generate(model, messages, max_new_tokens, repetition_penalty=1.0):
    """Greedy decoding with a KV cache, equivalent to ``StepAudio2.__call__``.

    The checkpoint's ``forward`` never returns ``past_key_values``, so the
    official ``generate`` path re-runs the audio encoder and the full prefix for
    every new token; long audio then decodes at ~2 tokens/s.
    """
    import torch
    from transformers import DynamicCache
    from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
    from utils import padding_mels

    llm, tokenizer = model.llm, model.llm_tokenizer
    parts, mels = model.apply_chat_template(messages)
    prompt_ids = []
    for part in parts:
        if isinstance(part, str):
            prompt_ids.append(tokenizer(text=part, return_tensors="pt")["input_ids"])
        else:
            prompt_ids.append(torch.tensor([part], dtype=torch.int32))
    input_ids = torch.cat(prompt_ids, dim=-1).long().cuda()
    penalty = (
        RepetitionPenaltyLogitsProcessor(repetition_penalty)
        if repetition_penalty != 1.0
        else None
    )
    with torch.no_grad():
        embeds = llm.model.embed_tokens(input_ids)
        if mels:
            wavs, wav_lens = padding_mels(mels)
            wavs, wav_lens = wavs.cuda(), wav_lens.cuda()
            if llm.bf16:
                wavs = wavs.bfloat16()
            audio, feat_lens = llm.encoder(wavs, wav_lens)
            audio = llm.adapter(audio)
            feat_lens = (feat_lens - 1) // 2 + 1
            locations = torch.nonzero(input_ids == 151688)
            locations[:, 1] += 1
            for idx in range(len(locations)):
                row, start = locations[idx]
                embeds[row][start : start + feat_lens[idx]] = audio[idx][: feat_lens[idx]]
        out = llm.model(inputs_embeds=embeds, past_key_values=DynamicCache(), use_cache=True)
        all_ids = input_ids
        generated = []
        for _ in range(int(max_new_tokens)):
            scores = llm.lm_head(out.last_hidden_state[:, -1:])[:, -1].float()
            if penalty is not None:
                scores = penalty(all_ids, scores)
            next_id = scores.argmax(dim=-1, keepdim=True)
            generated.append(int(next_id))
            all_ids = torch.cat([all_ids, next_id], dim=-1)
            if generated[-1] == model.eos_token_id:
                break
            out = llm.model(
                inputs_embeds=llm.model.embed_tokens(next_id),
                past_key_values=out.past_key_values,
                use_cache=True,
            )
    # Same slice as the official path: drop the final (EOT or cut-off) token.
    output_ids = generated[:-1]
    text = tokenizer.decode([i for i in output_ids if i < 151688])
    audio_tokens = [i - 151696 for i in output_ids if i > 151695]
    return output_ids, text, audio_tokens


def load(args):
    # Concurrent replicas otherwise race on writing the trust_remote_code
    # modules into one shared HF modules cache.
    os.environ["HF_MODULES_CACHE"] = os.path.join(
        tempfile.gettempdir(), f"voxmatrix-step-hf-modules-{os.getpid()}"
    )
    # Eager attention materialises heads x L x L scores: ~50 GiB at 30k tokens
    # (AudioAgentBench dialogues with the full knowledge base).
    os.environ.setdefault("STEP_AUDIO2_ATTN_IMPLEMENTATION", "sdpa")
    import stepaudio2

    stepaudio2.load_audio = load_audio_compatible
    model = stepaudio2.StepAudio2(args.path)
    converter = None
    if args.speech:
        import s3tokenizer
        import token2wav as token2wav_module

        s3tokenizer.load_audio = lambda file, sr=16000: load_audio_compatible(
            file, target_rate=sr
        )
        token2wav_module.torchaudio.load = load_audio_channels_compatible
        token2wav_module.torchaudio.save = save_audio_compatible
        converter = token2wav_module.Token2wav(str(Path(args.path) / "token2wav"))
    return model, converter


def run_request(model, converter, payload, args):
    messages = payload["messages"]
    generate_kwargs = {"do_sample": False, "repetition_penalty": 1.05}
    generate_kwargs.update(payload.get("generate_kwargs") or {})
    limit = int(generate_kwargs.pop("max_new_tokens", args.max_new_tokens))
    if args.speech:
        limit = speech_token_budget(messages, limit)
    try:
        if args.kv_cache and not generate_kwargs.get("do_sample"):
            tokens, text, audio_tokens = cached_generate(
                model,
                messages,
                limit,
                float(generate_kwargs.get("repetition_penalty", 1.0)),
            )
        else:
            tokens, text, audio_tokens = model(
                messages, max_new_tokens=limit, **generate_kwargs
            )
        response = {"text": str(text or "").strip()}
        if args.speech:
            if converter is None or not audio_tokens:
                raise RuntimeError(
                    f"Step-Audio-2 returned no speech tokens (output_tokens={len(tokens)})"
                )
            audio_bytes = converter(audio_tokens, prompt_wav=args.prompt_wav)
            out_dir = args.speech_output_dir or None
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(suffix=".wav", dir=out_dir, delete=False) as f:
                f.write(audio_bytes)
                response["audio"] = f.name
        return response
    finally:
        if args.speech:
            gc.collect()
            import torch

            torch.cuda.empty_cache()


def emit(prefix, response):
    print(prefix + json.dumps(response, ensure_ascii=False), flush=True)
    ack = sys.stdin.readline().strip()
    if ack != f"{prefix}close":
        raise RuntimeError("response acknowledgement was not received")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--speech", action="store_true")
    parser.add_argument("--prompt-wav", default=DEFAULT_PROMPT_WAV)
    parser.add_argument("--speech-output-dir", default="")
    parser.add_argument(
        "--no-kv-cache", dest="kv_cache", action="store_false",
        help="use the official uncached generate path",
    )
    args = parser.parse_args()
    model, converter = load(args)
    print(f"{READY_SENTINEL} Model loaded from checkpoint: {args.path}", flush=True)
    for raw in sys.stdin:
        prefix = ""
        try:
            anchor = raw.find("->")
            if anchor < 0:
                raise ValueError("request is missing protocol separator")
            prefix = raw[:anchor].strip() + "->"
            payload = json.loads(raw[anchor + 2 :])
            emit(prefix, run_request(model, converter, payload, args))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            if prefix:
                emit(prefix, {"error": f"{type(exc).__name__}: {exc}"})
            else:
                print(f"Error:{type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
