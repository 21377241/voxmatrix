"""Independent ASR for semantic LLM-judge template A1 (source audio transcript).

Default backends (local weights):
  - en / mix → Whisper-large-v3 @ /mnt/afs/models/whisper-large-v3
  - zh → FunASR paraformer-zh @ /mnt/afs/models/paraformer-zh

Results are cached under ``cache_dir`` keyed by wav path + mtime + size + lang.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_WHISPER = os.environ.get(
    "JUDGE_WHISPER_PATH", "/mnt/afs/models/whisper-large-v3"
)
DEFAULT_PARAFORMER = os.environ.get(
    "JUDGE_PARAFORMER_PATH", "/mnt/afs/models/paraformer-zh"
)

_whisper_model = None
_paraformer_model = None


def _wav_fingerprint(wav_path: str) -> str:
    st = os.stat(wav_path)
    raw = f"{os.path.abspath(wav_path)}|{st.st_mtime_ns}|{st.st_size}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: Path, fp: str, lang: str) -> Path:
    return cache_dir / f"{fp}.{lang}.json"


def normalize_lang(lang: Optional[str]) -> str:
    text = (lang or "en").strip().lower()
    if text in ("zh", "zh-cn", "zh_cn", "chinese", "cmn"):
        return "zh"
    if text in ("mix", "code_switch", "cs", "mixed", "zh-en", "en-zh"):
        return "mix"
    return "en"


def transcribe_for_judge(
    wav_path: str,
    lang: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force: bool = False,
) -> str:
    """Return transcript text for judge A1. Raises on hard failure."""
    if not wav_path or not os.path.isfile(wav_path):
        raise FileNotFoundError(f"wav not found: {wav_path}")
    lang_n = normalize_lang(lang)
    cache = Path(cache_dir) if cache_dir else None
    fp = _wav_fingerprint(wav_path)
    if cache is not None and not force:
        cache.mkdir(parents=True, exist_ok=True)
        hit = _cache_path(cache, fp, lang_n)
        if hit.is_file():
            data = json.loads(hit.read_text(encoding="utf-8"))
            text = str(data.get("text") or "").strip()
            if text:
                return text

    if lang_n == "zh":
        backend = os.environ.get("JUDGE_ASR_BACKEND", "local").strip().lower()
        if backend in ("openai", "openai_whisper_api", "api", "whisper-1"):
            text = _transcribe_whisper(wav_path, language="zh")
        else:
            text = _transcribe_paraformer(wav_path)
    else:
        # en + mix: Whisper (local large-v3 or API whisper-1)
        text = _transcribe_whisper(
            wav_path, language=None if lang_n == "mix" else "en"
        )

    text = (text or "").strip()
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        out = _cache_path(cache, fp, lang_n)
        out.write_text(
            json.dumps(
                {"wav": wav_path, "lang": lang_n, "text": text, "asr": "whisper" if lang_n != "zh" else "paraformer-zh"},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return text


def _transcribe_whisper(wav_path: str, language: Optional[str] = "en") -> str:
    global _whisper_model
    backend = os.environ.get("JUDGE_ASR_BACKEND", "local").strip().lower()
    if backend in ("openai", "openai_whisper_api", "api", "whisper-1"):
        from openai import OpenAI

        client = OpenAI()
        with open(wav_path, "rb") as handle:
            kwargs = {"model": os.environ.get("JUDGE_WHISPER_API_MODEL", "whisper-1"), "file": handle}
            if language in ("en", "zh"):
                kwargs["language"] = language
            result = client.audio.transcriptions.create(**kwargs)
        return str(getattr(result, "text", result) or "").strip()

    path = DEFAULT_WHISPER
    device = os.environ.get("JUDGE_ASR_DEVICE", "cpu")
    # HF checkout (config.json + safetensors) → transformers; else openai-whisper .pt
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json")):
        if _whisper_model is None:
            from transformers import pipeline

            logger.info("Loading HF Whisper pipeline from %s device=%s", path, device)
            _whisper_model = pipeline(
                "automatic-speech-recognition",
                model=path,
                device=0 if device.startswith("cuda") else -1,
            )
        gen_kwargs = {"task": "transcribe"}
        if language == "en":
            gen_kwargs["language"] = "english"
        elif language == "zh":
            gen_kwargs["language"] = "chinese"
        result = _whisper_model(wav_path, generate_kwargs=gen_kwargs)
        if isinstance(result, dict):
            return str(result.get("text") or "").strip()
        return str(result).strip()

    import whisper

    if _whisper_model is None:
        logger.info("Loading openai-whisper from %s", path)
        _whisper_model = whisper.load_model(path, device=device)
    kwargs = {"fp16": False}
    if language:
        kwargs["language"] = language
        kwargs["task"] = "transcribe"
    result = _whisper_model.transcribe(wav_path, **kwargs)
    return str(result.get("text") or "").strip()


def _transcribe_paraformer(wav_path: str) -> str:
    global _paraformer_model
    from funasr import AutoModel

    path = DEFAULT_PARAFORMER
    if _paraformer_model is None:
        logger.info("Loading paraformer from %s", path)
        _paraformer_model = AutoModel(model=path, disable_update=True)
    result = _paraformer_model.generate(input=wav_path)
    if not result:
        return ""
    first = result[0]
    if isinstance(first, dict):
        return str(first.get("text") or "").strip()
    return str(first).strip()
