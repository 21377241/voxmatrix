#!/usr/bin/env python3
"""Concatenate same-speaker LibriSpeech utterances into one ~60s single-speaker clip."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

SPEAKER_DIR = Path(
    "/mnt/afs/eval_data/01_basic_transcription/LibriSpeech/dev-clean/1272/128104"
)
OUT_DIR = Path("/mnt/afs/users/wangyl/VoxMatrix/smoke_asr_extra/results_native_qwen3")
OUT_WAV = OUT_DIR / "librispeech_60s_single.wav"
OUT_META = OUT_DIR / "librispeech_60s_single.meta.json"
TARGET_SECONDS = 60.0


def main():
    trans = {}
    for line in (SPEAKER_DIR / "1272-128104.trans.txt").read_text().splitlines():
        uid, text = line.split(" ", 1)
        trans[uid] = text

    chunks = []
    segments = []
    total = 0.0
    sr = None
    for flac in sorted(SPEAKER_DIR.glob("*.flac")):
        audio, file_sr = sf.read(flac)
        if sr is None:
            sr = file_sr
        elif file_sr != sr:
            raise RuntimeError(f"sample rate mismatch: {flac}")
        chunks.append(audio)
        dur = len(audio) / sr
        segments.append(
            {
                "file": str(flac),
                "utterance_id": flac.stem,
                "duration_s": round(dur, 3),
                "text": trans[flac.stem],
            }
        )
        total += dur
        if total >= TARGET_SECONDS:
            break

    audio = np.concatenate(chunks)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(OUT_WAV, audio, sr)

    ref = " ".join(seg["text"] for seg in segments)
    meta = {
        "source": "LibriSpeech dev-clean",
        "speaker": "1272-128104",
        "description": "single-speaker reading concatenated from consecutive utterances",
        "audio": str(OUT_WAV),
        "duration_s": round(len(audio) / sr, 3),
        "sample_rate": sr,
        "channels": 1,
        "segment_count": len(segments),
        "segments": segments,
        "ref": ref,
        "ref_words": len(ref.split()),
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
