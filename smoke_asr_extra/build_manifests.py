#!/usr/bin/env python3
"""Build 10-sample ASR smoke manifests for WenetSpeech / SBCSAE / EasyCom."""
from __future__ import annotations

import json
import os
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent
N = 10


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_wenetspeech() -> list[dict]:
    from datasets import load_from_disk

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    hf_root = Path(
        "/mnt/afs/users/chenrj/UltraEval-Audio/hf_datasets/WenetSpeech_c/test_net"
    )
    out_audio = ROOT / "wenetspeech" / "audio"
    out_audio.mkdir(parents=True, exist_ok=True)

    ds = load_from_disk(str(hf_root))
    rows: list[dict] = []
    for idx in range(len(ds)):
        if len(rows) >= N:
            break
        ex = ds[idx]
        text = str(ex.get("text") or "").strip()
        sid = str(ex.get("sid") or f"wenetspeech_{idx}")
        audio = ex.get("audio") or {}
        array = audio.get("array")
        sr = int(audio.get("sampling_rate") or 16000)
        if not text or array is None:
            continue
        wav_path = out_audio / f"{sid}.wav"
        sf.write(str(wav_path), array, sr, subtype="PCM_16", format="WAV")
        rows.append(
            {
                "WavPath": str(wav_path),
                "text": text,
                "sample_id": f"wenetspeech_test_net_{sid}",
                "dataset": "wenetspeech",
            }
        )
    if len(rows) < N:
        raise RuntimeError(f"WenetSpeech: only collected {len(rows)} samples")
    return rows


def build_sbcsae() -> list[dict]:
    manifest = Path(
        "/mnt/afs/eval_data/benchmarks/SBCSAE_Public_Speech/manifests/asr.jsonl"
    )
    rows: list[dict] = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= N:
                break
            row = json.loads(line)
            audio = Path(row["audio"])
            if not audio.is_file():
                continue
            start = float(row["start_seconds"])
            end = float(row["end_seconds"])
            text = str(row.get("text") or "").strip()
            if not text or end <= start:
                continue
            rows.append(
                {
                    "WavPath": {
                        "path": str(audio),
                        "start_time": start,
                        "end_time": end,
                    },
                    "text": text,
                    "sample_id": row["id"],
                    "dataset": "sbcsae",
                }
            )
    if len(rows) < N:
        raise RuntimeError(f"SBCSAE: only collected {len(rows)} samples")
    return rows


def _easycom_reference(transcription_json: Path) -> str:
    raw = transcription_json.read_bytes()
    payload = None
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "latin-1"):
        try:
            payload = json.loads(raw.decode(encoding))
            break
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    if payload is None:
        raise ValueError(f"cannot decode transcription json: {transcription_json}")
    parts: list[str] = []
    for item in payload:
        text = str(item.get("Transcription") or "").strip()
        if not text:
            continue
        if text.startswith("[") and "]" in text:
            text = text.split("]", 1)[1].strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _to_mono_16k(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(str(source), always_2d=True)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1, keepdims=True)
    if sr != 16000:
        try:
            import librosa

            audio = librosa.resample(
                audio[:, 0], orig_sr=sr, target_sr=16000
            ).reshape(-1, 1)
            sr = 16000
        except Exception:
            step = max(1, int(round(sr / 16000)))
            audio = audio[::step, :1]
            sr = 16000
    sf.write(str(target), audio, sr, subtype="PCM_16", format="WAV")


def build_easycom() -> list[dict]:
    manifest = Path("/mnt/afs/eval_data/benchmarks/EasyCom/manifests/asr.jsonl")
    out_audio = ROOT / "easycom" / "audio"
    rows: list[dict] = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= N:
                break
            row = json.loads(line)
            source = Path(row["audio"])
            transcription = Path(row["transcription_json"])
            if not source.is_file() or not transcription.is_file():
                continue
            try:
                text = _easycom_reference(transcription)
            except ValueError:
                continue
            if not text:
                continue
            sample_id = row["id"]
            target = out_audio / f"{sample_id}.wav"
            if not target.is_file():
                _to_mono_16k(source, target)
            rows.append(
                {
                    "WavPath": str(target),
                    "text": text,
                    "sample_id": sample_id,
                    "dataset": "easycom",
                }
            )
    if len(rows) < N:
        raise RuntimeError(f"EasyCom: only collected {len(rows)} samples")
    return rows


def main() -> None:
    datasets = {
        "wenetspeech": ("zh", build_wenetspeech()),
        "sbcsae": ("en", build_sbcsae()),
        "easycom": ("en", build_easycom()),
    }
    summary = {"count": len(datasets), "datasets": []}
    for name, (lang, rows) in datasets.items():
        manifest = ROOT / name / "manifest.jsonl"
        write_jsonl(manifest, rows)
        summary["datasets"].append(
            {"name": name, "lang": lang, "manifest": str(manifest), "n": len(rows)}
        )
        print(f"[ok] {name}: {len(rows)} samples -> {manifest}")
    (ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
