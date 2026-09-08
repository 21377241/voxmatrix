#!/usr/bin/env python3
"""Pre-cut sentence wavs for smoke datasets that lost start/end timestamps."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

FFMPEG = "/mnt/afs/conda/envs/audioeval/bin/ffmpeg"
FULL_EVAL = Path("/mnt/afs/users/wangyl/vocal_bench/work/full_eval")
SMOKE = Path("/mnt/afs/users/wangyl/VoxMatrix/smoke_asr_all")
N = 10
DATASETS = ("aishell_4", "aishell_5", "alimeeting", "magicdata_ramc")


def slice_wav(src: Path, dst: Path, start: float, end: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if end <= start:
        raise ValueError(f"bad interval {start} {end} for {src}")
    cmd = [
        FFMPEG,
        "-y",
        "-ss",
        f"{start:.6f}",
        "-to",
        f"{end:.6f}",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size < 100:
        raise RuntimeError(
            f"ffmpeg failed {src} [{start},{end}]\n{proc.stderr[-800:]}"
        )


def rebuild(name: str) -> list[dict]:
    src_manifest = FULL_EVAL / name / "final" / "manifest.jsonl"
    out_dir = SMOKE / name
    audio_dir = out_dir / "audio_sliced"
    bak = out_dir / "manifest_unslliced.jsonl"
    cur = out_dir / "manifest.jsonl"
    if cur.is_file() and not bak.is_file():
        shutil.copy2(cur, bak)

    rows: list[dict] = []
    with src_manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= N:
                break
            doc = json.loads(line)
            inp = doc.get("input") or {}
            src = Path(inp["audio_path"])
            start = float(inp["start_time"])
            end = float(inp["end_time"])
            text = str((doc.get("reference") or {}).get("text") or "").strip()
            sample_id = str(doc.get("sample_id") or f"{name}_{len(rows):04d}")
            if not src.is_file() or not text or end <= start:
                continue
            dst = audio_dir / f"{sample_id}.wav"
            if not dst.is_file():
                print(f"[slice] {name} {len(rows)} {start:.3f}-{end:.3f}s {src.name}")
                slice_wav(src, dst, start, end)
            rows.append(
                {
                    "WavPath": str(dst),
                    "text": text,
                    "sample_id": sample_id,
                    "dataset": name,
                    "src_audio": str(src),
                    "start_time": start,
                    "end_time": end,
                }
            )
    if len(rows) < N:
        raise RuntimeError(f"{name}: only {len(rows)} samples")
    with cur.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def main() -> None:
    for name in DATASETS:
        rows = rebuild(name)
        print(f"[ok] {name}: {len(rows)} -> {SMOKE / name / 'manifest.jsonl'}")
        print(f"     first {rows[0]['start_time']:.3f}-{rows[0]['end_time']:.3f} {rows[0]['text'][:40]}")


if __name__ == "__main__":
    main()
