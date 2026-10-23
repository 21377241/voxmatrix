#!/usr/bin/env python3
"""Extract the verified WenetSpeech Parquet audio and build full ASR indexes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/mnt/afs/eval_data/01_basic_transcription/WenetSpeech")
PARQUET_ROOT = DATA_ROOT / "data"
FULL_ROOT = ROOT / "mesh_eval" / "data" / "full" / "asr"

SPLITS = {
    "test_net": ("test_net-00000-of-00002.parquet", "test_net-00001-of-00002.parquet"),
    "test_meeting": ("test_meeting-00000-of-00001.parquet",),
}


def build(split: str) -> dict:
    parquet_paths = [PARQUET_ROOT / name for name in SPLITS[split]]
    audio_root = DATA_ROOT / "audio" / split
    audio_root.mkdir(parents=True, exist_ok=True)
    output = FULL_ROOT / f"asr_wenetspeech_{split}.jsonl"
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    seen = set()
    records = 0
    missing_audio = 0
    with tmp_output.open("w", encoding="utf-8") as out:
        for parquet_path in parquet_paths:
            for batch in pq.ParquetFile(parquet_path).iter_batches(
                batch_size=256, columns=["sid", "text", "subsets", "WavPath", "audio"]
            ):
                for row in batch.to_pylist():
                    sid = str(row["sid"])
                    if sid in seen:
                        raise ValueError(f"duplicate sid: {sid}")
                    seen.add(sid)
                    audio = row.get("audio") or {}
                    audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
                    audio_name = str(row.get("WavPath") or f"{sid}.opus")
                    if not audio_bytes:
                        missing_audio += 1
                        continue
                    audio_path = audio_root / audio_name
                    if not audio_path.is_file():
                        audio_path.write_bytes(audio_bytes)
                    item = {
                        "schema_version": "full-index/1.0",
                        "sample_id": sid,
                        "dataset": "wenetspeech",
                        "benchmark": "wenetspeech",
                        "subset": split.upper(),
                        "capability": "asr",
                        "split": "test",
                        "language": "zh",
                        "WavPath": str(audio_path),
                        "text": str(row.get("text") or "").strip(),
                        "source_index": str(parquet_path),
                        "audio_format": "opus",
                    }
                    out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    records += 1
    os.replace(tmp_output, output)
    from mesh_eval.data.full_index_meta import finalize_jsonl

    # Streaming write above; attach subset-manifest fields after close.
    # test_net is orphan; still annotate with the meeting/test catalog record.
    finalize_jsonl(output, "asr__wenetspeech")
    return {"records": records, "missing_audio": missing_audio, "output": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(SPLITS), action="append")
    args = parser.parse_args()
    FULL_ROOT.mkdir(parents=True, exist_ok=True)
    selected = args.split or sorted(SPLITS)
    summary = {split: build(split) for split in selected}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
