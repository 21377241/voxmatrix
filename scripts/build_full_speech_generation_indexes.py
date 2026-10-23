#!/usr/bin/env python3
"""Build full speech_generation indexes (LibriTTS text → TTS source).

Native Catalog: speech_generation__libritts (test-clean).
Full Index stores every utterance with text ref; WavPath is the LibriTTS
source clip for provenance (prompt is text-only, does not feed WavPath).
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "mesh_eval/data/full/speech_generation"

LIBRITTS_ROOT = Path("/mnt/afs/eval_data/01_basic_transcription/LibriTTS")
# Catalog / Native do test-clean only; other splits listed in manifest but not slotted.
SPLITS = (
    ("test-clean", "test", "test-clean"),
)


def _write(rows: list[dict], name: str) -> tuple[int, Path]:
    from mesh_eval.data.full_index_meta import slot_id_from_filename, write_index_jsonl

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    slot_id = slot_id_from_filename(OUT.name, name)
    n = write_index_jsonl(path, rows, slot_id)
    return n, path


def build_split(split_dir: str, split: str, subset: str) -> tuple[list[dict], dict]:
    root = LIBRITTS_ROOT / split_dir
    rows: list[dict] = []
    missing = 0
    skipped = 0
    if not root.is_dir():
        return rows, {"missing_root": True, "missing_audio": 0, "skipped": 0}

    for tsv in sorted(root.glob("*/*/*.trans.tsv")):
        with tsv.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    skipped += 1
                    continue
                utt_id, text = parts[0], parts[1]
                normalized = parts[2] if len(parts) >= 3 and parts[2].strip() else text
                if not utt_id or not text.strip():
                    skipped += 1
                    continue
                wav = tsv.parent / f"{utt_id}.wav"
                if not wav.is_file():
                    missing += 1
                    continue
                speaker_id = utt_id.split("_", 1)[0]
                rows.append(
                    {
                        "schema_version": "full-index/1.0",
                        "sample_id": utt_id,
                        "dataset": "libritts",
                        "benchmark": "libritts",
                        "capability": "speech_generation",
                        "protocol": "speech_generation",
                        "subset": subset,
                        "split": split,
                        "speaker_id": speaker_id,
                        "WavPath": str(wav),
                        "text": text,
                        "answer": text,
                        "normalized_text": normalized,
                        "language": "en",
                        "source_index": str(tsv),
                    }
                )
    return rows, {"missing_audio": missing, "skipped": skipped, "missing_root": False}


def main() -> None:
    report: dict = {}
    all_rows: list[dict] = []
    for split_dir, split, subset in SPLITS:
        rows, meta = build_split(split_dir, split, subset)
        all_rows.extend(rows)
        report[subset] = {"n": len(rows), **meta}

    n, path = _write(all_rows, "speech_generation_libritts.jsonl")
    speakers = {r["speaker_id"] for r in all_rows}
    report["total"] = {
        "n": n,
        "path": str(path),
        "n_speakers": len(speakers),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
