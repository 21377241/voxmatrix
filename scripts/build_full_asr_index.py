#!/usr/bin/env python3
"""Build the first full-index pilot: ASR on AISHELL-1 test.

The source is the L0 AISHELL-1 manifest.  This deliberately does not read
the existing native 10-sample index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(
    "/mnt/afs/oss_data/datasets/01_basic_transcription/AIShell-1/aishell1_test_asr.jsonl"
)
DEFAULT_OUTPUT = ROOT / "mesh_eval" / "data" / "full" / "asr" / "asr_aishell_1.jsonl"
DATASET_ROOT = DEFAULT_SOURCE.parent


def convert(source: Path) -> tuple[list[dict], dict]:
    rows = []
    seen = set()
    missing_audio = 0
    invalid = 0
    for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            invalid += 1
            continue
        sample_id = str(doc.get("id") or "aishell_1_%06d" % len(rows))
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id at source line {line_no}: {sample_id}")
        seen.add(sample_id)

        rel_audio = None
        text = ""
        for message in doc.get("messages") or []:
            for part in message.get("content") or []:
                if not isinstance(part, dict):
                    continue
                if part.get("audio_path"):
                    rel_audio = part["audio_path"]
                if part.get("text"):
                    text = str(part["text"]).strip()
        audio = DATASET_ROOT / rel_audio if rel_audio else None
        if audio is None or not audio.is_file():
            missing_audio += 1
            continue
        if not text:
            raise ValueError(f"empty reference at source line {line_no}: {sample_id}")
        rows.append(
            {
                "schema_version": "full-index/1.0",
                "sample_id": sample_id,
                "dataset": "aishell_1",
                "benchmark": "aishell_1",
                "capability": "asr",
                "split": "test",
                "WavPath": str(audio),
                "text": text,
                "source_index": str(source),
                "source_line": line_no,
            }
        )
    return rows, {
        "source_records": len(source.read_text(encoding="utf-8").splitlines()),
        "output_records": len(rows),
        "missing_audio": missing_audio,
        "invalid_json": invalid,
        "duplicate_ids": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.source.is_file():
        raise SystemExit(f"source manifest not found: {args.source}")
    rows, stats = convert(args.source)
    if not rows:
        raise SystemExit("full index is empty")
    from mesh_eval.data.full_index_meta import write_index_jsonl

    write_index_jsonl(args.output, rows, "asr__aishell_1")
    print(json.dumps({"output": str(args.output), **stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
