#!/usr/bin/env python3
"""Rebuild 10-sample Native ASR indexes from verified Full indexes."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "mesh_eval" / "data" / "full" / "asr"
NATIVE = ROOT / "mesh_eval" / "data" / "native"

SOURCES = {
    "aishell_1": "asr_aishell_1.jsonl",
    "aishell_5": "asr_aishell_5.jsonl",
    "common_voice": "asr_common_voice.jsonl",
    "librispeech": "asr_librispeech.jsonl",
    "magicdata_ramc": "asr_magicdata_ramc.jsonl",
    "sbcsae": "asr_sbcsae.jsonl",
    # Match the current Native catalog subset record (TEST_MEETING).
    "wenetspeech": "asr_wenetspeech_test_meeting.jsonl",
}


def pick(rows: list[dict], count: int = 10) -> list[dict]:
    if len(rows) <= count:
        return rows
    # Deterministic spread across the full index, avoiding a contiguous first-10
    # slice from one recording/session.
    positions = [round(i * (len(rows) - 1) / (count - 1)) for i in range(count)]
    return [rows[i] for i in positions]


def main() -> int:
    summary = {}
    for benchmark, filename in SOURCES.items():
        source = FULL / filename
        if not source.is_file():
            raise SystemExit(f"missing full index: {source}")
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        selected = pick(rows)
        for row in selected:
            row["native_source_index"] = str(source)
            row["native_sample_count"] = len(rows)
            if not Path(row["WavPath"]).is_file():
                raise SystemExit(f"missing audio for {benchmark}: {row['WavPath']}")
        output = NATIVE / f"asr_{benchmark}.jsonl"
        output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
            encoding="utf-8",
        )
        summary[benchmark] = {"full_count": len(rows), "native_count": len(selected), "output": str(output)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
