#!/usr/bin/env python3
"""Build an Audio-Reasoning LLM-Judge pack from Native indexes.

Lexical metrics stay on the Native suite; this pack drives the semantic
auxiliary (local qwen3-omni-thinking multimodal judge).

Rubric mapping:
  - mmsu / vocalbench           -> semi-open (1-5, soft-align to ref)
  - uro_bench / vocalbench_zh / voicebench -> binary
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "mesh_eval/data/native"
OUT = ROOT / "outputs/audio_reasoning_judge_pack.jsonl"

SOURCES = [
    ("audio_reasoning_mmsu.jsonl", "semi-open", "en"),
    ("audio_reasoning_uro_bench.jsonl", "binary", "en"),
    ("audio_reasoning_vocalbench.jsonl", "semi-open", "en"),
    ("audio_reasoning_vocalbench_zh.jsonl", "binary", "zh"),
    ("audio_reasoning_voicebench.jsonl", "binary", "en"),
]


def main() -> int:
    rows = []
    for filename, rubric, lang in SOURCES:
        source = NATIVE / filename
        if not source.is_file():
            raise SystemExit(f"missing Native index: {source}")
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            ref = item.get("text", item.get("answer", ""))
            if ref in (None, ""):
                continue
            rows.append(
                {
                    "sample_id": item["sample_id"],
                    "capability": "audio_reasoning",
                    "rubric": rubric,
                    "lang": lang,
                    "source_lang": lang,
                    "WavPath": item["WavPath"],
                    "question": item.get("question", ""),
                    "ref": ref,
                    "pred": "",
                    "source_index": str(source),
                    "benchmark": item.get("dataset") or item.get("benchmark"),
                }
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
        encoding="utf-8",
    )
    print(json.dumps({"records": len(rows), "output": str(OUT)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
