#!/usr/bin/env python3
"""Build a QA LLM-Judge pack from Native indexes (8p31-aligned samples).

Lexical qa-exist-match stays on the Native suite; this pack drives the
semantic auxiliary (local qwen3-omni-thinking multimodal judge).

Rubric mapping (aligned with 乙-QA / URO scales):
  - ear_wdyl / vocalbench / vocalbench_zh / voicebench -> binary
  - spoken_squad / uro_bench -> semi-open
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "mesh_eval/data/native"
OUT = ROOT / "outputs/qa_judge_pack.jsonl"

SOURCES = [
    ("qa_ear_wdyl.jsonl", "binary", "en"),
    ("qa_spoken_squad.jsonl", "semi-open", "en"),
    ("qa_vocalbench.jsonl", "binary", "en"),
    ("qa_vocalbench_zh.jsonl", "binary", "zh"),
    ("qa_uro_bench.jsonl", "semi-open", "zh"),
    ("qa_voicebench.jsonl", "binary", "en"),
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
                    "capability": "qa",
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
