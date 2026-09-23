#!/usr/bin/env python3
"""Build a Translation LLM-Judge pack from the corrected Native indexes."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "mesh_eval" / "data" / "native"
OUT = ROOT / "outputs" / "speech_translation_judge_pack.jsonl"

SOURCES = [
    ("speech_translation_covost2_zh_en.jsonl", "Translate into English"),
    ("speech_translation_covost2_en_zh.jsonl", "Translate into Chinese"),
    ("speech_translation_mmsu.jsonl", "Translate into English"),
]


def main() -> int:
    rows = []
    for filename, direction in SOURCES:
        source = NATIVE / filename
        if not source.is_file():
            raise SystemExit(f"missing Native index: {source}")
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append({
                "sample_id": item["sample_id"],
                "capability": "speech_translation",
                "rubric": "translation",
                "WavPath": item["WavPath"],
                "question": item.get("question", direction),
                "direction": direction,
                "source_lang": item.get("source_lang"),
                "target_lang": item.get("target_lang"),
                "ref": item.get("text", ""),
                "pred": "",
                "source_index": str(source),
            })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    print(json.dumps({"records": len(rows), "output": str(OUT)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
