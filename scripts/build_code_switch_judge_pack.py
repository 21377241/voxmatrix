#!/usr/bin/env python3
"""Build a Code-Switch LLM-Judge pack from Native indexes.

Rubric mapping (semantic auxiliary; lexical metrics stay on the Native suite):
  - MMSU open-answer QA          -> semi-open (1-5, soft-align to ref)
  - VocalBench-zh knowledge      -> binary   (yes/no vs short ref)
  - VocalBench-zh open_ended     -> open     (1-5, free-form)
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "mesh_eval/data/native"
OUT = ROOT / "outputs/code_switch_judge_pack.jsonl"

SOURCES = [
    ("code_switch_mmsu.jsonl", "semi-open"),
    ("code_switch_vocalbench_zh.jsonl", "binary"),
    ("code_switch_vocalbench_zh_open_ended.jsonl", "open"),
]


def main() -> int:
    rows = []
    for filename, rubric in SOURCES:
        source = NATIVE / filename
        if not source.is_file():
            raise SystemExit(f"missing Native index: {source}")
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                {
                    "sample_id": item["sample_id"],
                    "capability": "code_switch",
                    "rubric": rubric,
                    "lang": "mix",
                    "source_lang": "mix",
                    "WavPath": item["WavPath"],
                    "question": item.get("question", ""),
                    "ref": item.get("text", item.get("answer", "")),
                    "pred": "",
                    "source_index": str(source),
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
