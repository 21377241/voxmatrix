#!/usr/bin/env python3
"""Compute corpus chrF from saved Translation smoke predictions."""

from __future__ import annotations

import json
from pathlib import Path

import sacrebleu


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/native_translation_smoke"
FILES = {
    "covost2_zh_en": OUT_DIR / "native-covost2-zh-en-smoke.jsonl",
    "covost2_en_zh": OUT_DIR / "native-covost2-en-zh-smoke.jsonl",
    "mmsu": OUT_DIR / "native-mmsu-translation-smoke.jsonl",
}


def load(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("type") == "eval":
            rows.append(item["data"])
    return rows


def score(rows):
    pred = [str(x.get("pred", "")) for x in rows]
    ref = [str(x.get("ref", "")) for x in rows]
    return sacrebleu.corpus_chrf(pred, [ref]).score


def main():
    report = {}
    all_rows = []
    for name, path in FILES.items():
        rows = load(path)
        report[name] = {"sample_count": len(rows), "chrf": score(rows)}
        all_rows.extend(rows)
    report["overall"] = {"sample_count": len(all_rows), "chrf": score(all_rows)}
    output = OUT_DIR / "translation-chrf.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
