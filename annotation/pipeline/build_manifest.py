"""合并预标注与人工标注，生成最终 manifest。"""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from annotation.pipeline.common import load_jsonl, set_nested, write_jsonl


def load_human_annotations(csv_path: Path) -> dict[str, list[dict[str, str]]]:
    """按 sample_id 聚合人工标注结果。"""
    by_sample: dict[str, list[dict[str, str]]] = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "done":
                continue
            sid = row["sample_id"]
            by_sample.setdefault(sid, []).append(row)
    return by_sample


def apply_human_annotation(sample: dict[str, Any], annotations: list[dict[str, str]]) -> dict[str, Any]:
    result = deepcopy(sample)
    result.setdefault("label_meta", {"sources": {}, "confidence": 1.0, "reviewed": False})

    for ann in annotations:
        field = ann["label_field"]
        value = ann.get("suggested_value", "")

        if field == "reference_json" or field.startswith("reference."):
            if field == "reference_json" and value:
                result["reference"] = json.loads(value)
            else:
                ref_field = field.split(".", 1)[1]
                if value.startswith("{") or value.startswith("["):
                    result["reference"][ref_field] = json.loads(value)
                else:
                    result["reference"][ref_field] = value
        elif field.startswith("condition."):
            cond_key = field.split(".", 1)[1]
            result.setdefault("condition", {})[cond_key] = value
        elif field == "label_meta.reviewed":
            result["label_meta"]["reviewed"] = value.lower() in ("true", "1", "yes")
        else:
            set_nested(result, field, value)

        result["label_meta"]["sources"][field] = "human"
        if ann.get("annotator"):
            result["label_meta"]["annotator"] = ann["annotator"]

    if any(a.get("reviewed", "").lower() == "true" for a in annotations):
        result["label_meta"]["reviewed"] = True

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="生成最终 manifest")
    parser.add_argument("--prelabeled", required=True, type=Path)
    parser.add_argument("--human", default=None, type=Path, help="人工标注 CSV")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    samples = load_jsonl(args.prelabeled)
    human_map: dict[str, list[dict[str, str]]] = {}
    if args.human and args.human.exists():
        human_map = load_human_annotations(args.human)

    final_samples = []
    for sample in samples:
        sid = sample.get("sample_id", "")
        if sid in human_map:
            sample = apply_human_annotation(sample, human_map[sid])
        final_samples.append(sample)

    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / "manifest.jsonl"
    write_jsonl(out_path, final_samples)
    print(f"Manifest 生成完成: {len(final_samples)} 条 → {out_path}")


if __name__ == "__main__":
    main()
