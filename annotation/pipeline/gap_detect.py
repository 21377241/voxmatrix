"""缺口识别：对比 required_labels，导出人工标注任务。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from annotation.pipeline.common import get_nested, load_jsonl, load_taxonomy


def get_required_reference(capability: str, taxonomy: dict) -> list[str]:
    return taxonomy.get("required_reference", {}).get(capability, [])


def get_required_conditions(capability: str, taxonomy: dict) -> list[str]:
    req = taxonomy.get("required_conditions", {})
    return req.get(capability, req.get("default", []))


def detect_gaps(sample: dict[str, Any], taxonomy: dict) -> list[dict[str, Any]]:
    """返回该样本的所有标签缺口。"""
    gaps = []
    capability = sample.get("capability", "")
    sample_id = sample.get("sample_id", "")

    for field in get_required_reference(capability, taxonomy):
        if get_nested(sample, f"reference.{field}") is None:
            gaps.append({
                "sample_id": sample_id,
                "label_field": f"reference.{field}",
                "current_value": "",
                "priority": "P0",
                "reason": f"capability={capability} 缺少必填 reference.{field}",
            })

    for cond_dim in get_required_conditions(capability, taxonomy):
        if not sample.get("condition", {}).get(cond_dim):
            gaps.append({
                "sample_id": sample_id,
                "label_field": f"condition.{cond_dim}",
                "current_value": "",
                "priority": "P1",
                "reason": f"缺少 condition.{cond_dim}",
            })

    if not sample.get("input", {}).get("audio_path"):
        gaps.append({
            "sample_id": sample_id,
            "label_field": "input.audio_path",
            "current_value": "",
            "priority": "P0",
            "reason": "缺少音频路径",
        })

    confidence = get_nested(sample, "label_meta.confidence", 1.0)
    if confidence < 0.7:
        gaps.append({
            "sample_id": sample_id,
            "label_field": "label_meta.reviewed",
            "current_value": str(confidence),
            "priority": "P1",
            "reason": f"标签置信度 {confidence} < 0.7，需人工确认",
        })

    llm_sources = [
        k for k, v in get_nested(sample, "label_meta.sources", {}).items()
        if v == "llm_draft"
    ]
    if llm_sources and not get_nested(sample, "label_meta.reviewed", False):
        gaps.append({
            "sample_id": sample_id,
            "label_field": "label_meta.reviewed",
            "current_value": "false",
            "priority": "P0",
            "reason": f"LLM 草稿未审核: {', '.join(llm_sources)}",
        })

    # P0：知识核对门禁
    if not get_nested(sample, "label_meta.knowledge_verify_done", False):
        gaps.append({
            "sample_id": sample_id,
            "label_field": "label_meta.knowledge_verify_done",
            "current_value": "false",
            "priority": "P0",
            "reason": "尚未完成知识核对（verify_prelabel），不得进 formal_subscores",
        })
    else:
        gate = get_nested(sample, "label_meta.knowledge_gate", "")
        if gate == "blocked":
            gaps.append({
                "sample_id": sample_id,
                "label_field": "label_meta.knowledge_gate",
                "current_value": "blocked",
                "priority": "P0",
                "reason": "知识核对 gate=blocked（含未决议 capability 冲突）",
            })
        elif gate == "needs_review":
            gaps.append({
                "sample_id": sample_id,
                "label_field": "label_meta.knowledge_gate",
                "current_value": "needs_review",
                "priority": "P1",
                "reason": "知识核对存疑，需人工复核",
            })

    return gaps


def sample_to_csv_row(sample: dict[str, Any], gap: dict[str, Any]) -> dict[str, str]:
    condition = sample.get("condition", {})
    speaker = condition.get("speaker", "")
    if isinstance(speaker, dict):
        speaker = str(speaker)

    import json
    return {
        "sample_id": sample.get("sample_id", ""),
        "dataset": sample.get("dataset", ""),
        "task": sample.get("task", ""),
        "capability": sample.get("capability", ""),
        "scenario": sample.get("scenario", ""),
        "condition_acoustic": str(condition.get("acoustic", "")),
        "condition_spatial": str(condition.get("spatial", "")),
        "condition_speaker": speaker,
        "condition_device": str(condition.get("device", "")),
        "condition_interaction": str(condition.get("interaction", "")),
        "audio_path": sample.get("input", {}).get("audio_path", ""),
        "reference_json": json.dumps(sample.get("reference", {}), ensure_ascii=False),
        "metrics": ",".join(sample.get("metrics", [])),
        "use_bucket": sample.get("use_bucket", ""),
        "split": sample.get("split", ""),
        "label_field": gap["label_field"],
        "current_value": gap.get("current_value", ""),
        "suggested_value": "",
        "confidence": str(get_nested(sample, "label_meta.confidence", "")),
        "label_source": "human",
        "status": "pending",
        "annotator": "",
        "reviewed": "false",
        "notes": gap.get("reason", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="识别标签缺口")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    taxonomy = load_taxonomy()
    samples = load_jsonl(args.input)
    args.output.mkdir(parents=True, exist_ok=True)

    all_gaps: list[dict] = []
    csv_rows: list[dict[str, str]] = []
    samples_with_gaps = 0

    for sample in samples:
        gaps = detect_gaps(sample, taxonomy)
        if gaps:
            samples_with_gaps += 1
            all_gaps.extend({"sample_id": sample["sample_id"], **g} for g in gaps)
            for gap in gaps:
                csv_rows.append(sample_to_csv_row(sample, gap))

    gap_report = {
        "total_samples": len(samples),
        "samples_with_gaps": samples_with_gaps,
        "total_gaps": len(all_gaps),
        "gaps_by_field": {},
        "gaps": all_gaps,
    }
    for gap in all_gaps:
        field = gap["label_field"]
        gap_report["gaps_by_field"][field] = gap_report["gaps_by_field"].get(field, 0) + 1

    import json
    report_path = args.output / "gap_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(gap_report, f, ensure_ascii=False, indent=2)

    if csv_rows:
        csv_path = args.output / "annotation_tasks.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"人工任务: {csv_path} ({len(csv_rows)} 条)")

    print(f"缺口报告: {report_path}")
    print(f"  总样本: {len(samples)}, 有缺口: {samples_with_gaps}, 缺口数: {len(all_gaps)}")


if __name__ == "__main__":
    main()
