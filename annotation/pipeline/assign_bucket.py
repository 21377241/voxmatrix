"""use_bucket 自动分配（含 P0 知识核对门禁）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from annotation.pipeline.common import get_nested, load_jsonl, load_taxonomy, write_jsonl
from annotation.pipeline.gap_detect import detect_gaps


def assign_bucket(sample: dict[str, Any], taxonomy: dict, *, require_knowledge_gate: bool = True) -> str:
    """根据知识门禁、标签完整度和置信度分配 use_bucket。"""
    meta = sample.get("label_meta") or {}
    gate = meta.get("knowledge_gate")
    verify_done = bool(meta.get("knowledge_verify_done"))

    # P0：未做知识核对 → 不得进 formal
    if require_knowledge_gate and not verify_done:
        return "diagnostic_evidence"

    # P0：知识门禁阻断
    if gate == "blocked":
        return "coverage_debt"

    # P0：知识存疑 → 诊断桶
    if gate == "needs_review" or meta.get("knowledge_flags"):
        return "diagnostic_evidence"

    gaps = detect_gaps(sample, taxonomy)
    p0_gaps = [g for g in gaps if g.get("priority") == "P0"]
    if p0_gaps:
        return "coverage_debt"

    confidence = get_nested(sample, "label_meta.confidence", 1.0)
    reviewed = get_nested(sample, "label_meta.reviewed", True)
    sources = get_nested(sample, "label_meta.sources", {})

    has_llm_unreviewed = any(v == "llm_draft" for v in sources.values()) and not reviewed
    has_flagged = any(v == "knowledge_flagged" for v in sources.values())
    has_low_confidence = confidence < 0.7

    # 仅 rule/acoustic 推断且无知识确认时，更谨慎
    verified_sources = {
        "knowledge_verified",
        "knowledge_revise",
        "native",
        "human",
    }
    has_unverified_infer = any(
        v in ("rule", "acoustic_model") for v in sources.values()
    ) and not any(v in verified_sources for v in sources.values())

    acoustic = sample.get("condition", {}).get("acoustic", "")
    if acoustic in ("traffic_noise", "wind_noise") and (
        "rule" in sources.values() or "acoustic_model" in sources.values()
    ):
        return "stress_test"

    if has_llm_unreviewed or has_flagged or has_low_confidence:
        return "diagnostic_evidence"

    if has_unverified_infer and gaps:
        return "diagnostic_evidence"

    # 关键字段（task/capability）必须通过知识门禁或人工
    cap_src = sources.get("capability", "rule")
    if require_knowledge_gate and cap_src in ("rule", "llm_draft", "knowledge_flagged"):
        # rule 但门禁 pass 时，verify 会把 capability 标成 knowledge_verified
        if gate not in {"pass", "pass_with_revise"} and not reviewed:
            return "diagnostic_evidence"

    capability = sample.get("capability", "")
    required_ref = taxonomy.get("required_reference", {}).get(capability, [])
    for field in required_ref:
        if get_nested(sample, f"reference.{field}") is None:
            return "coverage_debt"

    # gate pass / pass_with_revise：尊重 mapping 预填上限，不擅自升 formal
    if gate in {"pass", "pass_with_revise"}:
        preset = sample.get("use_bucket")
        if preset in {"diagnostic_evidence", "coverage_debt", "stress_test"}:
            return preset
        return "formal_subscores"

    # 兼容：人工 reviewed 且完整
    if reviewed and not p0_gaps:
        return "formal_subscores"

    return "diagnostic_evidence"


def main() -> None:
    parser = argparse.ArgumentParser(description="分配 use_bucket")
    parser.add_argument("--input", required=True, type=Path, help="manifest 目录或 jsonl 文件")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-without-knowledge-gate",
        action="store_true",
        help="调试用：允许跳过知识核对门禁",
    )
    args = parser.parse_args()

    taxonomy = load_taxonomy()
    require_gate = not args.allow_without_knowledge_gate

    if args.input.is_file():
        samples = load_jsonl(args.input)
    else:
        samples = []
        for jsonl_file in sorted(args.input.glob("*.jsonl")):
            # 跳过分流副文件，避免重复
            if any(
                jsonl_file.name.endswith(f".{g}.jsonl")
                for g in ("pass", "pass_with_revise", "needs_review", "blocked")
            ):
                continue
            samples.extend(load_jsonl(jsonl_file))

    bucket_counts: dict[str, int] = {}
    gate_counts: dict[str, int] = {}
    for sample in samples:
        bucket = assign_bucket(sample, taxonomy, require_knowledge_gate=require_gate)
        sample["use_bucket"] = bucket
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        g = get_nested(sample, "label_meta.knowledge_gate", "none")
        gate_counts[str(g)] = gate_counts.get(str(g), 0) + 1

    if args.output.suffix == ".jsonl":
        out_path = args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        out_path = args.output / "manifest.jsonl"
    write_jsonl(out_path, samples)

    summary_path = out_path.parent / "bucket_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total": len(samples),
                "buckets": bucket_counts,
                "knowledge_gates": gate_counts,
                "require_knowledge_gate": require_gate,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"分桶完成: {out_path}")
    for bucket, count in sorted(bucket_counts.items()):
        print(f"  {bucket}: {count}")
    print("知识门禁:")
    for g, count in sorted(gate_counts.items()):
        print(f"  {g}: {count}")


if __name__ == "__main__":
    main()
