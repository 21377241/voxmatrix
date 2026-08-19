"""规则预标注：按 dataset mapping 自动打标签。"""

from __future__ import annotations

import argparse
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from annotation.pipeline.common import (
    apply_field_mapping,
    load_jsonl,
    load_mapping,
    merge_defaults,
    write_jsonl,
)


def eval_condition_rule(raw: dict[str, Any], expr: str) -> bool:
    """简单条件表达式求值，支持 num_speakers 等数值比较。"""
    match = re.match(r"(\w+)\s*([><=]+)\s*(\d+)", expr.strip())
    if not match:
        return False
    field, op, val = match.group(1), match.group(2), int(match.group(3))
    field_val = raw.get(field)
    if field_val is None:
        return False
    if op == ">":
        return field_val > val
    if op == ">=":
        return field_val >= val
    if op == "==":
        return field_val == val
    if op == "<":
        return field_val < val
    if op == "<=":
        return field_val <= val
    return False


def apply_condition_rules(sample: dict[str, Any], raw: dict[str, Any], rules: list[dict]) -> dict[str, Any]:
    for rule in rules:
        if eval_condition_rule(raw, rule.get("if", "")):
            for key, value in rule.get("set", {}).items():
                if key.startswith("condition."):
                    sample["condition"][key.split(".", 1)[1]] = value
                    sample["label_meta"]["sources"][key] = "rule"
    return sample


def apply_subset_policy(
    sample: dict[str, Any],
    raw: dict[str, Any],
    mapping: dict[str, Any],
    taxonomy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """按 mapping.subset_policy 覆盖 condition/capability/split。"""
    policy = mapping.get("subset_policy") or {}
    subset = raw.get("subset") or sample.get("label_meta", {}).get("raw_subset")
    subset_cfg = (policy.get("subsets") or {}).get(subset or "", {})
    if subset_cfg:
        cond = subset_cfg.get("condition") or {}
        sample.setdefault("condition", {}).update(cond)
        for k in cond:
            sample["label_meta"]["sources"][f"condition.{k}"] = "rule"
        if subset_cfg.get("capability"):
            sample["capability"] = subset_cfg["capability"]
            sample["label_meta"]["sources"]["capability"] = "rule"
        if subset_cfg.get("split"):
            sample["split"] = subset_cfg["split"]
        sample["label_meta"]["subset_policy"] = subset

    duration = raw.get("duration_sec")
    threshold = float(policy.get("long_form_duration_sec") or 0)
    if duration is not None and threshold > 0 and float(duration) >= threshold:
        sample["capability"] = "long_form_asr"
        sample["label_meta"]["sources"]["capability"] = "rule"
        sample["label_meta"]["duration_sec"] = float(duration)
        if taxonomy is None:
            from annotation.pipeline.common import load_taxonomy

            taxonomy = load_taxonomy()
        metrics = taxonomy.get("default_metrics", {}).get("long_form_asr")
        if metrics:
            sample["metrics"] = list(metrics)
    elif duration is not None:
        sample["label_meta"]["duration_sec"] = float(duration)

    return sample


def prelabel_record(
    raw: dict[str, Any],
    mapping: dict[str, Any],
    expansion: dict[str, Any] | None = None,
    taxonomy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = expansion or mapping
    defaults = deepcopy(cfg.get("defaults", mapping.get("defaults", {})))
    defaults["dataset"] = mapping["dataset_id"]
    field_mapping = cfg.get("field_mapping", mapping.get("field_mapping", {}))

    sample = apply_field_mapping(raw, field_mapping, defaults)
    merge_defaults(sample, defaults)

    rules = mapping.get("condition_rules", [])
    if rules:
        sample = apply_condition_rules(sample, raw, rules)

    for key in ("task", "capability", "scenario"):
        sample["label_meta"]["sources"][key] = "rule"

    for cond_key, cond_val in sample.get("condition", {}).items():
        src_key = f"condition.{cond_key}"
        if src_key not in sample["label_meta"]["sources"]:
            sample["label_meta"]["sources"][src_key] = "rule"

    if raw.get("subset"):
        sample["label_meta"]["raw_subset"] = raw["subset"]
    if "split" in raw and raw["split"]:
        sample["split"] = raw["split"]

    sample = apply_subset_policy(sample, raw, mapping, taxonomy=taxonomy)
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description="规则预标注")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from annotation.pipeline.common import load_taxonomy

    mapping = load_mapping(args.dataset)
    taxonomy = load_taxonomy()
    raw_records = load_jsonl(args.input)
    if args.limit > 0:
        raw_records = raw_records[: args.limit]
    prelabeled: list[dict[str, Any]] = []

    expansions = mapping.get("manifest_expansions")
    if expansions:
        for raw in raw_records:
            for expansion in expansions:
                prelabeled.append(prelabel_record(raw, mapping, expansion, taxonomy))
    else:
        for raw in raw_records:
            prelabeled.append(prelabel_record(raw, mapping, taxonomy=taxonomy))

    write_jsonl(args.output, prelabeled)
    print(f"预标注完成: {len(prelabeled)} 条 manifest 草稿 → {args.output}")


if __name__ == "__main__":
    main()
