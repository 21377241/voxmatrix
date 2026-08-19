"""标注流水线公共工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ANNOTATION_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ANNOTATION_ROOT / "schema"
MAPPINGS_DIR = ANNOTATION_ROOT / "mappings"


def load_taxonomy() -> dict[str, Any]:
    with open(SCHEMA_DIR / "taxonomy.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_mapping(dataset_id: str) -> dict[str, Any]:
    mapping_path = MAPPINGS_DIR / f"{dataset_id}.yaml"
    if not mapping_path.exists():
        raise FileNotFoundError(f"未找到映射规则: {mapping_path}")
    with open(mapping_path, encoding="utf-8") as f:
        mapping = yaml.safe_load(f) or {}
    parent_id = mapping.get("extends")
    if not parent_id:
        return mapping
    if parent_id == dataset_id:
        raise ValueError(f"mapping 不能继承自身: {dataset_id}")
    parent = load_mapping(str(parent_id))
    return deep_merge(
        parent,
        {key: value for key, value in mapping.items() if key != "extends"},
    )


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_nested(data: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def set_nested(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = data
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def apply_field_mapping(
    raw: dict[str, Any],
    mapping: dict[str, Any],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """将原始记录按 mapping 规则转为 manifest 草稿。"""
    sample: dict[str, Any] = {
        "dataset": defaults.get("dataset"),
        "task": defaults.get("task"),
        "capability": defaults.get("capability"),
        "scenario": defaults.get("scenario"),
        "condition": dict(defaults.get("condition", {})),
        "metrics": list(defaults.get("metrics", [])),
        "use_bucket": defaults.get("use_bucket", "coverage_debt"),
        "split": defaults.get("split", "test"),
        "input": {},
        "reference": {},
        "label_meta": {
            "sources": {},
            "confidence": 1.0,
            "reviewed": False,
        },
    }

    for target_key, rule in mapping.items():
        if isinstance(rule, dict):
            source_key = rule.get("from", "")
            value = get_nested(raw, source_key) if "." in source_key else raw.get(source_key)
            transform = rule.get("transform")
            prefix = rule.get("prefix", "")

            if transform == "relative_path" and value:
                value = str(value)
            elif transform == "slurp_action_to_slots" and value:
                value = {"action": value}
            elif transform == "rttm_to_segments":
                value = value or []
            elif transform == "attribution_format":
                value = value or []

            if target_key == "sample_id" and value is not None:
                value = f"{prefix}{value}"
        else:
            value = get_nested(raw, rule) if "." in rule else raw.get(rule)

        if target_key.startswith("input.") or target_key.startswith("reference."):
            set_nested(sample, target_key, value)
            field_name = target_key.split(".", 1)[1]
            sample["label_meta"]["sources"][field_name] = "native"
        elif target_key == "sample_id":
            sample["sample_id"] = value
        elif target_key.startswith("condition."):
            set_nested(sample, target_key, value)
            sample["label_meta"]["sources"][target_key] = "native"
        else:
            sample[target_key] = value

    return sample


def merge_defaults(sample: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    for key, value in defaults.items():
        if key == "condition":
            sample.setdefault("condition", {}).update(value)
        elif key not in sample or sample[key] in (None, "", []):
            sample[key] = value
    return sample
