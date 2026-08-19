"""Manifest schema 校验。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from annotation.pipeline.common import load_jsonl, load_taxonomy
from mesh_eval.core.schema import normalize_record, validate_record
from mesh_eval.core.schema_v2 import is_canonical_v2, validate_canonical_sample


def validate_sample(sample: dict[str, Any], taxonomy: dict, index: int) -> list[str]:
    del taxonomy
    errors: list[str] = []
    prefix = f"样本[{index}]"

    if is_canonical_v2(sample):
        return [
            f"{prefix}: {error}"
            for error in validate_canonical_sample(
                sample, formal=sample.get("use_bucket") == "formal_subscores"
            )
        ]

    required = [
        "sample_id", "dataset", "task", "capability", "scenario",
        "condition", "input", "reference", "metrics", "use_bucket", "split",
    ]
    for field in required:
        if field not in sample or sample[field] in (None, "", []):
            errors.append(f"{prefix}: 缺少必填字段 '{field}'")

    normalized = normalize_record(sample, ref_col="text")
    errors.extend(
        f"{prefix}: {error}"
        for error in validate_record(normalized, ref_col="text")
    )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="校验 manifest")
    parser.add_argument("--manifest-dir", default=None, type=Path)
    parser.add_argument("--manifest-file", default=None, type=Path)
    parser.add_argument("--schema", default=None, type=Path, help="JSON Schema（可选）")
    args = parser.parse_args()

    taxonomy = load_taxonomy()
    schema_validator = None
    if args.schema:
        try:
            from jsonschema import Draft7Validator
        except ImportError:
            print(
                "警告: 当前环境未安装 jsonschema，仅执行 taxonomy 校验；"
                "安装 jsonschema>=4.0 后会同时执行 JSON Schema 校验。"
            )
        else:
            schema_data = json.loads(args.schema.read_text(encoding="utf-8"))
            schema_validator = Draft7Validator(schema_data)
    all_errors: list[str] = []
    total = 0

    files: list[Path] = []
    if args.manifest_file:
        files = [args.manifest_file]
    elif args.manifest_dir:
        files = sorted(args.manifest_dir.glob("**/*.jsonl"))

    for fpath in files:
        samples = load_jsonl(fpath)
        for i, sample in enumerate(samples):
            total += 1
            all_errors.extend(validate_sample(sample, taxonomy, i))
            if schema_validator is not None:
                for error in sorted(
                    schema_validator.iter_errors(sample),
                    key=lambda item: tuple(str(part) for part in item.path),
                ):
                    path = ".".join(str(part) for part in error.path) or "$"
                    all_errors.append(f"样本[{i}]: JSON Schema {path}: {error.message}")

    if all_errors:
        print(f"校验失败: {len(all_errors)} 个错误 / {total} 条样本")
        for err in all_errors[:50]:
            print(f"  ✗ {err}")
        if len(all_errors) > 50:
            print(f"  ... 还有 {len(all_errors) - 50} 个错误")
        sys.exit(1)
    else:
        print(f"校验通过: {total} 条样本")


if __name__ == "__main__":
    main()
