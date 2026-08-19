"""比对 mappings/*.yaml 与方案 §13.1 期望（spec_131.yaml）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from annotation.pipeline.common import ANNOTATION_ROOT, MAPPINGS_DIR, load_mapping

SPEC_PATH = ANNOTATION_ROOT / "schema" / "spec_131.yaml"


def load_spec() -> dict[str, Any]:
    with open(SPEC_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def contribution_key(item: dict[str, Any]) -> tuple:
    return (
        item.get("task"),
        item.get("capability"),
        item.get("scenario"),
        item.get("use_bucket"),
    )


def check_dataset(dataset_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    datasets = spec.get("datasets") or {}
    mapping_path = MAPPINGS_DIR / f"{dataset_id}.yaml"
    if not mapping_path.exists():
        # 方案有、仓库尚未建 mapping：视为待办警告，不阻断单库流水线外的总览
        return {
            "dataset_id": dataset_id,
            "status": "pending_mapping",
            "errors": [],
            "warnings": [f"方案 §13.1 已收录，但缺少 mapping: {mapping_path}"],
        }

    mapping = load_mapping(dataset_id)
    spec_dataset = mapping.get("spec_dataset", dataset_id)
    if spec_dataset not in datasets:
        return {
            "dataset_id": dataset_id,
            "spec_dataset": spec_dataset,
            "status": "no_spec",
            "errors": [],
            "warnings": [f"方案 §13.1 未收录 {spec_dataset}，跳过硬对齐"],
        }
    expected_list = datasets[spec_dataset].get("expected") or []
    contributions = mapping.get("contributions") or []
    defaults = mapping.get("defaults") or {}

    errors: list[str] = []
    warnings: list[str] = []

    # 每个 expected 至少被 contributions 或 defaults 覆盖
    contrib_keys = {contribution_key(c) for c in contributions}
    for exp in expected_list:
        key = contribution_key(exp)
        covered = key in contrib_keys
        # defaults 也算主贡献覆盖
        if (
            defaults.get("task") == exp.get("task")
            and defaults.get("capability") == exp.get("capability")
            and (exp.get("scenario") is None or defaults.get("scenario") == exp.get("scenario"))
            and (
                exp.get("use_bucket") is None
                or defaults.get("use_bucket") == exp.get("use_bucket")
            )
        ):
            covered = True

        if not covered:
            # 宽松：只比 task+capability
            soft = any(
                c.get("task") == exp.get("task") and c.get("capability") == exp.get("capability")
                for c in contributions
            ) or (
                defaults.get("task") == exp.get("task")
                and defaults.get("capability") == exp.get("capability")
            )
            if soft:
                warnings.append(
                    f"task/capability 对齐但 scenario/bucket 可能漂移: expected={exp}"
                )
            else:
                errors.append(f"未覆盖方案期望: {exp}")

        # metrics
        metrics_opts = exp.get("metrics_any_of")
        if metrics_opts and defaults.get("capability") == exp.get("capability"):
            actual = defaults.get("metrics") or []
            if actual and actual not in metrics_opts and list(actual) not in metrics_opts:
                # 允许超集
                if not any(set(opt).issubset(set(actual)) for opt in metrics_opts):
                    warnings.append(
                        f"metrics 与方案不一致: actual={actual}, expected_one_of={metrics_opts}"
                    )

    status = "ok" if not errors else "mismatch"
    return {
        "dataset_id": dataset_id,
        "spec_dataset": spec_dataset,
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "mapping_file": str(mapping_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 mapping 与方案 §13.1 对齐")
    parser.add_argument("--dataset", default=None, help="只检查单个数据集；默认检查全部有 mapping 的")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="有 error 则非零退出")
    args = parser.parse_args()

    spec = load_spec()
    if args.dataset:
        ids = [args.dataset]
    else:
        ids = sorted({p.stem for p in MAPPINGS_DIR.glob("*.yaml")} | set(spec.get("datasets", {})))

    results = []
    for did in ids:
        r = check_dataset(did, spec)
        # 指定单个数据集时，缺 mapping 升为 error
        if args.dataset and r["status"] == "pending_mapping":
            r["status"] = "missing_mapping"
            r["errors"] = r.pop("warnings", []) or [f"缺少 mapping: {did}"]
            r["warnings"] = []
        results.append(r)
    summary = {
        "spec_version": spec.get("version"),
        "source_doc": spec.get("source_doc"),
        "results": results,
        "error_count": sum(len(r["errors"]) for r in results),
        "warning_count": sum(len(r["warnings"]) for r in results),
    }

    print(f"§13.1 对齐检查 (spec={spec.get('version')})")
    for r in results:
        mark = {
            "ok": "✓",
            "mismatch": "✗",
            "missing_mapping": "✗",
            "pending_mapping": "·",
            "no_spec": "·",
        }.get(r["status"], "?")
        print(f"  {mark} {r['dataset_id']}: {r['status']}")
        for e in r["errors"]:
            print(f"      ERROR: {e}")
        for w in r["warnings"]:
            print(f"      WARN: {w}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"报告: {args.report}")

    if args.strict and summary["error_count"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
