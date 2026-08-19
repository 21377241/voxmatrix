import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audio_evals.registry import Registry
from audio_evals.config import get_environment
from mesh_eval.core.benchmark_map import DEFAULT_BENCHMARK_MAP, load_benchmark_map
from mesh_eval.core.schema import (
    CONDITION_GROUPS,
    load_taxonomy,
    metric_catalog_names,
    normalize_metrics,
)


REQUIRED_FIELDS = (
    "registry_dataset",
    "task",
    "capability",
    "answer_type",
    "scenario",
    "condition",
    "metrics",
    "evaluators",
    "prompt",
    "use_bucket",
    "smoke_status",
)


def validate(path: str = "") -> Dict[str, Any]:
    data = load_benchmark_map(path)
    registry = Registry([REPO_ROOT / "registry", REPO_ROOT / "mesh_eval" / "registry"])
    dataset_names = set(registry._dataset)
    evaluator_names = set(registry._evaluator)
    prompt_names = set(registry._prompt)
    taxonomy = load_taxonomy()
    catalog_metrics = metric_catalog_names()
    strict_paths = get_environment("VOXMATRIX_STRICT_PATHS", "0") == "1"
    results: List[Dict[str, Any]] = []
    for name, config in data["benchmarks"].items():
        errors = []
        warnings = []
        for field in REQUIRED_FIELDS:
            if config.get(field) in (None, "", []):
                errors.append(f"missing field: {field}")
        dataset_name = config.get("registry_dataset")
        if dataset_name and dataset_name not in dataset_names:
            errors.append(f"dataset is not registered: {dataset_name}")
        prompt_name = config.get("prompt")
        if prompt_name and prompt_name not in prompt_names:
            errors.append(f"prompt is not registered: {prompt_name}")
        for evaluator in config.get("evaluators") or []:
            if evaluator not in evaluator_names:
                errors.append(f"evaluator is not registered: {evaluator}")
        task = config.get("task")
        capability = config.get("capability")
        if task not in (taxonomy.get("tasks") or []):
            errors.append(f"unknown task: {task}")
        if capability not in (taxonomy.get("capabilities") or {}).get(task, []):
            errors.append(f"capability {capability} does not belong to task {task}")
        scenario = config.get("scenario")
        if scenario not in (taxonomy.get("scenarios") or []):
            errors.append(f"unknown scenario: {scenario}")
        use_bucket = config.get("use_bucket")
        if use_bucket not in (taxonomy.get("use_buckets") or []):
            errors.append(f"unknown use_bucket: {use_bucket}")
        metrics = normalize_metrics(config.get("metrics"))
        if metrics != (config.get("metrics") or []):
            errors.append(f"metrics are not canonical: {config.get('metrics')}")
        for metric in metrics:
            if metric not in catalog_metrics:
                errors.append(f"metric is not in catalog: {metric}")
        condition = config.get("condition") or {}
        condition_taxonomy = taxonomy.get("conditions") or {}
        for group in CONDITION_GROUPS:
            if group not in condition:
                errors.append(f"condition is missing group: {group}")
                continue
            if condition[group] not in condition_taxonomy.get(group, []):
                errors.append(f"unknown condition {group}: {condition[group]}")
        for group in condition:
            if group not in condition_taxonomy:
                errors.append(f"unknown condition group: {group}")
        local_path = (config.get("data") or {}).get("local_path")
        if local_path and not os.path.exists(local_path):
            message = f"local_path does not exist: {local_path}"
            (errors if strict_paths else warnings).append(message)
        if config.get("smoke_status", "").startswith("blocked") and not config.get("requirements"):
            warnings.append("blocked benchmark has no requirements list")
        results.append(
            {
                "benchmark": name,
                "status": "ok" if not errors else "error",
                "errors": errors,
                "warnings": warnings,
            }
        )
    return {
        "version": data.get("version"),
        "mapping": str(path or DEFAULT_BENCHMARK_MAP),
        "benchmark_count": len(results),
        "error_count": sum(len(item["errors"]) for item in results),
        "warning_count": sum(len(item["warnings"]) for item in results),
        "results": results,
    }


def markdown_table(path: str = "") -> str:
    data = load_benchmark_map(path)
    lines = [
        "# Benchmark 与 Evaluator 映射表",
        "",
        f"> 配置源：`{path or DEFAULT_BENCHMARK_MAP}`；版本：`{data.get('version')}`。",
        "",
        "| Benchmark | Registry dataset | Task / Capability | Metrics | Evaluator | 本地数据 | 状态 | 额外依赖 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, config in data["benchmarks"].items():
        values = [
            name,
            config.get("registry_dataset", ""),
            f"{config.get('task', '')} / {config.get('capability', '')}",
            ", ".join(config.get("metrics") or []),
            ", ".join(config.get("evaluators") or []),
            (config.get("data") or {}).get("local_path", ""),
            config.get("smoke_status", ""),
            ", ".join(config.get("requirements") or []),
        ]
        escaped = [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="校验 Benchmark-evaluator 映射")
    parser.add_argument("--mapping", default="")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    report = validate(args.mapping)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_table(args.mapping), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
