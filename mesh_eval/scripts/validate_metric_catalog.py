import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audio_evals.registry import Registry


DEFAULT_CATALOG = REPO_ROOT / "mesh_eval" / "config" / "metric_evaluator_catalog.yaml"


def as_list(value: Any) -> List[str]:
    if value in (None, ""):
        return []
    return [str(item) for item in value] if isinstance(value, list) else [str(value)]


def validate(path: Path) -> Dict[str, Any]:
    catalog = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    registry = Registry([REPO_ROOT / "registry", REPO_ROOT / "mesh_eval" / "registry"])
    available = set(registry._evaluator)
    router_metrics = set(
        registry._evaluator["mesh-router-evaluator"]["args"]["metric_routes"]
    )
    results = []
    for metric, config in (catalog.get("metrics") or {}).items():
        errors = []
        status = config.get("status")
        if status not in (catalog.get("status_definitions") or {}):
            errors.append(f"unknown status: {status}")
        for evaluator in as_list(config.get("evaluator")):
            if evaluator not in available:
                errors.append(f"evaluator is not registered: {evaluator}")
        language_routes = config.get("evaluator_by_language") or {}
        if language_routes and not isinstance(language_routes, dict):
            errors.append("evaluator_by_language must be an object")
            language_routes = {}
        for language, evaluator in language_routes.items():
            if not str(language).strip():
                errors.append("evaluator_by_language has an empty language")
            for name in as_list(evaluator):
                if name not in available:
                    errors.append(
                        f"language evaluator is not registered: {language}={name}"
                    )
        if config.get("evaluator") and metric not in router_metrics:
            errors.append("metric is not present in mesh-router-evaluator.metric_routes")
        if status and status.startswith("requires_") and not config.get("dependencies"):
            errors.append("dependency status has no dependencies")
        results.append({"metric": metric, "status": status, "errors": errors})
    return {
        "version": catalog.get("version"),
        "metric_count": len(results),
        "status_counts": dict(Counter(item["status"] for item in results)),
        "error_count": sum(len(item["errors"]) for item in results),
        "results": results,
    }


def markdown(path: Path) -> str:
    catalog = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    lines = [
        "# Evaluator 与依赖清单",
        "",
        f"> 配置源：`{path}`；版本：`{catalog.get('version')}`。",
        "",
        "| Metric | 类别 | Evaluator | 状态 | 依赖/说明 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for metric, config in (catalog.get("metrics") or {}).items():
        evaluator = ", ".join(as_list(config.get("evaluator"))) or "-"
        detail = ", ".join(str(item) for item in config.get("dependencies") or [])
        if config.get("notes"):
            detail = f"{detail}; {config['notes']}" if detail else str(config["notes"])
        values = [metric, config.get("category", ""), evaluator, config.get("status", ""), detail]
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="校验并导出 metric evaluator catalog")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    report = validate(args.catalog)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(args.catalog), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
