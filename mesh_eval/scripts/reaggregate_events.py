import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mesh_eval.agg.mesh import MeshAgg
from mesh_eval.core.subset_manifest import (
    DEFAULT_EXCLUDED_CAPABILITIES,
    flatten_subset_profile,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Reaggregate one or more VoxMatrix result-event JSONL files"
    )
    parser.add_argument("result_jsonl", nargs="+")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--report-format", choices=("flat", "nested", "both"), default="flat"
    )
    parser.add_argument("--min-slice-size", type=int, default=5)
    parser.add_argument("--metric-field", action="append", default=[])
    parser.add_argument("--exclude-capability", action="append", default=[])
    return parser.parse_args()


def discover_result_files(values: Sequence[str]) -> List[str]:
    files = []
    seen = set()
    for value in values:
        path = Path(value).expanduser().resolve()
        candidates = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
        for candidate in candidates:
            if not candidate.is_file():
                raise FileNotFoundError(str(candidate))
            resolved = str(candidate.resolve())
            if resolved not in seen:
                seen.add(resolved)
                files.append(resolved)
    if not files:
        raise ValueError("no result JSONL files were found")
    return files


def _compact(value: Any) -> Any:
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return value


def flatten_event_dimensions(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: Dict[str, Any] = {}
    for key, item in value.items():
        if item in (None, "", []):
            continue
        if key == "scenario" and isinstance(item, dict):
            result["scenario"] = item
            for field in ("primary", "secondary"):
                if item.get(field) not in (None, "", []):
                    result[f"scenario__{field}"] = item[field]
        elif key == "metadata" and isinstance(item, dict):
            for field, metadata_value in item.items():
                if metadata_value not in (None, "", []):
                    result[f"metadata__{field}"] = _compact(metadata_value)
        elif key == "subset_profile":
            result.update(flatten_subset_profile(item))
        else:
            result[key] = _compact(item)
    return result


def event_score_row(event: Dict[str, Any], location: str = "event") -> Dict[str, Any]:
    if not isinstance(event, dict):
        raise ValueError(f"{location}: event must be an object")
    data = event.get("data") or {}
    if not isinstance(data, dict):
        raise ValueError(f"{location}: event.data must be an object")
    row = flatten_event_dimensions(event.get("dimensions"))
    for key, value in data.items():
        if (
            key in row
            and row[key] not in (None, "", [])
            and value not in (None, "", [])
            and row[key] != value
        ):
            raise ValueError(
                f"{location}: event dimension {key!r} conflicts with event.data"
            )
        row[key] = value
    run_context = event.get("run_context") or {}
    if isinstance(run_context, dict) and run_context.get("benchmark_id"):
        row.setdefault("benchmark_id", run_context["benchmark_id"])
    return row


def load_event_scores(
    paths: Sequence[str], *, excluded_capabilities: Iterable[str] = ()
) -> List[Dict[str, Any]]:
    excluded = {str(item) for item in excluded_capabilities}
    scores: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                location = f"{path}:{line_no}"
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{location}: invalid JSON: {exc}") from exc
                event_type = event.get("type")
                if event_type not in {"eval", "error"}:
                    continue
                row = event_score_row(event, location)
                if str(row.get("capability") or "") in excluded:
                    continue
                if event_type == "error":
                    row.setdefault("failure", 1)
                    row.setdefault("timeout", 0)
                scores.append(row)
    return scores


def _insert_path(target: Dict[str, Any], path: str, value: Any) -> None:
    parts = [part for part in path.split("/") if part]
    if not parts:
        return
    current = target
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _prefix_tree(flat: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    marker = prefix.rstrip("/") + "/"
    for key, value in flat.items():
        if key.startswith(marker):
            _insert_path(result, key[len(marker) :], value)
    return result


def _prefix_values(flat: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    marker = prefix.rstrip("/") + "/"
    for key, value in flat.items():
        if key.startswith(marker):
            result[key[len(marker) :].replace("/", "__")] = value
    return result


def _set_group_value(target: Dict[str, Any], suffix: str, value: Any) -> None:
    target[suffix.replace("/", "__")] = value


def build_nested_report(
    flat: Dict[str, Any], input_files: Sequence[str], *, include_flat: bool = False
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "schema_version": "suite-aggregation/1.0",
        "input_file_count": len(input_files),
        "input_files": list(input_files),
        "sample_count": int(flat.get("sample_count") or 0),
        "overall": _prefix_values(flat, "overall"),
        "coverage": _prefix_tree(flat, "coverage"),
        "by_capability": {},
        "by_scenario": {},
    }
    if any(key.startswith("formal/") for key in flat):
        report["formal"] = {
            "sample_count": int(flat.get("formal/sample_count") or 0),
            "overall": _prefix_values(flat, "formal/overall"),
        }

    capability_prefix = "capability/"
    for key, value in flat.items():
        if not key.startswith(capability_prefix):
            continue
        remainder = key[len(capability_prefix) :]
        if "/" not in remainder:
            continue
        capability, suffix = remainder.split("/", 1)
        target = report["by_capability"].setdefault(capability, {})
        _set_group_value(target, suffix, value)

    scenario_prefix = "scenario__primary+capability/"
    for key, value in flat.items():
        if not key.startswith(scenario_prefix):
            continue
        remainder = key[len(scenario_prefix) :]
        if "/" not in remainder:
            continue
        label, suffix = remainder.split("/", 1)
        if "|" not in label:
            continue
        scenario, capability = label.split("|", 1)
        scenario_target = report["by_scenario"].setdefault(
            scenario, {"by_capability": {}}
        )
        capability_target = scenario_target["by_capability"].setdefault(
            capability, {}
        )
        _set_group_value(capability_target, suffix, value)

    if include_flat:
        report["flat"] = flat
    return report


def aggregate_event_files(
    input_paths: Sequence[str],
    *,
    excluded_capabilities: Iterable[str] = DEFAULT_EXCLUDED_CAPABILITIES,
    report_format: str = "flat",
    min_slice_size: int = 5,
    metric_fields: Sequence[str] = (),
) -> Dict[str, Any]:
    if report_format not in {"flat", "nested", "both"}:
        raise ValueError(f"unsupported report format: {report_format!r}")
    files = discover_result_files(input_paths)
    scores = load_event_scores(files, excluded_capabilities=excluded_capabilities)
    flat = MeshAgg(
        metric_fields=list(metric_fields) or None,
        min_slice_size=min_slice_size,
    )(scores)
    if report_format == "flat":
        return flat
    return build_nested_report(flat, files, include_flat=report_format == "both")


def _atomic_text(path: str, value: str) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main():
    args = get_args()
    excluded = set(DEFAULT_EXCLUDED_CAPABILITIES)
    excluded.update(args.exclude_capability)
    result = aggregate_event_files(
        args.result_jsonl,
        excluded_capabilities=excluded,
        report_format=args.report_format,
        min_slice_size=args.min_slice_size,
        metric_fields=args.metric_field,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        _atomic_text(args.output, text)
    print(text, end="")


if __name__ == "__main__":
    main()
