import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_REQUIRED_TYPES = ("prompt", "inference", "post_process", "eval")


def count_manifest_rows(path: Path) -> int:
    with open(path, encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def count_registry_dataset(
    dataset_name: str, registry_paths: str = "mesh_eval/registry", limit: int = 0
) -> int:
    from audio_evals.registry import Registry

    paths = [REPO_ROOT / "registry"]
    paths.extend(Path(item) for item in registry_paths.split() if item)
    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(path)
    dataset = Registry(unique_paths).get_dataset(dataset_name)
    if dataset is None:
        raise ValueError(f"dataset is not registered: {dataset_name}")
    counter = getattr(dataset, "count", None)
    if callable(counter):
        return int(counter(limit))
    return len(dataset.load(limit))


def validate_overall_summary(
    path: Path,
    expected_count: int,
    required_keys: Iterable[str] = (),
    require_sample_count: bool = True,
) -> Dict[str, Any]:
    errors = []
    payload = None
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
    if not isinstance(payload, dict):
        if payload is not None:
            errors.append("overall summary must be a JSON object")
    else:
        if payload.get("error"):
            errors.append(f"aggregation error: {payload['error']}")
        if require_sample_count and payload.get("sample_count") != expected_count:
            errors.append(
                "overall sample_count mismatch: "
                f"expected {expected_count}, got {payload.get('sample_count')}"
            )
        elif payload.get("sample_count") is not None and payload.get(
            "sample_count"
        ) != expected_count:
            errors.append(
                "overall sample_count mismatch: "
                f"expected {expected_count}, got {payload.get('sample_count')}"
            )
        for key in required_keys:
            if key not in payload:
                errors.append(f"overall summary missing required key: {key}")
        failure_rate = payload.get("failure_rate")
        if failure_rate not in (None, 0, 0.0):
            errors.append(f"overall failure_rate is non-zero: {failure_rate}")
    return {"overall_path": str(path), "errors": errors, "error_count": len(errors)}


def validate_event_completion(
    event_path: Path,
    expected_count: int,
    required_types: Iterable[str] = DEFAULT_REQUIRED_TYPES,
    expected_start: int = 0,
) -> Dict[str, Any]:
    required = tuple(required_types)
    counts = Counter()
    error_ids = []
    malformed = []
    with open(event_path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                event_id = int(event["id"])
                event_type = str(event["type"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                malformed.append({"line": line_no, "error": str(exc)})
                continue
            counts[(event_id, event_type)] += 1
            if event_type == "error":
                error_ids.append(event_id)

    expected_ids = set(range(expected_start, expected_start + expected_count))
    observed_ids = {event_id for event_id, _ in counts}
    missing = {
        event_type: sorted(
            event_id
            for event_id in expected_ids
            if counts[(event_id, event_type)] == 0
        )
        for event_type in required
    }
    duplicates = [
        {"id": event_id, "type": event_type, "count": count}
        for (event_id, event_type), count in sorted(counts.items())
        if event_type in required and count > 1
    ]
    unexpected_ids = sorted(observed_ids - expected_ids)
    error_count = (
        len(malformed)
        + len(set(error_ids))
        + sum(len(ids) for ids in missing.values())
        + len(duplicates)
        + len(unexpected_ids)
    )
    return {
        "event_path": str(event_path),
        "expected_count": expected_count,
        "expected_start": expected_start,
        "observed_id_count": len(observed_ids & expected_ids),
        "required_types": list(required),
        "error_event_count": len(error_ids),
        "missing_counts": {key: len(value) for key, value in missing.items()},
        "missing": {key: value[:20] for key, value in missing.items()},
        "duplicate_count": len(duplicates),
        "duplicates": duplicates[:20],
        "unexpected_ids": unexpected_ids[:20],
        "malformed": malformed[:20],
        "error_count": error_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="校验 event JSONL 是否逐样本完整")
    parser.add_argument("event_path", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--expected-count", type=int)
    source.add_argument("--dataset")
    parser.add_argument("--registry-path", default="mesh_eval/registry")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--expected-start", type=int, default=0)
    parser.add_argument("--overall", type=Path)
    parser.add_argument("--required-overall-keys", nargs="+", default=[])
    parser.add_argument(
        "--allow-missing-overall-sample-count", action="store_true"
    )
    parser.add_argument(
        "--required-types", nargs="+", default=list(DEFAULT_REQUIRED_TYPES)
    )
    args = parser.parse_args()

    if args.expected_count is not None:
        expected_count = args.expected_count
    elif args.manifest is not None:
        expected_count = count_manifest_rows(args.manifest)
        if args.limit > 0:
            expected_count = min(expected_count, args.limit)
    else:
        expected_count = count_registry_dataset(
            args.dataset, registry_paths=args.registry_path, limit=args.limit
        )
    report = validate_event_completion(
        args.event_path,
        expected_count,
        required_types=args.required_types,
        expected_start=args.expected_start,
    )
    if args.manifest:
        report["manifest"] = str(args.manifest)
    if args.dataset:
        report["dataset"] = args.dataset
    if args.overall:
        overall_report = validate_overall_summary(
            args.overall,
            expected_count,
            required_keys=args.required_overall_keys,
            require_sample_count=not args.allow_missing_overall_sample_count,
        )
        report["overall"] = overall_report
        report["error_count"] += overall_report["error_count"]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
