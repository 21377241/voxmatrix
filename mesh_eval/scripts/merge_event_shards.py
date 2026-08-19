import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audio_evals.registry import Registry
from mesh_eval.scripts.validate_event_completion import validate_event_completion


TYPE_ORDER = {
    "prompt": 0,
    "inference": 1,
    "post_process": 2,
    "eval": 3,
    "error": 4,
}


def load_events(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    events = []
    seen = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                event = json.loads(line)
                event_id = int(event["id"])
                event_type = str(event["type"])
                key = (event_id, event_type)
                if key in seen:
                    previous_path, previous_line = seen[key]
                    raise ValueError(
                        f"duplicate event {key}: {previous_path}:{previous_line} and "
                        f"{path}:{line_no}"
                    )
                seen[key] = (path, line_no)
                events.append(event)
    return sorted(
        events,
        key=lambda event: (
            int(event["id"]),
            TYPE_ORDER.get(str(event["type"]), len(TYPE_ORDER)),
        ),
    )


def write_events(events: Iterable[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_name(f".{output.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    os.replace(temp_path, output)


def aggregate_events(
    events: Iterable[Dict[str, Any]], agg_name: str, registry_paths: str
) -> Dict[str, Any]:
    paths = [REPO_ROOT / "registry"]
    paths.extend(Path(item) for item in registry_paths.split() if item)
    registry = Registry(paths)
    scores = []
    for event in events:
        if event["type"] == "eval":
            scores.append(dict(event.get("data") or {}))
        elif event["type"] == "error":
            score = dict(event.get("data") or {})
            score.setdefault("failure", 1)
            score.setdefault("timeout", 0)
            scores.append(score)
    summary = dict(registry.get_agg(agg_name)(scores))
    failure_count = sum(bool(score.get("failure", 0)) for score in scores)
    failure_rate = failure_count / len(scores) if scores else 0.0
    summary["sample_count"] = len(scores)
    summary["failure_rate"] = failure_rate
    summary["fail_rate(%d)"] = failure_rate * 100
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge disjoint event shards and recompute the global aggregate"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--agg", required=True)
    parser.add_argument("--overall-output", required=True, type=Path)
    parser.add_argument("--registry-path", default="mesh_eval/registry")
    args = parser.parse_args()

    events = load_events(args.inputs)
    write_events(events, args.output)
    report = validate_event_completion(args.output, args.expected_count)
    if report["error_count"]:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    summary = aggregate_events(events, args.agg, args.registry_path)
    args.overall_output.parent.mkdir(parents=True, exist_ok=True)
    with args.overall_output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"completion": report, "overall": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
