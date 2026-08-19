import argparse
import json
from pathlib import Path

from mesh_eval.scripts.merge_event_shards import load_events, write_events
from mesh_eval.scripts.validate_event_completion import validate_event_completion


def slice_event_range(
    source: Path,
    output: Path,
    start: int,
    stop: int,
    required_types,
):
    if start < 0 or stop <= start:
        raise ValueError("event range must satisfy 0 <= start < stop")
    events = [
        event
        for event in load_events([source])
        if start <= int(event["id"]) < stop
    ]
    write_events(events, output)
    return validate_event_completion(
        output,
        expected_count=stop - start,
        expected_start=start,
        required_types=required_types,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and validate an event ID range")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--stop", required=True, type=int)
    parser.add_argument(
        "--required-types", nargs="+", default=["prompt", "inference"]
    )
    args = parser.parse_args()

    report = slice_event_range(
        args.source,
        args.output,
        args.start,
        args.stop,
        args.required_types,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
