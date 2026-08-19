import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mesh_eval.core.benchmark_map import benchmark_profile


def read_events(path: Path) -> Dict[int, Dict[str, Any]]:
    events: Dict[int, Dict[str, Any]] = defaultdict(dict)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            events[int(event["id"])][event["type"]] = event.get("data") or {}
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="把已录制推理挂回统一 Manifest")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--benchmark-map", default="")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    event_rows = read_events(args.events)
    output_rows = []
    missing = []
    record_index = 0
    with open(args.manifest, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            index = record_index
            record_index += 1
            row = json.loads(line)
            event = event_rows.get(index, {})
            inference = (event.get("inference") or {}).get("content")
            processed = (event.get("post_process") or {}).get("content", inference)
            if inference in (None, ""):
                missing.append(index)
                if not args.allow_missing:
                    continue
            profile = benchmark_profile(
                str(row.get("dataset_id") or row.get("dataset")), args.benchmark_map
            )
            row["evaluators"] = profile.get("evaluators", [])
            row["prompt_name"] = profile.get("prompt", row.get("prompt_name", ""))
            row["eval_info"] = {
                "inference": {"content": inference},
                "post_process": {"content": processed},
            }
            output_rows.append(row)

    if missing and not args.allow_missing:
        raise SystemExit(
            f"missing inference for {len(missing)} rows; first indexes={missing[:20]}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "events": str(args.events),
                "output": str(args.output),
                "attached": len(output_rows),
                "missing": len(missing),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
