import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_jsonl")
    parser.add_argument("--max_rows", type=int, default=20)
    return parser.parse_args()


def main():
    args = get_args()
    events = defaultdict(dict)
    with open(args.result_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            events[event["id"]][event["type"]] = event.get("data", {})

    for index in sorted(events)[: args.max_rows]:
        row = events[index]
        inference = row.get("inference", {}).get("content", "")
        eval_data = row.get("eval", {})
        metrics = {
            key: value
            for key, value in eval_data.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        dims = {
            key: eval_data.get(key)
            for key in ("sample_id", "dataset_id", "task", "capability", "scenario", "language", "mesh_evaluator")
            if key in eval_data
        }
        print(
            json.dumps(
                {
                    "id": index,
                    "dims": dims,
                    "metrics": metrics,
                    "inference": str(inference)[:300],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
