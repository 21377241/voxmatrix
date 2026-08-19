import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mesh_eval.agg.mesh import MeshAgg


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_jsonl")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main():
    args = get_args()
    scores = []
    with open(args.result_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("type") == "eval":
                scores.append(event.get("data") or {})
            elif event.get("type") == "error":
                error = dict(event.get("data") or {})
                error.setdefault("failure", 1)
                error.setdefault("timeout", 0)
                scores.append(error)

    result = MeshAgg()(scores)
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
