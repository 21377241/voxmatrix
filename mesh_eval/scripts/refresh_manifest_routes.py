import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mesh_eval.core.benchmark_map import apply_benchmark_fallback, benchmark_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="按 benchmark map 刷新 Manifest 路由字段")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--benchmark-map", default="")
    parser.add_argument(
        "--overwrite-routes",
        action="store_true",
        help="Replace prompt/metric/evaluator routing only; sample attributes are never overwritten.",
    )
    args = parser.parse_args()

    output = args.output or args.input
    rows = []
    updated = 0
    with open(args.input, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            dataset = row.get("dataset_id") or row.get("dataset")
            profile = benchmark_profile(str(dataset), args.benchmark_map)
            row = apply_benchmark_fallback(row, profile, benchmark=str(dataset))
            if args.overwrite_routes:
                if row.get("schema_version") == "2.0":
                    row["evaluation_hints"] = {
                        "metrics": profile["metrics"],
                        "evaluators": profile.get("evaluators", []),
                        "prompt": profile.get("prompt", ""),
                        "answer_type": profile.get("answer_type", ""),
                        "source": "benchmark_map_override",
                    }
                else:
                    row["metrics"] = profile["metrics"]
                    row["evaluators"] = profile.get("evaluators", [])
                    row["prompt_name"] = profile.get("prompt", "")
                    row.setdefault("route_provenance", {})[
                        "source"
                    ] = "benchmark_map"
            row["language"] = row.get("language") or row.get("metadata", {}).get(
                "language", ""
            )
            for key, value in (row.get("condition") or {}).items():
                row.setdefault(f"condition__{key}", value)
            rows.append(row)
            updated += 1

    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with open(temporary, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"input": str(args.input), "output": str(output), "updated": updated}, indent=2))


if __name__ == "__main__":
    main()
