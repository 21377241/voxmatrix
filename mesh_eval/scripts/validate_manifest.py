import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mesh_eval.core.schema import normalize_record, validate_record


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--ref_col", default="text")
    parser.add_argument("--check_audio_exists", action="store_true")
    parser.add_argument("--cwd", default=".")
    return parser.parse_args()


def main():
    args = get_args()
    total = 0
    error_rows = 0
    error_counter = Counter()
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            record = normalize_record(json.loads(line), ref_col=args.ref_col)
            errors = validate_record(
                record,
                ref_col=args.ref_col,
                check_audio_exists=args.check_audio_exists,
                cwd=args.cwd,
            )
            if errors:
                error_rows += 1
                for error in errors:
                    error_counter[error] += 1
                print(f"{args.manifest}:{line_no}: {errors}")

    print(
        json.dumps(
            {
                "manifest": args.manifest,
                "total": total,
                "error_rows": error_rows,
                "errors": dict(error_counter),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if error_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
