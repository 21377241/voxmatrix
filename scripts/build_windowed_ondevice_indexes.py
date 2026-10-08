#!/usr/bin/env python3
"""CLI: build windowed on-device S/M/L/XL indexes under eval_data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from suites.native.windowed_ondevice import (  # noqa: E402
    DEFAULT_OUT,
    LOADERS,
    build_all,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=sorted(LOADERS.keys()),
        help="Repeatable; default = all",
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--family-gap", type=float, default=150.0)
    parser.add_argument("--max-families-per-meeting", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = build_all(
        datasets=args.benchmark,
        out_root=args.out_root,
        family_gap=args.family_gap,
        max_families=args.max_families_per_meeting,
        seed=args.seed,
    )
    print(json.dumps({"ok": True, "summary": str(args.out_root / "summaries" / "build_report.json")}, ensure_ascii=False))
    print(json.dumps({k: v.get("families") for k, v in report["datasets"].items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
