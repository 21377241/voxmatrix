#!/usr/bin/env python3
"""Compare EasyCom pred/ref coverage across result jsonl files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def analyze(path: Path) -> dict:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("type") != "eval":
                continue
            data = event["data"]
            pred = data.get("pred") or ""
            ref = data.get("ref") or ""
            pred_w = len(pred.split())
            ref_w = len(ref.split())
            rows.append(
                {
                    "id": event.get("id"),
                    "pred_w": pred_w,
                    "ref_w": ref_w,
                    "ratio": pred_w / max(ref_w, 1),
                    "wer": data.get("wer%"),
                }
            )
    if not rows:
        return {"path": str(path), "n": 0}
    return {
        "path": str(path),
        "n": len(rows),
        "avg_ratio": sum(r["ratio"] for r in rows) / len(rows),
        "avg_wer": sum(r["wer"] for r in rows) / len(rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.jsonl:
        summary = analyze(path)
        label = path.parent.name if path.name == "easycom.jsonl" else path.name
        print(f"\n=== {label} ===")
        if summary["n"] == 0:
            print("no eval rows")
            continue
        print(
            f"n={summary['n']} avg_pred/ref={summary['avg_ratio']:.3f} avg_wer={summary['avg_wer']:.1f}%"
        )
        for row in summary["rows"]:
            print(
                f"  id={row['id']:2d} pred={row['pred_w']:4d} ref={row['ref_w']:4d} "
                f"ratio={row['ratio']:.2f} wer={row['wer']:.1f}%"
            )


if __name__ == "__main__":
    main()
