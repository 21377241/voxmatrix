#!/usr/bin/env python3
"""Merge sharded A1/B compare JSON outputs into one formal payload + report."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_compare_mod():
    path = Path(__file__).with_name("compare_semantic_llm_judge_tracks.py")
    spec = importlib.util.spec_from_file_location("compare_a1b", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--pack", type=Path, default=None)
    args = parser.parse_args()

    cmp = _load_compare_mod()

    rows: List[Dict[str, Any]] = []
    seen = set()
    judge_model = "qwen3-omni-thinking"
    probe: List[Dict[str, Any]] = []
    pack_path = str(args.pack or "")

    for path in args.shards:
        if not path.is_file():
            print(f"missing shard {path}", file=sys.stderr)
            return 2
        payload = json.loads(path.read_text(encoding="utf-8"))
        judge_model = payload.get("judge_model") or judge_model
        if payload.get("probe"):
            probe.extend(payload["probe"])
        if payload.get("pack"):
            pack_path = payload["pack"]
        for row in payload.get("rows") or []:
            key = (str(row.get("sample_id")), str(row.get("track")))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

    rows.sort(key=lambda r: (str(r.get("sample_id")), str(r.get("track"))))
    summary = cmp.summarize(rows)
    cell_pack = {}
    if args.pack and args.pack.is_file():
        pack = [
            json.loads(l)
            for l in args.pack.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        cell_pack = cmp.cell_counts(pack)
    out_payload = {
        "pack": pack_path,
        "n": len({r.get("sample_id") for r in rows}),
        "cell_counts": cell_pack,
        "judge_model": judge_model,
        "probe": probe,
        "summary": summary,
        "rows": rows,
        "merged_shards": [str(p) for p in args.shards],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Auto draft only; write_markdown refuses locked formal case-study report.
    cmp.write_markdown(
        summary,
        probe,
        args.report_md,
        judge_model=judge_model,
        cell_pack=cell_pack,
    )
    print(
        f"merged n_rows={len(rows)} n_paired={summary.get('n_paired')} "
        f"-> {args.out} report={args.report_md}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
