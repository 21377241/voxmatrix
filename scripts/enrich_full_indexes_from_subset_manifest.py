#!/usr/bin/env python3
"""Backfill language / scenario / metadata onto Full (+ Native) indexes.

Preferred API for builders: ``mesh_eval.data.full_index_meta.finalize_jsonl``.
This CLI rewrites every slot in ``slot_map.yaml`` (or ``--only``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mesh_eval.data.full_index_meta import (  # noqa: E402
    finalize_jsonl,
    resolve_record_id,
)

SLOT_MAP = ROOT / "mesh_eval" / "data" / "full" / "slot_map.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated slot ids, e.g. asr__aishell_1,qa__vocalbench",
    )
    args = parser.parse_args()
    only = {part.strip() for part in args.only.split(",") if part.strip()}
    slot_map = yaml.safe_load(SLOT_MAP.read_text(encoding="utf-8")) or {}

    print("=== enrich full/native from subset manifest ===")
    for slot_id, entry in sorted(slot_map.items()):
        if only and slot_id not in only:
            continue
        record_id = resolve_record_id(slot_id)
        if not record_id:
            print(f"{slot_id:48s} skip   no subset_record_id")
            continue
        for kind in ("full", "native"):
            rel = entry.get(kind)
            if not rel:
                continue
            path = ROOT / rel
            if not path.is_file():
                print(f"{slot_id:48s} {kind:6s} missing {rel}")
                continue
            if args.dry_run:
                print(f"{slot_id:48s} {kind:6s} dry-run {rel}")
                continue
            stats = finalize_jsonl(path, slot_id)
            print(f"{slot_id:48s} {kind:6s} rows={stats['rows']} -> {rel}")


if __name__ == "__main__":
    main()
