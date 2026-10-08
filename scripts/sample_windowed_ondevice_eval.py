#!/usr/bin/env python3
"""Sample the formal windowed on-device eval set from /mnt/afs/eval_data/windowed_ondevice.

Per dataset, picks N crop families that are complete (S/M/L/XL) for every
capability, so each bucket scores the same audio across long_form_asr,
diarization and speaker_attribution. Families are drawn round-robin over
seeded-shuffled meetings; rows are written family-major in pick order, so
NATIVE_LIMIT=4*k yields k complete families from k distinct meetings.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WOD = Path("/mnt/afs/eval_data/windowed_ondevice")
NATIVE = ROOT / "mesh_eval" / "data" / "native"
SUBSET_REPORT = WOD / "summaries" / "eval_subset.json"

DATASETS = ("aishell_4", "alimeeting", "ami", "chime_6")
CAPS = ("long_form_asr", "diarization", "speaker_attribution")
BUCKETS = ("S", "M", "L", "XL")


def load_index(cap: str, dataset: str) -> dict[str, dict[str, dict]]:
    by_family: dict[str, dict[str, dict]] = defaultdict(dict)
    for bucket in BUCKETS:
        path = WOD / "indexes" / cap / f"{dataset}_{bucket}.jsonl"
        for line in path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                by_family[row["crop_family_id"]][bucket] = row
    return by_family


def pick_families(meeting_of: dict[str, str], n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    by_meeting: dict[str, list[str]] = defaultdict(list)
    for fid in sorted(meeting_of):
        by_meeting[meeting_of[fid]].append(fid)
    meetings = sorted(by_meeting)
    rng.shuffle(meetings)
    for meeting in meetings:
        rng.shuffle(by_meeting[meeting])
    picked: list[str] = []
    while len(picked) < n and any(by_meeting.values()):
        for meeting in meetings:
            if by_meeting[meeting] and len(picked) < n:
                picked.append(by_meeting[meeting].pop())
    return picked


def out_path(dataset: str, cap: str) -> Path:
    if cap == "long_form_asr":
        return NATIVE / f"{dataset}_long_form_windowed.jsonl"
    return NATIVE / f"{cap}_{dataset}_windowed.jsonl"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-bucket", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", action="append", choices=list(DATASETS))
    args = parser.parse_args()

    report = json.loads(SUBSET_REPORT.read_text()) if SUBSET_REPORT.is_file() else {}
    for ds in args.dataset or DATASETS:
        indexes = {cap: load_index(cap, ds) for cap in CAPS}
        complete = set.intersection(
            *(
                {fid for fid, rows in idx.items() if len(rows) == len(BUCKETS)}
                for idx in indexes.values()
            )
        )
        asr = indexes["long_form_asr"]
        meeting_of = {fid: asr[fid]["S"]["meeting_id"] for fid in complete}
        picked = pick_families(meeting_of, args.per_bucket, args.seed)
        if len(picked) < args.per_bucket:
            print(f"[warn] {ds}: only {len(picked)} complete families (< {args.per_bucket})")
        for cap in CAPS:
            rows = [indexes[cap][fid][b] for fid in picked for b in BUCKETS]
            write_jsonl(out_path(ds, cap), rows)
            print(f"[ok] {ds} {cap}: {len(rows)} rows -> {out_path(ds, cap)}")
        report[ds] = {
            "per_bucket": len(picked),
            "seed": args.seed,
            "complete_families": len(complete),
            "meetings": len({meeting_of[f] for f in picked}),
            "families": picked,
        }
    SUBSET_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"[ok] subset report -> {SUBSET_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
