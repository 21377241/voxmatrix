#!/usr/bin/env python3
"""Sync the full S/M/L/XL meeting-window indexes into mesh_eval/data/full.

Source: /mnt/afs/eval_data/windowed_ondevice/indexes/{capability}/{dataset}_{bucket}.jsonl
(built by scripts/build_windowed_ondevice_indexes.py). Every available row is
kept, including families missing a bucket, because each bucket is scored as
its own slot. Rows are written family-major (S, M, L, XL within a family) with
the same file names and fields as the Native windowed indexes.

The retired 30-60s long_form indexes and their crops are moved to
mesh_eval/data/full/long_form_asr/legacy_30_60/ with WavPath rewritten.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WOD = Path("/mnt/afs/eval_data/windowed_ondevice")
FULL = ROOT / "mesh_eval" / "data" / "full"

DATASETS = ("aishell_4", "alimeeting", "ami", "chime_6")
CAPS = ("long_form_asr", "diarization", "speaker_attribution")
BUCKETS = ("S", "M", "L", "XL")

LEGACY_FILES = ("aishell_4_long_form_30_60.jsonl", "alimeeting_long_form_30_60.jsonl")


def out_path(cap: str, dataset: str) -> Path:
    if cap == "long_form_asr":
        return FULL / cap / f"{dataset}_long_form_windowed.jsonl"
    return FULL / cap / f"{cap}_{dataset}_windowed.jsonl"


def load_rows(cap: str, dataset: str) -> list[dict]:
    families: dict[str, dict[str, dict]] = {}
    for bucket in BUCKETS:
        path = WOD / "indexes" / cap / f"{dataset}_{bucket}.jsonl"
        for line in path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                families.setdefault(row["crop_family_id"], {})[bucket] = row
    order = sorted(
        families,
        key=lambda fid: (
            str(next(iter(families[fid].values())).get("meeting_id")),
            min(float(r.get("start_time") or 0.0) for r in families[fid].values()),
            fid,
        ),
    )
    return [families[fid][b] for fid in order for b in BUCKETS if b in families[fid]]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def retire_legacy() -> list[str]:
    src_dir = FULL / "long_form_asr"
    dst_dir = src_dir / "legacy_30_60"
    moved: list[str] = []
    old_prefix = str(src_dir / "windows") + "/"
    new_prefix = str(dst_dir / "windows") + "/"
    windows = src_dir / "windows"
    if windows.is_dir():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(windows), str(dst_dir / "windows"))
        moved.append("windows/")
    for name in LEGACY_FILES:
        src = src_dir / name
        if not src.is_file():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        rows = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
        for row in rows:
            wav = str(row.get("WavPath") or "")
            if wav.startswith(old_prefix):
                row["WavPath"] = new_prefix + wav[len(old_prefix):]
        write_jsonl(dst_dir / name, rows)
        src.unlink()
        moved.append(name)
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=list(DATASETS))
    parser.add_argument("--keep-legacy", action="store_true", help="do not move the 30-60s indexes")
    parser.add_argument("--skip-audio-check", action="store_true")
    args = parser.parse_args()

    summary: dict[str, dict[str, dict]] = {}
    missing_audio: list[str] = []
    for cap in CAPS:
        for ds in args.dataset or DATASETS:
            rows = load_rows(cap, ds)
            if not args.skip_audio_check:
                missing_audio += [r["sample_id"] for r in rows if not Path(r["WavPath"]).is_file()]
            path = out_path(cap, ds)
            write_jsonl(path, rows)
            from mesh_eval.data.full_index_meta import finalize_jsonl

            finalize_jsonl(path, f"{cap}__{ds}")
            counts = Counter(r["bucket"] for r in rows)
            summary.setdefault(cap, {})[ds] = {
                "rows": len(rows),
                "families": len({r["crop_family_id"] for r in rows}),
                **{b: counts.get(b, 0) for b in BUCKETS},
            }
            print(f"[ok] {cap} {ds}: {len(rows)} rows -> {out_path(cap, ds).relative_to(ROOT)}")

    if not args.keep_legacy:
        moved = retire_legacy()
        if moved:
            print(f"[ok] legacy 30-60s moved to long_form_asr/legacy_30_60/: {moved}")

    if missing_audio:
        print(f"[warn] {len(missing_audio)} rows point to missing audio, e.g. {missing_audio[:5]}")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 1 if missing_audio else 0


if __name__ == "__main__":
    raise SystemExit(main())
