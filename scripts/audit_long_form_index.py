#!/usr/bin/env python3
"""Audit long-form ASR window indexes and their timeline provenance."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "mesh_eval" / "data" / "native"
TIMELINES = NATIVE / "long_form_timelines"
TIMELINE_DATASETS = {"aishell_4", "alimeeting", "misp", "sbcsae"}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def audit(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    dataset = rows[0].get("dataset") if rows else path.stem
    errors = []
    ids = set()
    timeline_rows = defaultdict(list)
    timeline_path = TIMELINES / f"{dataset}.jsonl"
    if dataset in TIMELINE_DATASETS and timeline_path.is_file():
        for line in timeline_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                timeline_rows[str(item.get("src_audio"))].append(item)

    for line_no, row in enumerate(rows, 1):
        sid = row.get("sample_id")
        if sid in ids:
            errors.append(f"line {line_no}: duplicate sample_id={sid}")
        ids.add(sid)
        wav = Path(row.get("WavPath", ""))
        if not wav.is_file():
            errors.append(f"line {line_no}: missing WavPath={wav}")
        start, end = row.get("start_time"), row.get("end_time")
        duration = row.get("audio_duration_seconds")
        if dataset in {"ami", "chime_6"}:
            # These windows are assembled from timestamped utterance files;
            # the final manifest intentionally stores source_group instead of
            # a single parent-recording time range.
            start = end = None
        elif start is None or end is None or end <= start:
            errors.append(f"line {line_no}: invalid time range")
        if duration is None or not (30.0 <= float(duration) <= 60.0):
            errors.append(f"line {line_no}: duration outside 30-60: {duration}")
        if not norm(row.get("text")):
            errors.append(f"line {line_no}: empty reference")
        if any(marker in str(wav) for marker in ("utt_clips", "smoke_", "8p31/runs")):
            errors.append(f"line {line_no}: forbidden path marker in WavPath")

        src = str(row.get("src_audio", ""))
        candidates = []
        if start is not None and end is not None:
            # The builder packs the selected consecutive rows whose boundaries
            # define the window.  Use containment rather than overlap: nearby
            # overlapping annotations may belong to the next/previous pack.
            candidates = [
                x for x in timeline_rows.get(src, [])
                if x.get("start_time", 0) >= start - 1e-6
                and x.get("end_time", 0) <= end + 1e-6
            ]
        # Match pack_session_rows: stable sort by start_time only.  A
        # secondary end-time sort can reorder simultaneous speakers and create
        # a false reference mismatch.
        candidates.sort(key=lambda x: x.get("start_time", 0))
        if timeline_rows and start is not None and not candidates:
            errors.append(f"line {line_no}: no timeline segments overlap window")
        if candidates:
            expected = norm(" ".join(norm(x.get("text")) for x in candidates if norm(x.get("text"))))
            if expected != norm(row.get("text")):
                errors.append(f"line {line_no}: reference mismatch expected_len={len(expected)} actual_len={len(norm(row.get('text')))}")
            if row.get("n_utterances") != len(candidates):
                errors.append(f"line {line_no}: n_utterances={row.get('n_utterances')} overlap_segments={len(candidates)}")

    return {"file": str(path), "dataset": dataset, "records": len(rows), "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.paths or sorted(NATIVE.glob("*_long_form_30_60.jsonl"))
    report = [audit(path) for path in paths]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if any(item["errors"] for item in report) else 0


if __name__ == "__main__":
    raise SystemExit(main())
