#!/usr/bin/env python3
"""Build full ASR indexes from the existing L0-oriented ASR builders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "mesh_eval" / "data" / "full" / "asr"
UTT_CLIPS = Path("/mnt/afs/eval_data/utt_clips")
sys.path.insert(0, str(ROOT))

from suites.native import build_asr_native_samples as native_asr  # noqa: E402
BUILDERS = native_asr.BUILDERS


def from_utt_clips(benchmark: str) -> list[dict] | None:
    """Use the validated full utterance index when one is available."""
    source = UTT_CLIPS / benchmark / "indexes" / "asr.test.jsonl"
    if not source.is_file():
        return None
    rows = []
    for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        row["schema_version"] = "full-index/1.0"
        row["benchmark"] = benchmark
        row["capability"] = "asr"
        row["source_index"] = str(source)
        row["source_line"] = line_no
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", dest="benchmarks", action="append", choices=sorted(BUILDERS))
    args = parser.parse_args()
    # A large limit means "all" while preserving the existing builder logic.
    full_limit = 10**9
    summary = {}
    for benchmark in (args.benchmarks or sorted(BUILDERS)):
        rows = from_utt_clips(benchmark)
        source_kind = "utt_clips" if rows is not None else "L0 builder"
        if rows is None:
            rows = BUILDERS[benchmark](full_limit)
        path = OUT / f"asr_{benchmark}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        summary[benchmark] = {"records": len(rows), "output": str(path)}
        print(f"[done] {benchmark}: {len(rows)} ({source_kind}) -> {path}")
        if not rows:
            raise SystemExit(f"empty full ASR index: {benchmark}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
