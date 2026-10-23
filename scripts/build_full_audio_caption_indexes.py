#!/usr/bin/env python3
"""Build full AudioCaps and Clotho caption indexes from source manifests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "mesh_eval" / "data" / "full" / "audio_caption"
sys.path.insert(0, str(ROOT))

from suites.native import build_native_capability_samples as native  # noqa: E402


def main() -> int:
    native.LIMIT = 10**9
    native.register()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for benchmark in ("audiocaps", "clotho"):
        rows = native.BUILDERS[("audio_caption", benchmark)]()
        if benchmark == "clotho":
            grouped = {}
            for row in rows:
                key = row["WavPath"]
                item = grouped.setdefault(key, dict(row))
                item.setdefault("captions", []).append(row["text"])
            rows = list(grouped.values())
            for row in rows:
                row["text"] = row["captions"][0]
        path = OUT / f"audio_caption_{benchmark}.jsonl"
        from mesh_eval.data.full_index_meta import write_index_jsonl

        write_index_jsonl(path, rows, f"audio_caption__{benchmark}")
        summary[benchmark] = {"records": len(rows), "output": str(path)}
        print(f"[done] {benchmark}: {len(rows)} -> {path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
