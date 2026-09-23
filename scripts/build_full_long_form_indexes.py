#!/usr/bin/env python3
"""Build full long-form ASR indexes in an isolated full-data directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from suites.native import build_long_form_windows as native_long  # noqa: E402


FULL_ROOT = ROOT / "mesh_eval" / "data" / "full" / "long_form_asr"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="append", choices=["aishell_4", "alimeeting", "ami", "chime_6", "misp", "sbcsae"])
    args = parser.parse_args()

    # Reuse the audited L0/timeline/window implementation, but redirect all
    # generated clips and manifests away from the Native 10-sample directory.
    native_long.NATIVE = FULL_ROOT
    native_long.AUDIO_ROOT = FULL_ROOT / "windows"
    native_long.LIMIT = 10**9
    native_long.AUDIO_ROOT.mkdir(parents=True, exist_ok=True)

    selected = set(args.benchmark or [])
    native_long.main(selected or None)

    summary = {}
    for path in sorted(FULL_ROOT.glob("*_long_form_30_60.jsonl")):
        if selected and path.stem.split("_long_form_30_60")[0] not in selected:
            continue
        summary[path.stem] = {"records": sum(1 for line in path.open() if line.strip()), "path": str(path)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
