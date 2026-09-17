#!/usr/bin/env python3
"""Prefill local ASR cache for a stratified judge pack (separate process).

Run this BEFORE loading Qwen3-Omni Thinking so Whisper/paraformer GPU memory
is fully released before the judge subprocess starts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--asr-cache", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    # Default CUDA for speed during dedicated prefill; runner clears cache before Omni.
    os.environ.setdefault(
        "JUDGE_ASR_DEVICE", os.environ.get("JUDGE_ASR_DEVICE", "cuda")
    )
    os.environ.setdefault("JUDGE_ASR_BACKEND", os.environ.get("JUDGE_ASR_BACKEND", "local"))

    from audio_evals.evaluator.asr_for_judge import transcribe_for_judge

    rows = [
        json.loads(line)
        for line in args.pack.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    args.asr_cache.mkdir(parents=True, exist_ok=True)
    ok = fail = skip = 0
    for i, sample in enumerate(rows, 1):
        wav = sample.get("WavPath") or ""
        if not wav or not Path(wav).is_file():
            skip += 1
            print(f"[{i}/{len(rows)}] SKIP missing wav sample_id={sample.get('sample_id')}")
            continue
        try:
            text = transcribe_for_judge(
                wav,
                lang=sample.get("lang"),
                cache_dir=str(args.asr_cache),
            )
            if text.strip():
                ok += 1
                print(
                    f"[{i}/{len(rows)}] OK id={sample.get('sample_id')} "
                    f"chars={len(text)} lang={sample.get('lang')}",
                    flush=True,
                )
            else:
                fail += 1
                print(
                    f"[{i}/{len(rows)}] EMPTY id={sample.get('sample_id')} lang={sample.get('lang')}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(
                f"[{i}/{len(rows)}] FAIL id={sample.get('sample_id')} err={exc}",
                flush=True,
            )

    print(f"[prefill-asr] done ok={ok} empty_or_fail={fail} skip={skip} cache={args.asr_cache}")
    return 0 if fail == 0 else 0  # soft: judge will mark bad_audio_transcript


if __name__ == "__main__":
    raise SystemExit(main())
