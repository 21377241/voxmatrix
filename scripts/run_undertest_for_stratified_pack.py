#!/usr/bin/env python3
"""Run undertest (Qwen3-Omni-Instruct) on stratified pack rows missing pred.

Requires CUDA. Writes results JSONL compatible with join script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pack",
        type=Path,
        default=ROOT / "scripts/data/semantic_judge_stratified_pack.need_pred.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "scripts/data/semantic_judge_undertest_preds.jsonl",
    )
    parser.add_argument("--model", default="qwen3-omni-audio")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("CUDA unavailable; cannot run undertest Omni.", file=sys.stderr)
        return 3

    from audio_evals.registry import registry

    rows = [json.loads(l) for l in args.pack.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    done = set()
    if args.skip_existing and args.out.is_file():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line).get("sample_id"))

    model = registry.get_model(args.model)
    if model is None:
        print(f"model not registered: {args.model}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.skip_existing and args.out.is_file() else "w"
    with args.out.open(mode, encoding="utf-8") as handle:
        for row in rows:
            sid = row.get("sample_id")
            if sid in done:
                continue
            wav = row.get("WavPath")
            question = row.get("question") or row.get("direction") or "Listen and respond."
            prompt = [
                {
                    "role": "user",
                    "contents": [
                        {"type": "audio", "value": wav},
                        {"type": "text", "value": str(question)},
                    ],
                }
            ]
            try:
                pred = model.inference(prompt)
            except Exception as exc:  # noqa: BLE001
                handle.write(
                    json.dumps(
                        {
                            "sample_id": sid,
                            "pred": "",
                            "error": str(exc)[:500],
                            "capability": row.get("capability"),
                            "rubric": row.get("rubric"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.flush()
                print(f"FAIL {sid}: {exc}", flush=True)
                continue
            out = {
                "sample_id": sid,
                "pred": "" if pred is None else str(pred),
                "capability": row.get("capability"),
                "rubric": row.get("rubric"),
                "WavPath": wav,
                "model": args.model,
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"OK {sid} pred_len={len(out['pred'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
