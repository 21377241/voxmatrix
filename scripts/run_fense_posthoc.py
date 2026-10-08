#!/usr/bin/env python3
"""Compute FENSE from saved Audio Caption predictions without model inference."""

from __future__ import annotations

import json
import os
from pathlib import Path

from audio_evals.evaluator.fense import FENSE


ROOT = Path(__file__).resolve().parents[1]
if not os.environ.get("NATIVE_OUTPUT_DIR"):
    raise SystemExit("set NATIVE_OUTPUT_DIR to the native run dir holding AudioCaps predictions")
RUN_DIR = Path(os.environ["NATIVE_OUTPUT_DIR"])
INPUTS = {
    "audiocaps": RUN_DIR / "main/native-audio-caption--audiocaps.jsonl",
    "clotho": ROOT / "outputs/full_audio_caption_smoke/full-audio-caption-clotho-smoke.jsonl",
}
OUTPUT = RUN_DIR / "judge/fense-posthoc.json"


def load(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("type") != "eval":
            continue
        data = item["data"]
        rows.append((str(data["pred"]), data.get("caption_references") or [data.get("ref", "")]))
    return rows


def main() -> int:
    metric = FENSE(
        sbert_model="paraphrase-TinyBERT-L6-v2",
        echecker_model="echecker_clotho_audiocaps_base",
        penalty=0.9,
        error_threshold=0.9,
        device="cpu",
    )
    report = {"source": "saved_predictions", "benchmarks": {}}
    for name, path in INPUTS.items():
        rows = load(path)
        scores = metric._metric.corpus_score([p for p, _ in rows], [r for _, r in rows], agg_score="none")
        report["benchmarks"][name] = {
            "sample_count": len(scores),
            "FENSE": float(sum(scores) / len(scores)) if len(scores) else 0.0,
            "scores": [float(x) for x in scores],
            "input": str(path),
        }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
