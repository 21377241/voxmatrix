#!/usr/bin/env python3
"""Mark the confirmed bad Common Voice clip and recompute its smoke metric."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAD_ID = "cv_en_common_voice_en_30406597"
FULL = ROOT / "mesh_eval/data/full/asr/asr_common_voice.jsonl"
NATIVE = ROOT / "mesh_eval/data/native/asr_common_voice.jsonl"
RESULT = ROOT / "outputs/native_asr_smoke_rebuilt/native-asr-common-voice-rebuilt.jsonl"
REPORT = ROOT / "outputs/native_asr_smoke_rebuilt/common-voice-recomputed.json"


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text or "").lower())


def edits(ref: list[str], pred: list[str]) -> int:
    row = list(range(len(pred) + 1))
    for i, r in enumerate(ref, 1):
        nxt = [i]
        for j, p in enumerate(pred, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (r != p)))
        row = nxt
    return row[-1]


def main() -> int:
    rows = [json.loads(x) for x in FULL.read_text(encoding="utf-8").splitlines() if x.strip()]
    marked = False
    for row in rows:
        if row.get("sample_id") == BAD_ID:
            row["quality_status"] = "invalid_audio"
            row["quality_note"] = "audio unintelligible on human review"
            marked = True
    if not marked:
        raise SystemExit(f"sample not found: {BAD_ID}")
    FULL.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")

    # Rebuild the Native 10-sample index without the invalid row, preserving
    # the deterministic spread as much as possible.
    valid = [x for x in rows if x.get("quality_status") != "invalid_audio"]
    positions = [round(i * (len(valid) - 1) / 9) for i in range(10)]
    selected = [dict(valid[i]) for i in positions]
    NATIVE.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in selected), encoding="utf-8")

    evals = []
    for line in RESULT.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("type") == "eval" and item.get("data", {}).get("sample_id") != BAD_ID:
            evals.append(item["data"])
    total_edits = sum(edits(tokens(x.get("ref")), tokens(x.get("pred"))) for x in evals)
    total_ref = sum(len(tokens(x.get("ref"))) for x in evals)
    report = {
        "excluded_sample_id": BAD_ID,
        "reason": "invalid_audio",
        "valid_eval_samples": len(evals),
        "corpus_wer_percent": 100.0 * total_edits / total_ref if total_ref else None,
        "total_edits": total_edits,
        "total_reference_words": total_ref,
        "note": "Native replacement index is ready; rerun inference for replacement sample before publishing final metric.",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
