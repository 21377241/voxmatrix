#!/usr/bin/env python3
"""Offline-rescore smoke_attr results with attribution JSON salvage."""
from __future__ import annotations

import json
from pathlib import Path

from mesh_eval.evaluator.speaker import SpeakerAttributionEvaluator

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT / "results_rescored"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ev = SpeakerAttributionEvaluator()
    summary = {}
    for path in sorted(RESULTS.glob("*.jsonl")):
        name = path.stem
        if name in {"session"}:
            continue
        rows_out = []
        scores = []
        invalid_before = 0
        invalid_after = 0
        salvaged = 0
        for line in path.open(encoding="utf-8"):
            obj = json.loads(line)
            rows_out.append(obj)
            if obj.get("type") != "eval":
                continue
            data = obj["data"]
            pred = data.get("pred")
            ref = data.get("ref")
            old_valid = int(data.get("attribution_valid") or 0)
            if old_valid == 0:
                invalid_before += 1
            scored = ev(pred, ref, language=data.get("language") or "zh", reference=ref)
            data = dict(data)
            data.update(scored)
            obj = dict(obj)
            obj["data"] = data
            rows_out[-1] = obj
            scores.append(scored)
            if scored.get("attribution_valid", 0) == 0:
                invalid_after += 1
            salvaged += int(scored.get("attribution_salvaged") or 0)
        out_path = OUT / path.name
        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows_out:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if not scores:
            continue
        cpcer = sum(s["cpcer%"] for s in scores) / len(scores)
        acc = sum(s["attribution_acc"] for s in scores) / len(scores)
        valid = sum(int(s.get("attribution_valid") or 0) for s in scores) / len(scores)
        summary[name] = {
            "n": len(scores),
            "cpcer%_mean": cpcer,
            "attribution_acc_mean": acc,
            "valid_rate": valid,
            "invalid_before": invalid_before,
            "invalid_after": invalid_after,
            "salvaged_rows": salvaged,
        }
        print(
            f"{name}: cpCER {cpcer:.2f} acc {acc:.3f} valid {valid:.2f} "
            f"invalid {invalid_before}->{invalid_after} salvaged {salvaged}"
        )
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
