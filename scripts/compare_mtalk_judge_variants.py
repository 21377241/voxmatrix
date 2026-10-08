#!/usr/bin/env python3
"""Judge-only comparison of MTalk judge output rules on saved SUT outputs.

Variants (single attempt each; temperature 0 makes retries identical):
  soft          current one-line constraint
  strict        single-turn judge rules (_OUTPUT_THINKING_RULES + JSON line)
  strict_force  strict + force-close thinking at the token cap (512 tail tokens)

Usage (GPU node):
  python scripts/compare_mtalk_judge_variants.py --run-dir <native run dir> [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_evals.evaluator.voice_bench import get_judge_model  # noqa: E402
from mesh_eval.dataset.external_benchmarks import MTalkBenchDataset  # noqa: E402
from mesh_eval.evaluator.dialogue import MultiTurnDialogueEvaluator  # noqa: E402

VARIANTS = {
    "soft": {"strict_output_rules": False, "force_close_thinking_tokens": 0},
    "strict": {"strict_output_rules": True, "force_close_thinking_tokens": 0},
    "strict_force": {"strict_output_rules": True, "force_close_thinking_tokens": 512},
}
FORCE_MARK = "Thinking budget exhausted"


def load_predictions(run_dir: Path) -> dict[str, str]:
    path = run_dir / "tts" / "native-multi-turn-dialogue--mtalk-bench.jsonl"
    preds = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("type") == "inference":
            preds[row["sample_id"]] = row["data"]["content"]
    return preds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    args = parser.parse_args()

    out = args.out or ROOT / "outputs" / "mtalk_judge_compare" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    preds = load_predictions(args.run_dir)
    samples = [r for r in MTalkBenchDataset(benchmark_id="t4-mtalk-bench").load(len(preds)) if r["sample_id"] in preds]
    model = get_judge_model("qwen3-omni-thinking")
    print(f"samples={len(samples)} out={out}", flush=True)

    summary = {}
    with (out / "results.jsonl").open("w", encoding="utf-8") as handle:
        for name in args.variants.split(","):
            calls: list[dict] = []

            def judge(messages, **options):
                start = time.monotonic()
                raw = model.inference(messages, **options)
                raw = raw.get("text", "") if isinstance(raw, dict) else str(raw or "")
                calls.append(
                    {
                        "seconds": round(time.monotonic() - start, 1),
                        "chars": len(raw),
                        "forced": FORCE_MARK in raw,
                        "closed_naturally": "</think>" in raw and FORCE_MARK not in raw,
                        "raw_tail": raw[-1500:],
                    }
                )
                return raw

            evaluator = MultiTurnDialogueEvaluator(
                "mtalk_rubric",
                judge=judge,
                max_attempts=1,
                work_dir=str(out),
                **VARIANTS[name],
            )
            scored = []
            for sample in samples:
                before = len(calls)
                result = evaluator(preds[sample["sample_id"]], sample["reference_obj"])
                record = {
                    "variant": name,
                    "sample_id": sample["sample_id"],
                    "status": result.get("status"),
                    "failed_rubric": result.get("failed_rubric"),
                    "rubric_general": result.get("rubric_general"),
                    "rubric_specific": result.get("rubric_specific"),
                    "rubric_score": result.get("rubric_score"),
                    "criteria_general": result.get("criteria_general"),
                    "criteria_specific": result.get("criteria_specific"),
                    "calls": calls[before:],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                scored.append(record)
                print(
                    f"[{name}] {sample['sample_id']} status={record['status']} "
                    f"score={record['rubric_score']} calls="
                    + ",".join(
                        f"{c['seconds']}s/{'forced' if c['forced'] else 'closed' if c['closed_naturally'] else 'cut'}"
                        for c in record["calls"]
                    ),
                    flush=True,
                )
            ok = [r for r in scored if r["rubric_score"] is not None]
            summary[name] = {
                "samples": len(scored),
                "scored": len(ok),
                "mean_rubric_score": round(sum(r["rubric_score"] for r in ok) / len(ok), 4) if ok else None,
                "judge_calls": len(calls),
                "closed_naturally": sum(c["closed_naturally"] for c in calls),
                "forced": sum(c["forced"] for c in calls),
                "cut_without_answer": sum(not c["closed_naturally"] and not c["forced"] for c in calls),
                "mean_call_seconds": round(sum(c["seconds"] for c in calls) / len(calls), 1) if calls else None,
            }
            print(f"[{name}] summary {json.dumps(summary[name])}", flush=True)

    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
