#!/usr/bin/env python3
"""Run local multimodal Code-Switch LLM Judge on Native smoke predictions.

Lexical metrics (ROUGE-L / qa-exist-match) stay on the suite; this track is the
semantic auxiliary for mixed-language QA where a single lexical score is weak.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from audio_evals.evaluator.asr_for_judge import transcribe_for_judge
from audio_evals.evaluator.semantic_llm_judge import SemanticLLMJudgeEvaluator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/native_code_switch_smoke"
ASR_CACHE = OUTPUT / "judge_asr_cache"

TRACKS = [
    {
        "result_glob": "native-code-switch-mmsu-smoke.jsonl",
        "index": ROOT / "mesh_eval/data/native/code_switch_mmsu.jsonl",
        "rubric": "semi-open",
    },
    {
        "result_glob": "native-code-switch-vocalbench-zh-smoke.jsonl",
        "index": ROOT / "mesh_eval/data/native/code_switch_vocalbench_zh.jsonl",
        "rubric": "binary",
    },
    {
        "result_glob": "native-code-switch-vocalbench-zh-open-ended-smoke.jsonl",
        "index": ROOT / "mesh_eval/data/native/code_switch_vocalbench_zh_open_ended.jsonl",
        "rubric": "open",
    },
]


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if x is not None and str(x).strip()]
        return " | ".join(parts)
    return str(value).strip()


def _load_index(path: Path) -> dict[str, dict]:
    samples: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        samples[row["sample_id"]] = row
    return samples


def _shard_slice(items: list, shard_index: int, num_shards: int) -> list:
    if num_shards <= 1:
        return items
    return [x for i, x in enumerate(items) if i % num_shards == shard_index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("JUDGE_SHARD_INDEX", "0")))
    parser.add_argument("--num-shards", type=int, default=int(os.environ.get("JUDGE_NUM_SHARDS", "1")))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(f"bad shard args: index={args.shard_index} num={args.num_shards}")

    print(f"[cs-judge] start ROOT={ROOT}", flush=True)
    print(f"[cs-judge] asr_cache={ASR_CACHE} shard={args.shard_index}/{args.num_shards}", flush=True)

    work: list[tuple[dict, dict, str]] = []
    missing_results = []
    for track in TRACKS:
        result_path = OUTPUT / track["result_glob"]
        if not result_path.is_file():
            missing_results.append(str(result_path))
            continue
        if not track["index"].is_file():
            raise SystemExit(f"missing Native index: {track['index']}")
        samples = _load_index(track["index"])
        rubric = track["rubric"]
        print(
            f"[cs-judge] track={result_path.name} rubric={rubric} indexed={len(samples)}",
            flush=True,
        )
        for line in result_path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("type") != "eval":
                continue
            data = event["data"]
            source = samples.get(data["sample_id"])
            work.append((data, source, rubric))

    full_n = len(work)
    work = _shard_slice(work, args.shard_index, args.num_shards)
    print(f"[cs-judge] shard_n={len(work)} of {full_n}", flush=True)

    judge = SemanticLLMJudgeEvaluator(
        judge_model_name="qwen3-omni-thinking",
        mm_judge_model_name="qwen3-omni-thinking",
        default_template="multimodal",
        n_samples=1,
    )
    results = []
    for n_eval, (data, source, rubric) in enumerate(work, 1):
        sample_id = data["sample_id"]
        if source is None:
            results.append(
                {
                    "sample_id": sample_id,
                    "benchmark": data.get("benchmark_id"),
                    "rubric": rubric,
                    "skipped": 1,
                    "skip_reason": "missing_native_index_row",
                }
            )
            continue
        if n_eval == 1 or n_eval % 5 == 0:
            print(
                f"[cs-judge] sample {n_eval}/{len(work)} id={sample_id} rubric={rubric}",
                flush=True,
            )
        transcript = transcribe_for_judge(
            source["WavPath"],
            lang="mix",
            cache_dir=str(ASR_CACHE),
        )
        scored = judge(
            data.get("pred", ""),
            _as_text(data.get("ref")) or source.get("text") or source.get("answer") or "",
            capability="code_switch",
            rubric=rubric,
            judge_template="multimodal",
            question=source.get("question", ""),
            WavPath=source["WavPath"],
            audio_transcript=transcript,
            require_transcript=True,
            judge_model_name="qwen3-omni-thinking",
        )
        results.append(
            {
                "sample_id": sample_id,
                "benchmark": data.get("benchmark_id"),
                "rubric": rubric,
                **scored,
            }
        )

    if not results and missing_results and full_n == 0:
        raise SystemExit(
            "no smoke predictions found; run Native code_switch smoke first. "
            f"missing={missing_results}"
        )

    if args.out is not None:
        out = args.out
    elif args.num_shards > 1:
        out = OUTPUT / "shards" / f"code-switch-local-llm-judge.shard{args.shard_index}.jsonl"
    else:
        out = OUTPUT / "code-switch-local-llm-judge.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in results),
        encoding="utf-8",
    )
    summary = {
        "records": len(results),
        "output": str(out),
        "missing_result_files": missing_results,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "by_rubric": {},
    }
    for row in results:
        key = row.get("rubric") or "unknown"
        bucket = summary["by_rubric"].setdefault(key, {"n": 0, "skipped": 0})
        bucket["n"] += 1
        if row.get("skipped"):
            bucket["skipped"] += 1
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
