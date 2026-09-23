#!/usr/bin/env python3
"""Run the local multimodal Translation LLM Judge on saved predictions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from audio_evals.evaluator.asr_for_judge import transcribe_for_judge
from audio_evals.evaluator.semantic_llm_judge import SemanticLLMJudgeEvaluator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/native_translation_smoke"
ASR_CACHE = OUTPUT / "judge_asr_cache"
RESULT_FILES = sorted(OUTPUT.glob("native-*-smoke.jsonl"))
INDEXES = [
    ROOT / "mesh_eval/data/native/speech_translation_covost2_zh_en.jsonl",
    ROOT / "mesh_eval/data/native/speech_translation_covost2_en_zh.jsonl",
    ROOT / "mesh_eval/data/native/speech_translation_mmsu.jsonl",
]


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

    print(f"[translation-judge] start ROOT={ROOT}", flush=True)
    print(
        f"[translation-judge] shard={args.shard_index}/{args.num_shards} "
        f"result_files={[str(p.name) for p in RESULT_FILES]}",
        flush=True,
    )
    samples = {}
    for path in INDEXES:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                samples[row["sample_id"]] = row
    print(f"[translation-judge] indexed_samples={len(samples)}", flush=True)

    work = []
    for result_file in RESULT_FILES:
        for line in result_file.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("type") != "eval":
                continue
            work.append(event["data"])
    full_n = len(work)
    work = _shard_slice(work, args.shard_index, args.num_shards)
    print(f"[translation-judge] shard_n={len(work)} of {full_n}", flush=True)

    judge = SemanticLLMJudgeEvaluator(
        judge_model_name="qwen3-omni-thinking",
        mm_judge_model_name="qwen3-omni-thinking",
        default_template="multimodal",
        n_samples=1,
    )
    results = []
    for n_eval, data in enumerate(work, 1):
        source = samples[data["sample_id"]]
        if n_eval == 1 or n_eval % 5 == 0:
            print(
                f"[translation-judge] sample {n_eval}/{len(work)} id={data['sample_id']}",
                flush=True,
            )
        transcript = transcribe_for_judge(
            source["WavPath"],
            lang=source.get("source_lang"),
            cache_dir=str(ASR_CACHE),
        )
        scored = judge(
            data.get("pred", ""),
            data.get("ref", ""),
            capability="speech_translation",
            rubric="translation",
            judge_template="multimodal",
            question=source.get("question", ""),
            direction=(
                "Translate into Chinese"
                if source.get("target_lang") == "zh-CN"
                else "Translate into English"
            ),
            WavPath=source["WavPath"],
            audio_transcript=transcript,
            require_transcript=True,
            judge_model_name="qwen3-omni-thinking",
        )
        results.append(
            {
                "sample_id": data["sample_id"],
                "benchmark": data.get("benchmark_id"),
                **scored,
            }
        )

    if args.out is not None:
        out = args.out
    elif args.num_shards > 1:
        out = OUTPUT / "shards" / f"translation-local-llm-judge.shard{args.shard_index}.jsonl"
    else:
        out = OUTPUT / "translation-local-llm-judge.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in results),
        encoding="utf-8",
    )
    skipped = sum(1 for r in results if r.get("skipped"))
    print(
        json.dumps(
            {
                "records": len(results),
                "skipped": skipped,
                "output": str(out),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
