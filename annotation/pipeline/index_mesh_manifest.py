"""Build annotation raw indexes from a mesh unified manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from annotation.pipeline.common import load_jsonl, write_jsonl
from mesh_eval.core.benchmark_map import benchmark_names, get_benchmark


def to_raw_record(sample: dict[str, Any], benchmark: str, subset: str) -> dict[str, Any]:
    reference = sample.get("reference_obj") or sample.get("reference") or {}
    metadata = sample.get("metadata") or {}
    raw_id = metadata.get("source_event_id")
    if raw_id is None:
        raw_id = sample.get("sample_id")
    return {
        "id": raw_id,
        "audio_path": sample.get("WavPath") or (sample.get("input") or {}).get("audio_path", ""),
        "text": reference.get("text") if isinstance(reference, dict) else sample.get("text", ""),
        "subset": subset,
        "split": "test",
        "source_benchmark": benchmark,
        "source_sample_id": sample.get("sample_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="从 mesh manifest 生成标注 raw_index")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", default=[])
    parser.add_argument("--groups", nargs="+", default=[])
    parser.add_argument("--benchmark-map", default="")
    parser.add_argument("--limit-per-dataset", type=int, default=20)
    args = parser.parse_args()

    datasets = list(args.datasets)
    if args.groups:
        datasets.extend(benchmark_names(args.benchmark_map, args.groups))
    if not datasets:
        datasets = benchmark_names(args.benchmark_map, ["current_asr"])
    datasets = list(dict.fromkeys(datasets))

    samples = load_jsonl(args.manifest)
    summary = {}
    for dataset in datasets:
        config = get_benchmark(dataset, args.benchmark_map)
        split = str((config.get("data") or {}).get("split") or "test")
        subset = split.replace("_", "-")
        selected = [
            sample
            for sample in samples
            if sample.get("dataset_id", sample.get("dataset")) == dataset
        ][: args.limit_per_dataset]
        records = [to_raw_record(sample, dataset, subset) for sample in selected]
        output = args.output_dir / dataset / "raw_index.jsonl"
        write_jsonl(output, records)
        summary[dataset] = {"count": len(records), "output": str(output)}

    import json

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

