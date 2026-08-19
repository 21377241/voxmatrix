"""数据集接入：扫描原始数据，生成索引和 Dataset Card 草稿。"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import yaml

from annotation.pipeline.common import MAPPINGS_DIR, load_mapping, write_jsonl


def scan_audio_files(raw_dir: Path) -> list[dict]:
    """扫描常见音频格式，生成最小索引。"""
    extensions = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    records = []
    for idx, audio_path in enumerate(sorted(raw_dir.rglob("*"))):
        if audio_path.suffix.lower() not in extensions:
            continue
        records.append({
            "id": f"{idx:08d}",
            "audio_filepath": str(audio_path.relative_to(raw_dir)),
            "audio_path": str(audio_path),
        })
    return records


def build_dataset_card(dataset_id: str, raw_dir: Path, sample_count: int) -> dict:
    mapping = load_mapping(dataset_id)
    return {
        "dataset_id": dataset_id,
        "display_name": mapping.get("display_name", dataset_id),
        "version": "unknown",
        "language": mapping.get("language", "unknown"),
        "license": mapping.get("license", "unknown"),
        "raw_path": str(raw_dir),
        "ingested_at": date.today().isoformat(),
        "maintainer": "",
        "contributions": mapping.get("contributions", []),
        "confirmed_labels": {
            "scenario": [mapping.get("defaults", {}).get("scenario")],
            "condition": mapping.get("defaults", {}).get("condition", {}),
        },
        "missing_labels": mapping.get("missing_labels", []),
        "formal_ready": mapping.get("auto_complete", False),
        "formal_blockers": [
            m["field"] for m in mapping.get("missing_labels", [])
            if m.get("priority") == "P0"
        ],
        "raw_format": {
            "audio": "wav/flac/mp3",
            "sample_rate": "unknown",
            "annotations": "需后续对齐",
        },
        "mapping_file": str(MAPPINGS_DIR / f"{dataset_id}.yaml"),
        "sample_count": sample_count,
        "notes": mapping.get("notes", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="接入开源数据集")
    parser.add_argument("--dataset", required=True, help="数据集 ID，如 aishell_1")
    parser.add_argument("--raw-dir", required=True, type=Path, help="原始数据目录")
    parser.add_argument("--output", required=True, type=Path, help="工作目录输出路径")
    parser.add_argument("--index-file", default=None, help="已有索引 JSONL（优先于扫描）")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    if args.index_file:
        index_path = Path(args.index_file)
        records = []
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    else:
        records = scan_audio_files(args.raw_dir)

    raw_index_path = args.output / "raw_index.jsonl"
    write_jsonl(raw_index_path, records)

    card = build_dataset_card(args.dataset, args.raw_dir, len(records))
    card_path = args.output / "dataset_card.yaml"
    with open(card_path, "w", encoding="utf-8") as f:
        yaml.dump(card, f, allow_unicode=True, sort_keys=False)

    print(f"接入完成: {len(records)} 条样本")
    print(f"  索引: {raw_index_path}")
    print(f"  Dataset Card: {card_path}")


if __name__ == "__main__":
    main()
