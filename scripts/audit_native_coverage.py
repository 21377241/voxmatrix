#!/usr/bin/env python3
"""Audit the annotation -> native catalog -> native index coverage.

This is intentionally read-only.  It does not build, rewrite, or register
evaluation data.  It reports gaps between the annotation baseline, the native
catalog, and the JSONL files used by the native suite.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "annotation" / "benchmark_subset_manifest.jsonl"
DEFAULT_CATALOG = ROOT / "suites" / "native" / "catalog.yaml"
DEFAULT_NATIVE_DIR = ROOT / "mesh_eval" / "data" / "native"
DEFAULT_DATASET_REGISTRY = ROOT / "mesh_eval" / "registry" / "dataset" / "unified_speech.yaml"


def load_manifest(path: Path) -> dict[str, dict]:
    records = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        record_id = item.get("record_id")
        if not record_id:
            raise ValueError(f"manifest line {line_no}: missing record_id")
        if record_id in records:
            raise ValueError(f"manifest line {line_no}: duplicate record_id {record_id}")
        records[record_id] = item
    return records


def load_catalog(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(data.get("capabilities") or [])


def load_dataset_registry(path: Path) -> dict[str, dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def expected_index(entry: dict, dataset_registry: dict[str, dict]) -> Path | None:
    dataset = entry.get("dataset")
    if not dataset:
        return None
    # Local NativeClipDataset entries declare the authoritative f_name.  An
    # external benchmark entry has no local JSONL and is intentionally skipped.
    args = (dataset_registry.get(dataset) or {}).get("args") or {}
    filename = args.get("f_name")
    return ROOT / filename if filename else None


def jsonl_stats(path: Path) -> tuple[int, Counter, list[str]]:
    count = 0
    capabilities = Counter()
    datasets = Counter()
    errors = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON ({exc.msg})")
            continue
        count += 1
        if item.get("capability"):
            capabilities[item["capability"]] += 1
        if item.get("dataset"):
            datasets[item["dataset"]] += 1
    return count, capabilities, [*errors, *[f"dataset={k}: {v}" for k, v in datasets.items()]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--native-dir", type=Path, default=DEFAULT_NATIVE_DIR)
    parser.add_argument("--dataset-registry", type=Path, default=DEFAULT_DATASET_REGISTRY)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    catalog = load_catalog(args.catalog)
    dataset_registry = load_dataset_registry(args.dataset_registry)

    manifest_by_cap_bench = defaultdict(list)
    for record_id, item in manifest.items():
        framework = item.get("framework") or {}
        benchmark = (item.get("benchmark") or {}).get("id")
        capability = framework.get("capability")
        if capability and benchmark:
            manifest_by_cap_bench[(capability, benchmark)].append(record_id)

    catalog_keys = Counter()
    catalog_missing_manifest = []
    catalog_missing_index = []
    catalog_index_stats = []
    for entry in catalog:
        capability = entry.get("capability")
        benchmark = entry.get("benchmark")
        key = (capability, benchmark)
        catalog_keys[key] += 1
        record_id = entry.get("subset_record_id")
        if record_id and record_id not in manifest:
            catalog_missing_manifest.append((entry.get("id"), record_id))

        index = expected_index(entry, dataset_registry)
        if entry.get("status") != "blocked" and index is not None:
            # Use the caller's directory when checking, preserving the same
            # filename convention as expected_index.
            if not index.exists():
                catalog_missing_index.append((entry.get("id"), str(index)))
            else:
                count, capabilities, details = jsonl_stats(index)
                catalog_index_stats.append((entry, index, count, capabilities, details))

    duplicate_catalog = [key for key, count in catalog_keys.items() if count > 1]
    manifest_without_catalog = sorted(set(manifest_by_cap_bench) - set(catalog_keys))
    native_files = sorted(args.native_dir.glob("*.jsonl")) if args.native_dir.exists() else []
    referenced_files = {index for _, index, *_ in catalog_index_stats}
    orphan_native = sorted(set(native_files) - referenced_files)

    print("# VoxMatrix Native coverage audit")
    print(f"manifest_records: {len(manifest)}")
    print(f"manifest_capability_benchmark_pairs: {len(manifest_by_cap_bench)}")
    print(f"catalog_entries: {len(catalog)}")
    print(f"catalog_unique_capability_benchmark_pairs: {len(catalog_keys)}")
    print(f"native_jsonl_files: {len(native_files)}")
    print()

    print("## Manifest capabilities")
    for capability, count in sorted(Counter(k[0] for k in manifest_by_cap_bench for _ in [0]).items()):
        pairs = sum(1 for key in manifest_by_cap_bench if key[0] == capability)
        print(f"- {capability}: {pairs} capability/benchmark pairs")
    print()

    print("## Catalog status")
    for status, count in sorted(Counter(entry.get("status", "<missing>") for entry in catalog).items()):
        print(f"- {status}: {count}")
    print()

    print("## Native index sample counts")
    for entry, index, count, capabilities, details in sorted(catalog_index_stats, key=lambda x: x[0].get("id", "")):
        cap_text = ", ".join(f"{k}={v}" for k, v in capabilities.items()) or "<missing>"
        print(f"- {entry.get('id')}: {count} samples ({cap_text}) [{index.relative_to(ROOT)}]")
    print()

    def section(title: str, rows: list, formatter=str) -> None:
        print(f"## {title}: {len(rows)}")
        for row in rows:
            print(f"- {formatter(row)}")
        print()

    section("Manifest pairs without catalog entry", manifest_without_catalog, lambda x: f"{x[0]} / {x[1]}")
    section("Catalog entries with missing manifest record", catalog_missing_manifest, lambda x: f"{x[0]} -> {x[1]}")
    section("Executable catalog entries with missing native index", catalog_missing_index, lambda x: f"{x[0]} -> {x[1]}")
    section("Duplicate catalog capability/benchmark pairs", duplicate_catalog, lambda x: f"{x[0]} / {x[1]}")
    section("Unreferenced native JSONL files", orphan_native, lambda x: str(x.relative_to(ROOT)))

    return 1 if (catalog_missing_manifest or catalog_missing_index or duplicate_catalog) else 0


if __name__ == "__main__":
    raise SystemExit(main())
