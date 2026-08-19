"""Offline, non-destructive V1 → canonical EvaluationSample V2 migration."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from annotation.pipeline.common import load_jsonl
from mesh_eval.core.schema_v2 import (
    V2MigrationError,
    adapt_v1_to_v2,
    validate_canonical_sample,
    validate_json_schema,
)


def migrate_records(records: list[dict], ref_col: str = "text") -> tuple[list[dict], dict]:
    migrated = []
    warning_count = 0
    errors = []
    for index, record in enumerate(records):
        try:
            sample = adapt_v1_to_v2(record, ref_col=ref_col)
            sample_errors = validate_canonical_sample(sample, formal=False)
            sample_errors.extend(validate_json_schema(sample, "canonical"))
            if sample_errors:
                errors.extend(f"sample[{index}]: {item}" for item in sample_errors)
                continue
            warning_count += len(sample.get("conversion_warnings") or [])
            migrated.append(sample)
        except (V2MigrationError, ValueError, TypeError) as exc:
            errors.append(f"sample[{index}]: {exc}")
    report = {
        "input_count": len(records),
        "output_count": len(migrated),
        "warning_count": warning_count,
        "error_count": len(errors),
        "errors": errors,
        "adapter_version": "v1-to-v2@1",
    }
    return migrated, report


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate a V1 manifest to canonical V2")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--ref-col", default="text")
    args = parser.parse_args()
    rows, report = migrate_records(load_jsonl(args.input), args.ref_col)
    if report["error_count"]:
        detail = "\n".join(report["errors"][:20])
        raise SystemExit(f"V2 migration failed with {report['error_count']} errors:\n{detail}")
    _atomic_jsonl(args.output, rows)
    report_path = args.report or args.output.with_suffix(".migration.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
