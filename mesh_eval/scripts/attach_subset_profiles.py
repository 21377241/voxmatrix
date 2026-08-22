import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mesh_eval.core.schema_v2 import validate_canonical_sample, validate_json_schema
from mesh_eval.core.subset_manifest import (
    DEFAULT_EXCLUDED_CAPABILITIES,
    SubsetManifestCatalog,
    SubsetManifestError,
    enrich_canonical_sample,
)


def _benchmark_aliases(values):
    aliases = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(
                f"benchmark alias must have ALIAS=MANIFEST_ID form: {value!r}"
            )
        alias, canonical = value.split("=", 1)
        if not alias.strip() or not canonical.strip():
            raise ValueError(f"invalid benchmark alias: {value!r}")
        aliases[alias.strip()] = canonical.strip()
    return aliases


def get_args():
    parser = argparse.ArgumentParser(
        description="Attach subset-level routing metadata to canonical V2 samples"
    )
    parser.add_argument("input_jsonl")
    parser.add_argument("output_jsonl")
    parser.add_argument("--subset-manifest", required=True)
    parser.add_argument("--benchmark-id", default="")
    parser.add_argument("--subset-id", default="")
    parser.add_argument("--source-protocol-id", default="")
    parser.add_argument(
        "--benchmark-alias",
        action="append",
        default=[],
        metavar="ALIAS=MANIFEST_ID",
    )
    parser.add_argument("--exclude-capability", action="append", default=[])
    parser.add_argument("--allow-unmatched", action="store_true")
    parser.add_argument("--overwrite-profile", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    return parser.parse_args()


def attach_subset_profiles(
    input_jsonl,
    output_jsonl,
    *,
    subset_manifest,
    benchmark_id="",
    subset_id="",
    source_protocol_id="",
    benchmark_aliases=None,
    excluded_capabilities=None,
    allow_unmatched=False,
    overwrite_profile=False,
    validate=True,
):
    excluded = set(DEFAULT_EXCLUDED_CAPABILITIES)
    excluded.update(excluded_capabilities or [])
    catalog = SubsetManifestCatalog(
        subset_manifest,
        excluded_capabilities=excluded,
        benchmark_aliases=benchmark_aliases,
    )
    source = Path(input_jsonl).expanduser().resolve()
    target = Path(output_jsonl).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    counts = {"input": 0, "written": 0, "excluded": 0, "unmatched": 0}
    try:
        with source.open(encoding="utf-8") as input_handle, os.fdopen(
            fd, "w", encoding="utf-8"
        ) as output_handle:
            for line_no, line in enumerate(input_handle, start=1):
                if not line.strip():
                    continue
                counts["input"] += 1
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{source}:{line_no}: invalid JSON: {exc}") from exc
                capability = str(sample.get("capability") or "")
                if capability in excluded:
                    counts["excluded"] += 1
                    continue
                try:
                    record = catalog.find_for_sample(
                        sample,
                        benchmark_id=benchmark_id,
                        subset_id=subset_id,
                        source_protocol_id=source_protocol_id,
                    )
                    enriched = enrich_canonical_sample(
                        sample,
                        record,
                        catalog,
                        overwrite_profile=overwrite_profile,
                    )
                except SubsetManifestError:
                    if not allow_unmatched:
                        raise
                    enriched = sample
                    counts["unmatched"] += 1
                if validate and enriched.get("schema_version") == "2.0":
                    errors = validate_canonical_sample(enriched, formal=False)
                    errors.extend(validate_json_schema(enriched, "canonical"))
                    if errors:
                        raise ValueError(
                            f"{source}:{line_no}: enriched sample is invalid: "
                            + "; ".join(errors)
                        )
                output_handle.write(
                    json.dumps(enriched, ensure_ascii=False, sort_keys=True) + "\n"
                )
                counts["written"] += 1
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return {
        **counts,
        "catalog_records": len(catalog),
        "catalog_excluded_records": catalog.excluded_record_count,
        "catalog_sha256": catalog.sha256,
        "output": str(target),
    }


def main():
    args = get_args()
    result = attach_subset_profiles(
        args.input_jsonl,
        args.output_jsonl,
        subset_manifest=args.subset_manifest,
        benchmark_id=args.benchmark_id,
        subset_id=args.subset_id,
        source_protocol_id=args.source_protocol_id,
        benchmark_aliases=_benchmark_aliases(args.benchmark_alias),
        excluded_capabilities=args.exclude_capability,
        allow_unmatched=args.allow_unmatched,
        overwrite_profile=args.overwrite_profile,
        validate=not args.no_validate,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
