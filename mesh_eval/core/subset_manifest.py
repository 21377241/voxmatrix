"""Subset-level benchmark routing metadata and canonical-sample enrichment."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


SUBSET_MANIFEST_VERSION = "2.0.0"
DEFAULT_EXCLUDED_CAPABILITIES = frozenset(
    {"meeting_summary", "todo_extraction"}
)
SPLIT_ALIASES = {
    "validation": "dev",
    "valid": "dev",
    "development": "dev",
    "eval": "test",
    "evaluation": "test",
}
CANONICAL_SPLITS = frozenset({"train", "dev", "test", "heldout"})
MAPPING_STATUSES = frozenset({"direct", "adaptable", "insufficient"})
RESOURCE_STATUSES = frozenset({"available", "unavailable", "gated"})
SUBSET_PROFILE_RUNTIME_FIELDS = {
    "record_id": "subset_manifest_record_id",
    "catalog_schema_version": "subset_manifest_schema_version",
    "catalog_sha256": "subset_manifest_sha256",
    "source_benchmark_id": "source_benchmark_id",
    "subset_id": "subset_id",
    "source_protocol_id": "source_protocol_id",
    "source_metrics": "source_metric_names",
    "mapping_status": "mapping_status",
    "resource_status": "resource_status",
    "annotation_confidence": "annotation_confidence",
}


class SubsetManifestError(ValueError):
    """Raised when subset routing metadata is invalid or ambiguous."""


def normalize_subset_split(value: Any) -> str:
    split = str(value or "").strip().lower()
    split = SPLIT_ALIASES.get(split, split)
    if split not in CANONICAL_SPLITS:
        raise SubsetManifestError(f"unsupported subset split: {value!r}")
    return split


def flatten_subset_profile(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    flattened: Dict[str, Any] = {}
    for source, target in SUBSET_PROFILE_RUNTIME_FIELDS.items():
        item = value.get(source)
        if item in (None, "", []):
            continue
        flattened[target] = (
            "|".join(str(part) for part in item)
            if isinstance(item, list)
            else item
        )
    return flattened


def _required_text(value: Any, field: str, location: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise SubsetManifestError(f"{location}: missing {field}")
    return text


def _join_key(record: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    benchmark_id = str((record.get("benchmark") or {}).get("id") or "")
    subset = record.get("subset") or {}
    subset_id = str(subset.get("id") or "")
    split = normalize_subset_split(subset.get("split"))
    capability = str((record.get("framework") or {}).get("capability") or "")
    return benchmark_id, subset_id, split, capability


def _protocol_join_key(
    record: Mapping[str, Any],
) -> Tuple[str, str, str, str, str]:
    return (*_join_key(record), str((record.get("protocol") or {}).get("id") or ""))


def _validate_record(record: Any, location: str) -> Dict[str, Any]:
    if not isinstance(record, dict):
        raise SubsetManifestError(f"{location}: record must be an object")
    if record.get("schema_version") != SUBSET_MANIFEST_VERSION:
        raise SubsetManifestError(
            f"{location}: unsupported schema_version {record.get('schema_version')!r}"
        )

    _required_text(record.get("record_id"), "record_id", location)
    benchmark = record.get("benchmark") or {}
    _required_text(benchmark.get("id"), "benchmark.id", location)
    resource_status = _required_text(
        benchmark.get("resource_status"), "benchmark.resource_status", location
    )
    if resource_status not in RESOURCE_STATUSES:
        raise SubsetManifestError(
            f"{location}: unsupported benchmark.resource_status {resource_status!r}"
        )

    framework = record.get("framework") or {}
    _required_text(framework.get("task"), "framework.task", location)
    _required_text(framework.get("capability"), "framework.capability", location)
    mapping_status = _required_text(
        framework.get("mapping_status"), "framework.mapping_status", location
    )
    if mapping_status not in MAPPING_STATUSES:
        raise SubsetManifestError(
            f"{location}: unsupported framework.mapping_status {mapping_status!r}"
        )

    scenario = record.get("scenario") or {}
    _required_text(scenario.get("primary"), "scenario.primary", location)
    _required_text(scenario.get("secondary"), "scenario.secondary", location)

    protocol = record.get("protocol") or {}
    protocol_id = _required_text(protocol.get("id"), "protocol.id", location)
    metrics = protocol.get("metrics")
    if not isinstance(metrics, list) or not all(
        isinstance(item, str) and item.strip() for item in metrics
    ):
        raise SubsetManifestError(f"{location}: protocol.metrics must be a string list")

    subset = record.get("subset") or {}
    _required_text(subset.get("id"), "subset.id", location)
    split = normalize_subset_split(subset.get("split"))
    selector = subset.get("selector") or {}
    _required_text(selector.get("mode"), "subset.selector.mode", location)
    selector_split = normalize_subset_split(selector.get("split"))
    if split != selector_split:
        raise SubsetManifestError(
            f"{location}: subset.split {split!r} differs from selector.split "
            f"{selector_split!r}"
        )
    protocol_ids = selector.get("protocol_ids")
    if not isinstance(protocol_ids, list) or protocol_id not in protocol_ids:
        raise SubsetManifestError(
            f"{location}: protocol.id {protocol_id!r} is not declared by "
            "subset.selector.protocol_ids"
        )
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise SubsetManifestError(f"{location}: metadata must be an object")
    return copy.deepcopy(record)


class SubsetManifestCatalog:
    """Strict, immutable lookup view over a subset-level JSONL manifest."""

    def __init__(
        self,
        path: str,
        *,
        excluded_capabilities: Optional[Iterable[str]] = None,
        benchmark_aliases: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.path = str(Path(path).expanduser().resolve())
        self.excluded_capabilities = frozenset(
            str(item)
            for item in (
                DEFAULT_EXCLUDED_CAPABILITIES
                if excluded_capabilities is None
                else excluded_capabilities
            )
        )
        self.benchmark_aliases = {
            str(alias): str(canonical)
            for alias, canonical in (benchmark_aliases or {}).items()
        }
        source = Path(self.path)
        if not source.is_file():
            raise FileNotFoundError(self.path)
        self.sha256 = hashlib.sha256(source.read_bytes()).hexdigest()

        records: List[Dict[str, Any]] = []
        by_record_id: Dict[str, Dict[str, Any]] = {}
        by_join_key: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        by_protocol_key: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}
        self.total_record_count = 0
        self.excluded_record_count = 0
        with source.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                self.total_record_count += 1
                location = f"{self.path}:{line_no}"
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SubsetManifestError(f"{location}: invalid JSON: {exc}") from exc
                record = _validate_record(raw, location)
                capability = record["framework"]["capability"]
                if capability in self.excluded_capabilities:
                    self.excluded_record_count += 1
                    continue

                record_id = record["record_id"]
                if record_id in by_record_id:
                    raise SubsetManifestError(
                        f"{location}: duplicate record_id {record_id!r}"
                    )
                join_key = _join_key(record)
                if join_key in by_join_key:
                    previous = by_join_key[join_key]["record_id"]
                    raise SubsetManifestError(
                        f"{location}: ambiguous subset join key {join_key!r}; "
                        f"already used by {previous!r}"
                    )
                protocol_key = _protocol_join_key(record)
                if protocol_key in by_protocol_key:
                    previous = by_protocol_key[protocol_key]["record_id"]
                    raise SubsetManifestError(
                        f"{location}: duplicate protocol join key {protocol_key!r}; "
                        f"already used by {previous!r}"
                    )
                records.append(record)
                by_record_id[record_id] = record
                by_join_key[join_key] = record
                by_protocol_key[protocol_key] = record

        self._records = tuple(records)
        self._by_record_id = by_record_id
        self._by_join_key = by_join_key
        self._by_protocol_key = by_protocol_key

    @property
    def records(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(list(self._records))

    def __len__(self) -> int:
        return len(self._records)

    def _canonical_benchmark_id(self, value: Any) -> str:
        benchmark_id = str(value or "").strip()
        return self.benchmark_aliases.get(benchmark_id, benchmark_id)

    def find(
        self,
        *,
        record_id: str = "",
        benchmark_id: str = "",
        subset_id: str = "",
        split: str = "",
        capability: str = "",
        source_protocol_id: str = "",
    ) -> Dict[str, Any]:
        if record_id:
            record = self._by_record_id.get(str(record_id))
            if record is None:
                raise SubsetManifestError(
                    f"subset manifest record_id is not mapped: {record_id!r}"
                )
            return copy.deepcopy(record)

        key = (
            self._canonical_benchmark_id(benchmark_id),
            str(subset_id or "").strip(),
            normalize_subset_split(split),
            str(capability or "").strip(),
        )
        if not all(key):
            raise SubsetManifestError(f"incomplete subset join key: {key!r}")
        if source_protocol_id:
            protocol_key = (*key, str(source_protocol_id).strip())
            record = self._by_protocol_key.get(protocol_key)
        else:
            record = self._by_join_key.get(key)
        if record is None:
            suffix = (
                f", source_protocol_id={source_protocol_id!r}"
                if source_protocol_id
                else ""
            )
            raise SubsetManifestError(
                f"subset manifest join key is not mapped: {key!r}{suffix}"
            )
        return copy.deepcopy(record)

    def find_for_sample(
        self,
        sample: Mapping[str, Any],
        *,
        benchmark_id: str = "",
        subset_id: str = "",
        source_protocol_id: str = "",
    ) -> Dict[str, Any]:
        profile = sample.get("subset_profile") or {}
        if not isinstance(profile, dict):
            raise SubsetManifestError("sample.subset_profile must be an object")
        record_id = str(
            profile.get("record_id")
            or sample.get("subset_manifest_record_id")
            or ""
        )
        if record_id:
            return self.find(record_id=record_id)
        return self.find(
            benchmark_id=(
                benchmark_id
                or profile.get("source_benchmark_id")
                or sample.get("source_benchmark_id")
                or sample.get("benchmark_id")
                or sample.get("dataset")
                or ""
            ),
            subset_id=(
                subset_id
                or profile.get("subset_id")
                or sample.get("subset_id")
                or sample.get("subset")
                or ""
            ),
            split=str(sample.get("split") or profile.get("split") or ""),
            capability=str(sample.get("capability") or ""),
            source_protocol_id=(
                source_protocol_id
                or profile.get("source_protocol_id")
                or sample.get("source_protocol_id")
                or ""
            ),
        )

    def profile(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        validated = _validate_record(record, str(record.get("record_id") or "record"))
        return {
            "record_id": validated["record_id"],
            "catalog_schema_version": validated["schema_version"],
            "catalog_sha256": self.sha256,
            "source_benchmark_id": validated["benchmark"]["id"],
            "subset_id": validated["subset"]["id"],
            "split": normalize_subset_split(validated["subset"]["split"]),
            "source_protocol_id": validated["protocol"]["id"],
            "source_metrics": list(validated["protocol"].get("metrics") or []),
            "mapping_status": validated["framework"]["mapping_status"],
            "resource_status": validated["benchmark"]["resource_status"],
            "annotation_confidence": (validated.get("provenance") or {}).get(
                "annotation_confidence"
            ),
        }


def _is_missing(value: Any) -> bool:
    return value in (None, "", [], "unknown")


def _documented_provenance(record_id: str) -> Dict[str, Any]:
    return {
        "source": "subset_documented",
        "evidence": f"benchmark_subset_manifest:{record_id}",
    }


def _metadata_target_field(field: str, value: Any) -> str:
    """Keep manifest-only vocabulary without widening canonical enums."""

    from mesh_eval.core.schema_v2 import load_taxonomy_v2

    allowed = (
        ((load_taxonomy_v2().get("metadata") or {}).get("enums") or {}).get(field)
    )
    if not allowed:
        return field
    values = value if isinstance(value, list) else [value]
    if any(item not in (None, "") and item not in allowed for item in values):
        return f"{field}_subtype"
    return field


def enrich_canonical_sample(
    sample: Mapping[str, Any],
    record: Mapping[str, Any],
    catalog: SubsetManifestCatalog,
    *,
    overwrite_profile: bool = False,
) -> Dict[str, Any]:
    """Attach one exact subset record without overriding sample-native evidence."""

    if sample.get("schema_version") != "2.0":
        raise SubsetManifestError(
            "subset enrichment requires canonical schema_version='2.0'"
        )
    result = copy.deepcopy(dict(sample))
    profile = catalog.profile(record)
    existing_profile = result.get("subset_profile") or {}
    if not isinstance(existing_profile, dict):
        raise SubsetManifestError("sample.subset_profile must be an object")
    existing_id = str(existing_profile.get("record_id") or "")
    if existing_id and existing_id != profile["record_id"] and not overwrite_profile:
        raise SubsetManifestError(
            f"sample subset_profile points to {existing_id!r}, not "
            f"{profile['record_id']!r}"
        )

    framework = record["framework"]
    for field in ("task", "capability"):
        current = str(result.get(field) or "")
        documented = str(framework[field])
        if current and current != documented:
            raise SubsetManifestError(
                f"sample {field} {current!r} conflicts with subset record "
                f"{documented!r}"
            )
        result[field] = documented
    sample_split = normalize_subset_split(result.get("split"))
    if sample_split != profile["split"]:
        raise SubsetManifestError(
            f"sample split {sample_split!r} conflicts with subset record "
            f"{profile['split']!r}"
        )
    result["split"] = sample_split
    result["subset_profile"] = profile

    scenario = copy.deepcopy(result.get("scenario") or {})
    metadata = copy.deepcopy(result.get("metadata") or {})
    provenance = copy.deepcopy(result.get("provenance") or {})
    metadata_provenance = copy.deepcopy(provenance.get("metadata") or {})
    documented_scenario = record.get("scenario") or {}
    record_id = profile["record_id"]
    primary = str(documented_scenario.get("primary") or "unknown")
    secondary = str(documented_scenario.get("secondary") or "unknown")

    if _is_missing(scenario.get("primary")) and primary != "unknown":
        scenario["primary"] = primary
        provenance["scenario"] = _documented_provenance(record_id)
    scenario.setdefault("primary", "unknown")

    from mesh_eval.core.schema_v2 import load_taxonomy_v2

    allowed_secondary = (
        ((load_taxonomy_v2().get("scenario") or {}).get("secondary") or {}).get(
            scenario["primary"], []
        )
    )
    if secondary in allowed_secondary:
        if _is_missing(scenario.get("secondary")) and secondary != "unknown":
            scenario["secondary"] = secondary
            provenance["scenario"] = _documented_provenance(record_id)
    elif secondary != "unknown" and _is_missing(metadata.get("scenario_subtype")):
        metadata["scenario_subtype"] = secondary
        metadata_provenance["scenario_subtype"] = _documented_provenance(record_id)
    scenario.setdefault("secondary", "unknown")

    existing_speaker_count = metadata.get("speaker_count")
    if isinstance(existing_speaker_count, str):
        if _is_missing(metadata.get("speaker_count_bucket")):
            metadata["speaker_count_bucket"] = existing_speaker_count
            if metadata_provenance.get("speaker_count"):
                metadata_provenance["speaker_count_bucket"] = copy.deepcopy(
                    metadata_provenance["speaker_count"]
                )
        metadata["speaker_count"] = None

    for field, value in (record.get("metadata") or {}).items():
        if field == "speaker_count":
            if isinstance(value, int) and not isinstance(value, bool):
                if _is_missing(metadata.get("speaker_count")):
                    metadata["speaker_count"] = value
                    metadata_provenance[field] = _documented_provenance(record_id)
            elif value not in (None, "") and _is_missing(
                metadata.get("speaker_count_bucket")
            ):
                metadata["speaker_count_bucket"] = str(value)
                metadata_provenance["speaker_count_bucket"] = (
                    _documented_provenance(record_id)
                )
            continue
        target_field = _metadata_target_field(field, value)
        if not _is_missing(value) and _is_missing(metadata.get(target_field)):
            metadata[target_field] = copy.deepcopy(value)
            metadata_provenance[target_field] = _documented_provenance(record_id)

    provenance["metadata"] = metadata_provenance
    result["scenario"] = scenario
    result["metadata"] = metadata
    result["provenance"] = provenance
    return result
